# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Assemble docs/config.md from the per-subsystem config README.md files.

The top-level docs/config.md keeps a hand-written preamble (overview, full
end-to-end examples) and a generated region, delimited by the BEGIN/END markers
below, into which each top-level config block's colocated README is transcluded.
The READMEs next to each schema are the source of truth; this script keeps the
single-page guide in sync, the same way scripts/sync_cli_flags_doc.py keeps the
CLI flag table in sync.

Invariant enforced: every config subdirectory that defines a schema (contains a
config.py) is either expanded as a top-level section here (SUBSYSTEMS) or
explicitly listed as nested elsewhere (NESTED). Adding a new config subdirectory
forces a deliberate choice, so no config surface silently goes undocumented.

Usage:
    python scripts/sync_config_doc.py            # rewrite docs/config.md
    python scripts/sync_config_doc.py --check    # fail if out of sync
"""

import argparse
import enum
import inspect
import os
import re
import sys
import types
import typing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from inference_perf.config.config import Config

CONFIG_ROOT = Path("inference_perf/config")
DOC_PATH = Path("docs/config.md")
DOC_DIR = DOC_PATH.parent

BEGIN = "<!-- BEGIN GENERATED: config subsystems. Do not edit by hand; edit the per-subdirectory README.md files and run `pdm run update:config-doc`. -->"
END = "<!-- END GENERATED: config subsystems -->"

# Top-level config blocks, in display order: (yaml_key, subdir, section_title).
# Each subdir must contain a README.md. The order matches the root Config model.
SUBSYSTEMS: List[Tuple[str, str, str]] = [
    ("api", "apis", "API (`api`)"),
    ("data", "datagen", "Data Generation (`data`)"),
    ("load", "loadgen", "Load Generation (`load`)"),
    ("server", "client/modelserver", "Model Server (`server`)"),
    ("metrics", "metrics", "Metrics (`metrics`)"),
    ("report", "reportgen", "Reporting (`report`)"),
    ("storage", "client/filestorage", "Storage (`storage`)"),
    ("tokenizer", "utils", "Tokenizer (`tokenizer`)"),
]

# Schema-bearing config subdirs intentionally NOT expanded as their own section
# because they are nested inside another block. Value is the rationale.
NESTED: Dict[str, str] = {
    "client/server_metrics": "PrometheusClientConfig is the metrics.prometheus sub-block; documented in the metrics section.",
}

# Defaults that are non-deterministic (host- or time-dependent) and so must be
# rendered as a stable label rather than their runtime value, keyed by
# (model name, field name).
DEFAULT_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("LoadConfig", "num_workers"): "CPU core count",
    ("LoadConfig", "base_seed"): "per-run timestamp (ms)",
    # `path` (inherited by every storage backend) defaults to a per-run timestamp.
    ("StorageConfigBase", "path"): "`reports-<timestamp>`",
    ("GoogleCloudStorageConfig", "path"): "`reports-<timestamp>`",
    ("SimpleStorageServiceConfig", "path"): "`reports-<timestamp>`",
    ("StorageConfig", "local_storage"): "default block (always on)",
}


def _collect_models(root: type) -> Dict[str, type]:
    """Map every Pydantic model reachable from ``root`` by class name."""
    seen: set = set()
    out: Dict[str, type] = {}
    stack = [root]
    while stack:
        model = stack.pop()
        if model in seen or not (inspect.isclass(model) and issubclass(model, BaseModel)):
            continue
        seen.add(model)
        out[model.__name__] = model
        # Base models (e.g. the shared MediaDatagenConfig) may be documented even
        # when no field is typed as them directly.
        for base in model.__bases__:
            if inspect.isclass(base) and issubclass(base, BaseModel) and base is not BaseModel:
                stack.append(base)
        for field in model.model_fields.values():
            queue = [field.annotation]
            while queue:
                tp = queue.pop()
                queue.extend(typing.get_args(tp))
                if inspect.isclass(tp) and issubclass(tp, BaseModel):
                    stack.append(tp)
    return out


MODELS = _collect_models(Config)


def _base_type(annotation: object) -> str:
    """Render a type annotation as a short, human-readable token."""
    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if origin is typing.Union or origin is types.UnionType:
        parts: List[str] = []
        for a in args:
            token = _base_type(a)
            if token not in parts:
                parts.append(token)
        return " or ".join(parts)
    if origin is list:
        inner = _base_type(args[0]) if args else ""
        return f"list[{inner}]" if inner and inner != "object" else "list"
    if origin is dict:
        return "dict"
    if origin is typing.Literal:
        return "enum"
    if inspect.isclass(annotation):
        if issubclass(annotation, enum.Enum):
            return "enum"
        if issubclass(annotation, BaseModel):
            return annotation.__name__
        for prim, label in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
            if annotation is prim:
                return label
        if "Url" in annotation.__name__:
            return "URL"
        return annotation.__name__
    return getattr(annotation, "__name__", str(annotation))


def render_type(field: FieldInfo) -> str:
    """Type token plus any numeric constraints (e.g. `int > 0`)."""
    token = _base_type(field.annotation)
    constraints: List[str] = []
    for meta in field.metadata:
        for attr, sym in (("gt", ">"), ("ge", "≥"), ("lt", "<"), ("le", "≤")):
            value = getattr(meta, attr, None)
            if value is not None:
                constraints.append(f"{sym} {value}")
    return f"{token} {', '.join(constraints)}" if constraints else token


def render_default(model_name: str, field_name: str, field: FieldInfo) -> str:
    if (model_name, field_name) in DEFAULT_OVERRIDES:
        return DEFAULT_OVERRIDES[(model_name, field_name)]
    if field.is_required():
        return "(required)"
    if field.default_factory is not None:
        return "(computed)"
    value = field.default
    if value is None:
        return "`null`"
    if isinstance(value, enum.Enum):
        return f"`{value.value}`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, str):
        return f'`"{value}"`' if value else '`""`'
    if isinstance(value, (int, float)):
        return f"`{value}`"
    if isinstance(value, list):
        return "`[]`" if not value else f"`{value}`"
    if isinstance(value, dict):
        return "`{}`" if not value else f"`{value}`"
    if isinstance(value, BaseModel):
        return "default block"
    return f"`{value}`"


def _description(field: FieldInfo) -> str:
    return (field.description or "").replace("\n", " ").replace("|", "\\|").strip()


def render_field_table(model_name: str, fields: Optional[List[str]]) -> str:
    """A Markdown field table for ``model_name``, optionally limited to ``fields``."""
    model = MODELS.get(model_name)
    if model is None:
        raise SystemExit(f"FIELDS marker references unknown config model '{model_name}'.")
    model_fields = model.model_fields
    names = fields if fields else list(model_fields.keys())
    rows = ["| Field | Type | Default | Description |", "| --- | --- | --- | --- |"]
    for name in names:
        if name not in model_fields:
            raise SystemExit(f"FIELDS {model_name}: unknown field '{name}'.")
        field = model_fields[name]
        key = field.alias or name
        rows.append(f"| `{key}` | {render_type(field)} | {render_default(model_name, name, field)} | {_description(field)} |")
    return "\n".join(rows)


_FIELDS_MARKER = re.compile(r"<!-- FIELDS:\s*(?P<spec>.*?)\s*-->.*?<!-- /FIELDS -->", re.S)


def fill_field_markers(text: str) -> str:
    """Replace each ``<!-- FIELDS: Model[: f1, f2] -->...<!-- /FIELDS -->`` block
    with the generated table for that model."""

    def repl(match: "re.Match[str]") -> str:
        spec = match.group("spec")
        model_name, _, field_str = spec.partition(":")
        fields = [f.strip() for f in field_str.split(",") if f.strip()] if field_str.strip() else None
        table = render_field_table(model_name.strip(), fields)
        return f"<!-- FIELDS: {spec} -->\n\n{table}\n\n<!-- /FIELDS -->"

    return _FIELDS_MARKER.sub(repl, text)


def fill_all_readmes(write: bool) -> List[str]:
    """Fill the FIELDS markers in every subsystem README. Returns the subdirs
    whose on-disk content does not match (only meaningful when write=False)."""
    drift = []
    for _, subdir, _ in SUBSYSTEMS:
        path = CONFIG_ROOT / subdir / "README.md"
        current = path.read_text(encoding="utf-8")
        filled = fill_field_markers(current)
        if filled != current:
            if write:
                path.write_text(filled, encoding="utf-8")
            else:
                drift.append(subdir)
    return drift


def discover_config_dirs() -> List[str]:
    """Posix relative paths (under CONFIG_ROOT) of every subdir with a config.py."""
    dirs = []
    for path in CONFIG_ROOT.rglob("config.py"):
        rel = path.parent.relative_to(CONFIG_ROOT).as_posix()
        if rel != ".":  # the root config/config.py is the aggregator, not a block
            dirs.append(rel)
    return sorted(dirs)


def enforce_coverage() -> None:
    """Fail if any schema-bearing subdir is neither expanded nor marked nested."""
    discovered = set(discover_config_dirs())
    expanded = {subdir for _, subdir, _ in SUBSYSTEMS}
    accounted = expanded | set(NESTED)

    unaccounted = discovered - accounted
    if unaccounted:
        raise SystemExit(
            "Config subdirectories with a schema but no documentation:\n  "
            + "\n  ".join(sorted(unaccounted))
            + "\nAdd each to SUBSYSTEMS (with a README.md) or to NESTED in scripts/sync_config_doc.py."
        )

    stale = set(NESTED) - discovered
    if stale:
        raise SystemExit("NESTED lists subdirs that no longer exist: " + ", ".join(sorted(stale)))

    missing_readme = [subdir for subdir in expanded if not (CONFIG_ROOT / subdir / "README.md").exists()]
    if missing_readme:
        raise SystemExit("Missing README.md for config subsystem(s): " + ", ".join(sorted(missing_readme)))


def slugify(text: str) -> str:
    """Approximate GitHub's heading-anchor slug algorithm."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)  # drop punctuation (keeps word chars, space, hyphen)
    text = text.replace(" ", "-")
    return text


_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK = re.compile(r"(!?\[[^\]]*\])\(([^)]+)\)")


def _rebase_target(target: str, readme_dir: Path) -> str:
    """Rewrite a relative link target so it resolves from DOC_DIR instead of readme_dir."""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#") or target.startswith("/"):
        return target  # absolute URL, scheme, pure anchor, or root-relative: leave alone
    path, _, frag = target.partition("#")
    if not path:
        return target
    resolved = os.path.normpath(readme_dir / path)
    rebased = os.path.relpath(resolved, DOC_DIR)
    rebased = Path(rebased).as_posix()
    return rebased + ("#" + frag if frag else "")


# A TOC entry: (heading level, heading text, final slug).
TocEntry = Tuple[int, str, str]


def transclude(subdir: str, title: str) -> Tuple[str, List[TocEntry]]:
    """Render the README at subdir as a section and return (text, toc_entries).

    The README's H1 is dropped, its headings are demoted one level under a new
    `## {title}`, relative links are rebased, and intra-doc anchors are remapped
    to their final (de-duplicated) slugs. toc_entries lists the section heading
    and every sub-heading, in document order, for the nested table of contents."""
    readme_dir = CONFIG_ROOT / subdir
    raw = (readme_dir / "README.md").read_text(encoding="utf-8").splitlines()

    # First pass: find headings (outside code fences), drop the leading H1, and
    # record each heading's demoted level and original same-text slug so anchor
    # links can be remapped after global de-duplication.
    in_fence = False
    body: List[str] = []
    headings: List[Tuple[int, str, str]] = []  # (demoted level, original slug, text)
    dropped_h1 = False
    for line in raw:
        if _FENCE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        if not in_fence:
            m = _HEADING.match(line)
            if m:
                hashes, htext = m.group(1), m.group(2)
                if len(hashes) == 1 and not dropped_h1:
                    dropped_h1 = True  # the README's own title; replaced by ## {title}
                    continue
                demoted = "#" + hashes + " " + htext  # demote one level
                headings.append((len(hashes) + 1, slugify(htext), htext))
                body.append(demoted)
                continue
        body.append(line)
    while body and not body[0].strip():  # drop blank lines left by the removed H1
        body.pop(0)
    return _finish(subdir, title, body, headings, readme_dir)


# Global slug counter so duplicate headings across subsystems get -1/-2 suffixes,
# matching how a single rendered Markdown page de-duplicates anchors.
_slug_counts: Dict[str, int] = {}


def _final_slug(text: str) -> str:
    base = slugify(text)
    n = _slug_counts.get(base, 0)
    _slug_counts[base] = n + 1
    return base if n == 0 else f"{base}-{n}"


def _finish(
    subdir: str, title: str, body: List[str], headings: List[Tuple[int, str, str]], readme_dir: Path
) -> Tuple[str, List[TocEntry]]:
    # Assign final slugs in document order: the section's own `## {title}` first,
    # then each demoted heading. Build this README's anchor remap and TOC entries.
    toc: List[TocEntry] = [(2, title, _final_slug(title))]
    anchor_map: Dict[str, str] = {}
    for level, original_slug, htext in headings:
        slug = _final_slug(htext)
        anchor_map[original_slug] = slug
        toc.append((level, htext, slug))

    out_lines = [f"## {title}", ""]
    in_fence = False
    for line in body:
        if _FENCE.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue

        def repl(match: "re.Match[str]") -> str:
            label, target = match.group(1), match.group(2)
            if target.startswith("#"):
                remapped = anchor_map.get(target[1:])
                return f"{label}(#{remapped})" if remapped else match.group(0)
            return f"{label}({_rebase_target(target, readme_dir)})"

        out_lines.append(_LINK.sub(repl, line))
    return "\n".join(out_lines).rstrip() + "\n", toc


def build_generated_region() -> str:
    _slug_counts.clear()
    sections: List[str] = []
    blocks: List[Tuple[str, List[TocEntry]]] = []
    for _, subdir, title in SUBSYSTEMS:
        text, entries = transclude(subdir, title)
        sections.append(text)
        blocks.append((subdir, entries))

    # Nested table of contents: each block links to its section and to the
    # colocated README the fields are derived from; sub-headings nest beneath.
    toc = [
        "**Configuration reference.** Each block's field tables are generated from its colocated",
        "`README.md` (next to the schema); the sub-items below link to each section.",
        "",
    ]
    for subdir, entries in blocks:
        _, title, slug = entries[0]
        readme_rel = Path(os.path.relpath(CONFIG_ROOT / subdir / "README.md", DOC_DIR)).as_posix()
        toc.append(f"- [{title}](#{slug}) — source: [`config/{subdir}/README.md`]({readme_rel})")
        for level, text, sub_slug in entries[1:]:
            toc.append(f"{'  ' * (level - 2)}- [{text}](#{sub_slug})")
    toc_block = "\n".join(toc) + "\n"

    return toc_block + "\n" + "\n\n".join(sections)


def render_doc() -> str:
    enforce_coverage()
    text = DOC_PATH.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"{DOC_PATH} is missing the generated-region markers. Add:\n{BEGIN}\n{END}\nwhere the per-subsystem reference should appear."
        )
    pre = text.split(BEGIN)[0]
    post = text.split(END)[1]
    region = build_generated_region()
    return f"{pre}{BEGIN}\n\n{region}\n{END}{post}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync docs/config.md from per-subsystem READMEs.")
    parser.add_argument("--check", action="store_true", help="Fail if docs/config.md is out of sync.")
    args = parser.parse_args()

    enforce_coverage()

    if args.check:
        drift = fill_all_readmes(write=False)
        if drift:
            print(
                "Subsystem README field tables are out of sync with the schema: "
                + ", ".join(drift)
                + "\nRun `pdm run update:config-doc`.",
                file=sys.stderr,
            )
            sys.exit(1)
        # READMEs are in sync, so transcluding from disk yields the expected doc.
        if DOC_PATH.read_text(encoding="utf-8") != render_doc():
            print(f"{DOC_PATH} is out of sync. Run `pdm run update:config-doc`.", file=sys.stderr)
            sys.exit(1)
        print(f"{DOC_PATH} and subsystem field tables are in sync.")
        return

    fill_all_readmes(write=True)
    DOC_PATH.write_text(render_doc(), encoding="utf-8")
    print(f"Wrote {DOC_PATH} and refreshed subsystem field tables.")


if __name__ == "__main__":
    main()
