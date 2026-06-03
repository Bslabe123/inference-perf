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
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def transclude(subdir: str, title: str) -> str:
    """Return the README at subdir as a section: H1 dropped, headings demoted one
    level under a new `## {title}`, relative links rebased, intra-doc anchors
    remapped to their final (de-duplicated) slugs."""
    readme_dir = CONFIG_ROOT / subdir
    raw = (readme_dir / "README.md").read_text(encoding="utf-8").splitlines()

    # First pass: find headings (outside code fences), drop the leading H1, and
    # build a map from each heading's original same-text slug to its demoted text
    # so anchor links can be remapped after global de-duplication.
    in_fence = False
    body: List[str] = []
    headings: List[Tuple[str, str]] = []  # (original_slug, demoted_text)
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
                headings.append((slugify(htext), htext))
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


def _finish(subdir: str, title: str, body: List[str], headings: List[Tuple[str, str]], readme_dir: Path) -> str:
    # Assign final slugs in document order: the section's own `## {title}` first,
    # then each demoted heading. Build this README's anchor remap.
    _final_slug(title)  # reserve the section heading's slug
    anchor_map: Dict[str, str] = {}
    for original_slug, htext in headings:
        anchor_map[original_slug] = _final_slug(htext)

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
    return "\n".join(out_lines).rstrip() + "\n"


def build_generated_region() -> str:
    _slug_counts.clear()
    # Mini table of contents for the generated sections.
    toc = ["Jump to a block:", ""]
    for _, _, title in SUBSYSTEMS:
        toc.append(f"- [{title}](#{slugify(title)})")
    toc_block = "\n".join(toc) + "\n"

    # Reset and recompute slugs for the real render so the TOC slugs match.
    _slug_counts.clear()
    sections = [transclude(subdir, title) for _, subdir, title in SUBSYSTEMS]
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

    expected = render_doc()

    if args.check:
        current = DOC_PATH.read_text(encoding="utf-8")
        if current != expected:
            print(f"{DOC_PATH} is out of sync. Run `pdm run update:config-doc`.", file=sys.stderr)
            sys.exit(1)
        print(f"{DOC_PATH} is in sync.")
        return

    DOC_PATH.write_text(expected, encoding="utf-8")
    print(f"Wrote {DOC_PATH}.")


if __name__ == "__main__":
    main()
