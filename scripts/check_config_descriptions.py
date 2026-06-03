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
"""Enforce that every config field carries a description.

The config schema is the single source of truth for field documentation: the CLI
flag reference (docs/cli_flags.md) and the configuration guide (docs/config.md,
via the per-subsystem READMEs) both derive from ``Field(description=...)``. This
check walks every Pydantic model reachable from the root ``Config`` and fails if
any field is missing a non-empty description, so no config option ships
undocumented.

Usage:
    python scripts/check_config_descriptions.py
"""

import inspect
import sys
import typing
from typing import List, Set, Type

from pydantic import BaseModel

from inference_perf.config.config import Config


def _iter_nested_models(annotation: object) -> "list[type]":
    """Yield every BaseModel subclass nested anywhere inside a type annotation."""
    found = []
    stack = [annotation]
    seen_types: Set[object] = set()
    while stack:
        tp = stack.pop()
        if tp in seen_types:
            continue
        seen_types.add(tp)
        if inspect.isclass(tp) and issubclass(tp, BaseModel):
            found.append(tp)
        stack.extend(typing.get_args(tp))
    return found


def collect_models(root: Type[BaseModel]) -> List[Type[BaseModel]]:
    """Every Pydantic model reachable from ``root`` via field annotations."""
    seen: Set[Type[BaseModel]] = set()
    order: List[Type[BaseModel]] = []
    stack: List[Type[BaseModel]] = [root]
    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen.add(model)
        order.append(model)
        for field in model.model_fields.values():
            stack.extend(_iter_nested_models(field.annotation))
    return order


def main() -> None:
    gaps = []
    for model in collect_models(Config):
        for name, field in model.model_fields.items():
            if not (field.description and field.description.strip()):
                gaps.append(f"{model.__module__}.{model.__name__}.{name}")

    if gaps:
        print(
            f"{len(gaps)} config field(s) are missing a Field(description=...):\n  " + "\n  ".join(sorted(gaps)),
            file=sys.stderr,
        )
        print(
            "\nEvery config field must document itself so docs/cli_flags.md and docs/config.md stay in sync with the schema.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("All config fields have descriptions.")


if __name__ == "__main__":
    main()
