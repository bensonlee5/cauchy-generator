from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .common import REPO_ROOT, SRC_ROOT, repo_relative


@dataclass(frozen=True)
class ModuleSummary:
    imports: tuple[str, ...]
    direct_importers: tuple[str, ...]
    transitive_importers: tuple[str, ...]
    impacted_packages: tuple[str, ...]


@dataclass(frozen=True)
class ImportGraph:
    imports: dict[str, tuple[str, ...]]
    reverse_imports: dict[str, tuple[str, ...]]
    module_paths: dict[str, str]

    def transitive_importers(self, module: str) -> tuple[str, ...]:
        visited: set[str] = set()
        stack = list(self.reverse_imports.get(module, ()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.reverse_imports.get(current, ()))
        return tuple(sorted(visited))

    def module_summary(self, module: str) -> ModuleSummary:
        direct_importers = self.reverse_imports.get(module, ())
        transitive_importers = self.transitive_importers(module)
        impacted_packages = sorted(
            {
                module_to_package(importer)
                for importer in transitive_importers
                if module_to_package(importer)
                not in {module_to_package(module), "dagzoo", "dagzoo.__main__"}
            }
        )
        return ModuleSummary(
            imports=self.imports.get(module, ()),
            direct_importers=direct_importers,
            transitive_importers=transitive_importers,
            impacted_packages=tuple(impacted_packages),
        )


def module_to_package(module: str) -> str:
    parts = module.split(".")
    if len(parts) <= 2:
        return module
    return ".".join(parts[:2])


def module_name_for_path(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT)
    module = "dagzoo." + ".".join(relative.parts)
    module = module[:-3] if module.endswith(".py") else module
    if module.endswith(".__init__"):
        module = module[:-9]
    return module


def path_to_module(path_str: str) -> str | None:
    path = REPO_ROOT / path_str
    if not path.is_relative_to(SRC_ROOT) or path.suffix != ".py":
        return None
    return module_name_for_path(path)


def build_import_graph() -> ImportGraph:
    module_paths: dict[str, str] = {}
    known_modules: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = module_name_for_path(path)
        module_paths[module] = repo_relative(path)
        known_modules.add(module)

    imports: dict[str, tuple[str, ...]] = {}
    reverse_imports: dict[str, set[str]] = {module: set() for module in known_modules}
    for module, path_str in module_paths.items():
        path = REPO_ROOT / path_str
        module_imports = tuple(sorted(_collect_imports(path, module, known_modules)))
        imports[module] = module_imports
        for dependency in module_imports:
            reverse_imports.setdefault(dependency, set()).add(module)

    return ImportGraph(
        imports=imports,
        reverse_imports={key: tuple(sorted(value)) for key, value in reverse_imports.items()},
        module_paths=module_paths,
    )


def _collect_imports(path: Path, module: str, known_modules: set[str]) -> set[str]:
    tree = ast.parse(path.read_text())
    current_package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    discovered: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                normalized = _normalize_import(alias.name, known_modules)
                if normalized is not None and normalized != module:
                    discovered.add(normalized)
        elif isinstance(node, ast.ImportFrom):
            for dependency in _normalize_import_from(node, current_package, known_modules):
                if dependency != module:
                    discovered.add(dependency)
    return discovered


def _normalize_import(candidate: str, known_modules: set[str]) -> str | None:
    matches = [
        module
        for module in known_modules
        if candidate == module or candidate.startswith(module + ".")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _normalize_import_from(
    node: ast.ImportFrom, current_package: str, known_modules: set[str]
) -> set[str]:
    if node.level == 0:
        base_parts = ()
    else:
        current_parts = current_package.split(".")
        if node.level == 1:
            base_parts = tuple(current_parts)
        else:
            base_parts = tuple(current_parts[: -(node.level - 1)])
    module_parts = tuple(node.module.split(".")) if node.module else ()
    base_module = ".".join(base_parts + module_parts)
    discovered: set[str] = set()
    if base_module.startswith("dagzoo"):
        normalized = _normalize_import(base_module, known_modules)
        if normalized is not None:
            discovered.add(normalized)
    for alias in node.names:
        if alias.name == "*":
            continue
        child_candidate = f"{base_module}.{alias.name}" if base_module else alias.name
        normalized_child = _normalize_import(child_candidate, known_modules)
        if normalized_child is not None:
            discovered.add(normalized_child)
    return discovered
