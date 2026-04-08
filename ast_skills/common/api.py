from __future__ import annotations

import ast
import importlib
import inspect
import logging
import pkgutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import networkx as nx
from tqdm import tqdm
import rich

logger = logging.getLogger(__name__)

RICH_GITHUB_REPO = "Textualize/rich"
CACHE_ROOT = Path(".cache/ast_skills/repos")


class RichGithubTestRef(NamedTuple):
    """Reference to a pytest case in a Textualize/rich GitHub checkout."""

    repo_relative_path: str
    test_qualname: str
    lineno: int
    url: str


def _rich_github_blob_url(
    *,
    github_ref: str,
    repo_relative_path: str,
    lineno: int,
) -> str:
    return (
        f"https://github.com/{RICH_GITHUB_REPO}/blob/"
        f"{github_ref}/{repo_relative_path}#L{lineno}"
    )


@dataclass
class FunctionCallInfo:
    module_name: str
    qualname: str
    full_name: str
    kind: str  # function | method | async_function | async_method
    class_name: str | None
    lineno: int | None
    end_lineno: int | None
    calls: list[str] = field(default_factory=list)
    github_tests: list[RichGithubTestRef] = field(default_factory=list)


@dataclass
class ClassInfo:
    module_name: str
    class_name: str
    full_name: str
    lineno: int | None
    end_lineno: int | None
    methods: list[FunctionCallInfo] = field(default_factory=list)


@dataclass
class ModuleIndex:
    functions: dict[str, str] = field(default_factory=dict)  # local name -> full name
    classes: dict[str, str] = field(default_factory=dict)  # local class -> full name
    methods: dict[str, dict[str, str]] = field(
        default_factory=dict
    )  # class -> method -> full name


def _iter_package_module_names(package_name: str) -> Iterator[str]:
    package = importlib.import_module(package_name)
    yield package.__name__
    for _finder, name, _ispkg in pkgutil.walk_packages(
        path=package.__path__,
        prefix=f"{package.__name__}.",
    ):
        yield name


def _iter_rich_module_names() -> Iterator[str]:
    yield from _iter_package_module_names(rich.__name__)


def _iter_defined_classes(
    tree: ast.Module,
) -> Iterator[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield node


def _module_source(module_name: str) -> str | None:
    module = importlib.import_module(module_name)

    try:
        return inspect.getsource(module)
    except (OSError, TypeError):
        pass

    try:
        filename = inspect.getsourcefile(module)
        if filename is None:
            return None
        return Path(filename).read_text(encoding="utf-8")
    except OSError:
        return None


def _iter_defined_functions(
    tree: ast.Module,
) -> Iterator[tuple[str, str, str | None, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """
    Yields:
      (qualname, kind, class_name, node)

    Examples:
      ("get_console", "function", None, node)
      ("Console.print", "method", "Console", node)
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, "function", None, node
        elif isinstance(node, ast.AsyncFunctionDef):
            yield node.name, "async_function", None, node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    yield f"{node.name}.{child.name}", "method", node.name, child
                elif isinstance(child, ast.AsyncFunctionDef):
                    yield f"{node.name}.{child.name}", "async_method", node.name, child


def _build_module_index(module_name: str, tree: ast.Module) -> ModuleIndex:
    index = ModuleIndex()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            index.functions[node.name] = f"{module_name}.{node.name}"

        elif isinstance(node, ast.ClassDef):
            index.classes[node.name] = f"{module_name}.{node.name}"
            index.methods[node.name] = {}

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index.methods[node.name][
                        child.name
                    ] = f"{module_name}.{node.name}.{child.name}"

    return index


def _build_global_indexes(
    module_trees: dict[str, ast.Module],
) -> tuple[dict[str, ModuleIndex], set[str], dict[str, set[str]]]:
    module_indexes: dict[str, ModuleIndex] = {}
    all_full_names: set[str] = set()
    method_name_to_full_names: dict[str, set[str]] = {}

    for module_name, tree in module_trees.items():
        idx = _build_module_index(module_name, tree)
        module_indexes[module_name] = idx

        all_full_names.update(idx.functions.values())
        all_full_names.update(idx.classes.values())

        for methods in idx.methods.values():
            all_full_names.update(methods.values())
            for method_name, full_name in methods.items():
                method_name_to_full_names.setdefault(method_name, set()).add(full_name)

    return module_indexes, all_full_names, method_name_to_full_names


def _collect_import_aliases(module_name: str, tree: ast.Module) -> dict[str, str]:
    """
    Build local import alias -> fully-qualified target.

    Examples:
      from .console import Console  -> {"Console": "rich.console.Console"}
      import rich.console as rc     -> {"rc": "rich.console"}
    """
    aliases: dict[str, str] = {}

    package_parts = module_name.split(".")
    current_pkg = package_parts[:-1]

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue

            if node.level == 0:
                base = node.module
            else:
                prefix_parts = current_pkg[: len(current_pkg) - (node.level - 1)]
                base = ".".join(prefix_parts + [node.module])

            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                aliases[local] = f"{base}.{alias.name}"

    return aliases


def _parse_tests_module_dotted_name(repo_relative_path: str) -> str:
    path = Path(repo_relative_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else "tests"


def _iter_pytest_methods_in_class(
    body: list[ast.stmt],
    *,
    class_short_name: str,
    qualname_prefix: str,
) -> Iterator[tuple[str, str | None, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                yield f"{qualname_prefix}.{node.name}", class_short_name, node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            yield from _iter_pytest_methods_in_class(
                node.body,
                class_short_name=node.name,
                qualname_prefix=f"{qualname_prefix}.{node.name}",
            )


def _iter_pytest_test_nodes(
    body: list[ast.stmt],
    *,
    outer_class_qualname: str,
) -> Iterator[tuple[str, str | None, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                qualname = (
                    f"{outer_class_qualname}.{node.name}"
                    if outer_class_qualname
                    else node.name
                )
                yield qualname, None, node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            prefix = (
                f"{outer_class_qualname}.{node.name}"
                if outer_class_qualname
                else node.name
            )
            yield from _iter_pytest_methods_in_class(
                node.body,
                class_short_name=node.name,
                qualname_prefix=prefix,
            )


def _iter_pytest_module_tests(
    tree: ast.Module,
) -> Iterator[tuple[str, str | None, ast.FunctionDef | ast.AsyncFunctionDef]]:
    yield from _iter_pytest_test_nodes(tree.body, outer_class_qualname="")


_GITHUB_TEST_INDEX_CACHE: dict[tuple[str, ...], dict[str, list[RichGithubTestRef]]] = {}


def _repo_cache_dir_name(repo_url: str) -> str:
    sanitized = repo_url.rstrip("/").replace("https://", "").replace("http://", "")
    return sanitized.replace("/", "__").replace(":", "_")


def _run_git_command(cmd: list[str]) -> bool:
    logger.info(f"{cmd=}")
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning(f"{result.returncode=}")
        logger.warning(f"{result.stderr=}")
        return False
    return True


def _ensure_repo_root(
    *,
    repo_root: Path | None,
    repo_url: str | None,
    github_ref: str,
) -> Path | None:
    if repo_root is not None:
        resolved_root = repo_root.expanduser().resolve()
        logger.info(f"{resolved_root=}")
        return resolved_root

    if not repo_url:
        return None

    cache_dir = CACHE_ROOT / _repo_cache_dir_name(repo_url)
    logger.info(f"{repo_url=}")
    logger.info(f"{cache_dir=}")

    if not cache_dir.exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            github_ref,
            repo_url,
            str(cache_dir),
        ]
        if not _run_git_command(clone_cmd):
            return None
    else:
        fetch_cmd = [
            "git",
            "-C",
            str(cache_dir),
            "fetch",
            "origin",
            github_ref,
            "--depth",
            "1",
        ]
        checkout_cmd = ["git", "-C", str(cache_dir), "checkout", github_ref]
        reset_cmd = [
            "git",
            "-C",
            str(cache_dir),
            "reset",
            "--hard",
            f"origin/{github_ref}",
        ]
        if not _run_git_command(fetch_cmd):
            return None
        if not _run_git_command(checkout_cmd):
            return None
        if not _run_git_command(reset_cmd):
            return None
    return cache_dir.resolve()


def _build_github_test_index(
    *,
    package_prefix: str,
    repo_root: Path,
    tests_subdir: str,
    all_full_names: set[str],
    method_name_to_full_names: dict[str, set[str]],
    github_ref: str,
) -> dict[str, list[RichGithubTestRef]]:
    tests_root = repo_root / tests_subdir
    if not tests_root.is_dir():
        return {}

    reverse_index: dict[str, set[RichGithubTestRef]] = {}
    paths = sorted(tests_root.rglob("*.py"))

    for path in tqdm(paths, desc="Indexing repository tests"):
        rel_path = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            continue

        module_name = _parse_tests_module_dotted_name(rel_path)
        module_index = _build_module_index(module_name, tree)
        import_aliases = _collect_import_aliases(module_name, tree)

        for qualname, class_name, test_node in _iter_pytest_module_tests(tree):
            lineno = test_node.lineno or 1
            test_ref = RichGithubTestRef(
                rel_path,
                qualname,
                lineno,
                _rich_github_blob_url(
                    github_ref=github_ref,
                    repo_relative_path=rel_path,
                    lineno=lineno,
                ),
            )
            collector = RichCallCollector(
                module_name=module_name,
                module_index=module_index,
                import_aliases=import_aliases,
                all_full_names=all_full_names,
                method_name_to_full_names=method_name_to_full_names,
                package_prefix=package_prefix,
                current_class_name=class_name,
            )
            collector.visit(test_node)
            for symbol in set(collector.calls):
                reverse_index.setdefault(symbol, set()).add(test_ref)

    return {
        symbol: sorted(
            refs,
            key=lambda item: (
                item.repo_relative_path,
                item.lineno,
                item.test_qualname,
            ),
        )
        for symbol, refs in reverse_index.items()
    }


def _get_cached_github_test_index(
    *,
    package_name: str,
    package_prefix: str,
    repo_root: Path | None,
    repo_url: str | None,
    tests_subdir: str,
    all_full_names: set[str],
    method_name_to_full_names: dict[str, set[str]],
    github_ref: str,
) -> dict[str, list[RichGithubTestRef]]:
    resolved_repo_root = _ensure_repo_root(
        repo_root=repo_root,
        repo_url=repo_url,
        github_ref=github_ref,
    )
    if resolved_repo_root is None or not resolved_repo_root.is_dir():
        return {}

    cache_key = (
        package_name,
        package_prefix,
        str(resolved_repo_root),
        tests_subdir,
        github_ref,
    )
    if cache_key not in _GITHUB_TEST_INDEX_CACHE:
        _GITHUB_TEST_INDEX_CACHE[cache_key] = _build_github_test_index(
            package_prefix=package_prefix,
            repo_root=resolved_repo_root,
            tests_subdir=tests_subdir,
            all_full_names=all_full_names,
            method_name_to_full_names=method_name_to_full_names,
            github_ref=github_ref,
        )
    return _GITHUB_TEST_INDEX_CACHE[cache_key]


class RichCallCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        module_name: str,
        module_index: ModuleIndex,
        import_aliases: dict[str, str],
        all_full_names: set[str],
        method_name_to_full_names: dict[str, set[str]],
        package_prefix: str,
        current_class_name: str | None = None,
    ) -> None:
        self.module_name = module_name
        self.module_index = module_index
        self.import_aliases = import_aliases
        self.all_full_names = all_full_names
        self.method_name_to_full_names = method_name_to_full_names
        self.package_prefix = package_prefix
        self.current_class_name = current_class_name
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        resolved = self._resolve_call(node.func)
        if resolved is not None:
            self.calls.append(resolved)
        self.generic_visit(node)

    def _resolve_call(self, node: ast.AST) -> str | None:
        # foo()
        if isinstance(node, ast.Name):
            name = node.id

            if name in self.module_index.functions:
                return self.module_index.functions[name]

            if name in self.module_index.classes:
                return self.module_index.classes[name]

            target = self.import_aliases.get(name)
            if target and target.startswith(self.package_prefix) and target in self.all_full_names:
                return target

            return None

        # obj.method()
        if isinstance(node, ast.Attribute):
            # self.foo() or cls.foo()
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in {"self", "cls"}
                and self.current_class_name is not None
            ):
                method_map = self.module_index.methods.get(self.current_class_name, {})
                return method_map.get(node.attr)

            dotted = self._flatten_attribute(node)
            if dotted.startswith(self.package_prefix) and dotted in self.all_full_names:
                return dotted

            root_name = self._root_name(node)
            if root_name is not None and root_name in self.import_aliases:
                root_target = self.import_aliases[root_name]
                suffix = dotted[len(root_name) :]
                candidate = root_target + suffix
                if candidate.startswith(self.package_prefix) and candidate in self.all_full_names:
                    return candidate

            # heuristic: only resolve bare attribute methods if unique in entire package
            candidates = self.method_name_to_full_names.get(node.attr, set())
            if len(candidates) == 1:
                return next(iter(candidates))

            return None

        return None

    def _flatten_attribute(self, node: ast.Attribute) -> str:
        parts: list[str] = []
        current: ast.AST = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)
        else:
            return ""

        return ".".join(reversed(parts))

    def _root_name(self, node: ast.Attribute) -> str | None:
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return None


def collect_rich_class_infos(
    *,
    rich_repo_root: Path | None = None,
    rich_repo_url: str | None = None,
    tests_subdir: str = "tests",
    github_ref: str = "main",
) -> list[ClassInfo]:
    module_names = list(_iter_rich_module_names())

    module_trees: dict[str, ast.Module] = {}
    skipped_no_source = 0
    failed_imports: list[str] = []

    for module_name in tqdm(module_names, desc="Parsing modules"):
        try:
            source = _module_source(module_name)
            if source is None:
                skipped_no_source += 1
                continue
            module_trees[module_name] = ast.parse(source, filename=module_name)
        except Exception:
            failed_imports.append(module_name)

    module_indexes, all_full_names, method_name_to_full_names = _build_global_indexes(
        module_trees
    )
    test_by_symbol = _get_cached_github_test_index(
        package_name="rich",
        package_prefix="rich.",
        repo_root=rich_repo_root,
        repo_url=rich_repo_url,
        tests_subdir=tests_subdir,
        all_full_names=all_full_names,
        method_name_to_full_names=method_name_to_full_names,
        github_ref=github_ref,
    )

    results: list[ClassInfo] = []

    all_classes: list[tuple[str, ast.ClassDef]] = []
    for module_name, tree in module_trees.items():
        for class_node in _iter_defined_classes(tree):
            all_classes.append((module_name, class_node))

    for module_name, class_node in tqdm(all_classes, desc="Resolving class methods"):
        class_name = class_node.name
        class_full_name = f"{module_name}.{class_name}"
        import_aliases = _collect_import_aliases(module_name, module_trees[module_name])

        methods: list[FunctionCallInfo] = []

        for child in class_node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            qualname = f"{class_name}.{child.name}"
            full_name = f"{module_name}.{class_name}.{child.name}"
            kind = (
                "async_method" if isinstance(child, ast.AsyncFunctionDef) else "method"
            )

            collector = RichCallCollector(
                module_name=module_name,
                module_index=module_indexes[module_name],
                import_aliases=import_aliases,
                all_full_names=all_full_names,
                method_name_to_full_names=method_name_to_full_names,
                package_prefix="rich.",
                current_class_name=class_name,
            )
            collector.visit(child)

            methods.append(
                FunctionCallInfo(
                    module_name=module_name,
                    qualname=qualname,
                    full_name=full_name,
                    kind=kind,
                    class_name=class_name,
                    lineno=getattr(child, "lineno", None),
                    end_lineno=getattr(child, "end_lineno", None),
                    calls=sorted(set(collector.calls)),
                    github_tests=list(test_by_symbol.get(full_name, [])),
                )
            )

        results.append(
            ClassInfo(
                module_name=module_name,
                class_name=class_name,
                full_name=class_full_name,
                lineno=getattr(class_node, "lineno", None),
                end_lineno=getattr(class_node, "end_lineno", None),
                methods=methods,
            )
        )

    console = rich.get_console()
    console.rule("[bold green]Summary[/bold green]")
    console.print(f"Parsed modules: {len(module_trees)}")
    console.print(f"Collected functions/methods: {len(results)}")
    console.print(f"Modules with no source: {skipped_no_source}")
    if failed_imports:
        console.print(f"Failed imports: {failed_imports}")
    if rich_repo_root is not None or rich_repo_url:
        link_count = sum(len(refs) for refs in test_by_symbol.values())
        console.print(
            f"GitHub test index: {len(test_by_symbol)} rich symbols "
            f"with tests, {link_count} total links"
        )

    return results


def _sanitize_str(value):
    return "" if value is None else str(value)


def _sanitize_int(value):
    return -1 if value is None else int(value)


def collect_rich_only_call_infos(
    *,
    rich_repo_root: Path | None = None,
    rich_repo_url: str | None = None,
    tests_subdir: str = "tests",
    github_ref: str = "main",
) -> list[FunctionCallInfo]:
    module_names = list(_iter_rich_module_names())

    module_trees: dict[str, ast.Module] = {}
    skipped_no_source = 0
    failed_imports: list[str] = []

    for module_name in tqdm(module_names, desc="Parsing modules"):
        try:
            source = _module_source(module_name)
            if source is None:
                skipped_no_source += 1
                continue
            module_trees[module_name] = ast.parse(source, filename=module_name)
        except Exception:
            failed_imports.append(module_name)

    module_indexes, all_full_names, method_name_to_full_names = _build_global_indexes(
        module_trees
    )
    test_by_symbol = _get_cached_github_test_index(
        package_name="rich",
        package_prefix="rich.",
        repo_root=rich_repo_root,
        repo_url=rich_repo_url,
        tests_subdir=tests_subdir,
        all_full_names=all_full_names,
        method_name_to_full_names=method_name_to_full_names,
        github_ref=github_ref,
    )

    all_defs: list[
        tuple[str, str, str, str | None, str, ast.FunctionDef | ast.AsyncFunctionDef]
    ] = []
    for module_name, tree in module_trees.items():
        for qualname, kind, class_name, node in _iter_defined_functions(tree):
            if class_name is None:
                full_name = f"{module_name}.{qualname}"
            else:
                full_name = f"{module_name}.{class_name}.{node.name}"
            all_defs.append((module_name, qualname, full_name, class_name, kind, node))

    results: list[FunctionCallInfo] = []

    for module_name, qualname, full_name, class_name, kind, node in tqdm(
        all_defs, desc="Resolving calls"
    ):
        import_aliases = _collect_import_aliases(module_name, module_trees[module_name])

        collector = RichCallCollector(
            module_name=module_name,
            module_index=module_indexes[module_name],
            import_aliases=import_aliases,
            all_full_names=all_full_names,
            method_name_to_full_names=method_name_to_full_names,
            package_prefix="rich.",
            current_class_name=class_name,
        )
        collector.visit(node)

        results.append(
            FunctionCallInfo(
                module_name=module_name,
                qualname=qualname,
                full_name=full_name,
                kind=kind,
                class_name=class_name,
                lineno=getattr(node, "lineno", None),
                end_lineno=getattr(node, "end_lineno", None),
                calls=sorted(set(collector.calls)),
                github_tests=list(test_by_symbol.get(full_name, [])),
            )
        )

    console = rich.get_console()
    console.rule("[bold green]Summary[/bold green]")
    console.print(f"Parsed modules: {len(module_trees)}")
    console.print(f"Collected functions/methods: {len(results)}")
    console.print(f"Modules with no source: {skipped_no_source}")
    if failed_imports:
        console.print(f"Failed imports: {failed_imports}")
    if rich_repo_root is not None or rich_repo_url:
        link_count = sum(len(refs) for refs in test_by_symbol.values())
        console.print(
            f"GitHub test index: {len(test_by_symbol)} rich symbols "
            f"with tests, {link_count} total links"
        )

    return results


def collect_package_only_call_infos(
    *,
    package_name: str,
    repo_root: Path | None = None,
    repo_url: str | None = None,
    tests_subdir: str = "tests",
    github_ref: str = "main",
) -> list[FunctionCallInfo]:
    package_prefix = f"{package_name}."
    module_names = list(_iter_package_module_names(package_name))

    module_trees: dict[str, ast.Module] = {}
    skipped_no_source = 0
    failed_imports: list[str] = []

    for module_name in tqdm(module_names, desc="Parsing modules"):
        try:
            source = _module_source(module_name)
            if source is None:
                skipped_no_source += 1
                continue
            module_trees[module_name] = ast.parse(source, filename=module_name)
        except Exception:
            failed_imports.append(module_name)

    module_indexes, all_full_names, method_name_to_full_names = _build_global_indexes(
        module_trees
    )
    test_by_symbol = _get_cached_github_test_index(
        package_name=package_name,
        package_prefix=package_prefix,
        repo_root=repo_root,
        repo_url=repo_url,
        tests_subdir=tests_subdir,
        all_full_names=all_full_names,
        method_name_to_full_names=method_name_to_full_names,
        github_ref=github_ref,
    )

    all_defs: list[
        tuple[str, str, str, str | None, str, ast.FunctionDef | ast.AsyncFunctionDef]
    ] = []
    for module_name, tree in module_trees.items():
        for qualname, kind, class_name, node in _iter_defined_functions(tree):
            if class_name is None:
                full_name = f"{module_name}.{qualname}"
            else:
                full_name = f"{module_name}.{class_name}.{node.name}"
            all_defs.append((module_name, qualname, full_name, class_name, kind, node))

    results: list[FunctionCallInfo] = []
    for module_name, qualname, full_name, class_name, kind, node in tqdm(
        all_defs, desc="Resolving calls"
    ):
        import_aliases = _collect_import_aliases(module_name, module_trees[module_name])
        collector = RichCallCollector(
            module_name=module_name,
            module_index=module_indexes[module_name],
            import_aliases=import_aliases,
            all_full_names=all_full_names,
            method_name_to_full_names=method_name_to_full_names,
            package_prefix=package_prefix,
            current_class_name=class_name,
        )
        collector.visit(node)
        results.append(
            FunctionCallInfo(
                module_name=module_name,
                qualname=qualname,
                full_name=full_name,
                kind=kind,
                class_name=class_name,
                lineno=getattr(node, "lineno", None),
                end_lineno=getattr(node, "end_lineno", None),
                calls=sorted(set(collector.calls)),
                github_tests=list(test_by_symbol.get(full_name, [])),
            )
        )

    console = rich.get_console()
    console.rule("[bold green]Summary[/bold green]")
    console.print(f"Parsed modules: {len(module_trees)}")
    console.print(f"Collected functions/methods: {len(results)}")
    console.print(f"Modules with no source: {skipped_no_source}")
    if failed_imports:
        console.print(f"Failed imports: {failed_imports}")
    if repo_root is not None or repo_url:
        link_count = sum(len(refs) for refs in test_by_symbol.values())
        console.print(
            f"GitHub test index: {len(test_by_symbol)} symbols with tests, "
            f"{link_count} total links"
        )
    return results


def collect_package_class_infos(
    *,
    package_name: str,
    repo_root: Path | None = None,
    repo_url: str | None = None,
    tests_subdir: str = "tests",
    github_ref: str = "main",
) -> list[ClassInfo]:
    package_prefix = f"{package_name}."
    module_names = list(_iter_package_module_names(package_name))

    module_trees: dict[str, ast.Module] = {}
    skipped_no_source = 0
    failed_imports: list[str] = []
    for module_name in tqdm(module_names, desc="Parsing modules"):
        try:
            source = _module_source(module_name)
            if source is None:
                skipped_no_source += 1
                continue
            module_trees[module_name] = ast.parse(source, filename=module_name)
        except Exception:
            failed_imports.append(module_name)

    module_indexes, all_full_names, method_name_to_full_names = _build_global_indexes(
        module_trees
    )
    test_by_symbol = _get_cached_github_test_index(
        package_name=package_name,
        package_prefix=package_prefix,
        repo_root=repo_root,
        repo_url=repo_url,
        tests_subdir=tests_subdir,
        all_full_names=all_full_names,
        method_name_to_full_names=method_name_to_full_names,
        github_ref=github_ref,
    )

    results: list[ClassInfo] = []
    all_classes: list[tuple[str, ast.ClassDef]] = []
    for module_name, tree in module_trees.items():
        for class_node in _iter_defined_classes(tree):
            all_classes.append((module_name, class_node))

    for module_name, class_node in tqdm(all_classes, desc="Resolving class methods"):
        class_name = class_node.name
        import_aliases = _collect_import_aliases(module_name, module_trees[module_name])
        methods: list[FunctionCallInfo] = []
        for child in class_node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            full_name = f"{module_name}.{class_name}.{child.name}"
            collector = RichCallCollector(
                module_name=module_name,
                module_index=module_indexes[module_name],
                import_aliases=import_aliases,
                all_full_names=all_full_names,
                method_name_to_full_names=method_name_to_full_names,
                package_prefix=package_prefix,
                current_class_name=class_name,
            )
            collector.visit(child)
            methods.append(
                FunctionCallInfo(
                    module_name=module_name,
                    qualname=f"{class_name}.{child.name}",
                    full_name=full_name,
                    kind="async_method"
                    if isinstance(child, ast.AsyncFunctionDef)
                    else "method",
                    class_name=class_name,
                    lineno=getattr(child, "lineno", None),
                    end_lineno=getattr(child, "end_lineno", None),
                    calls=sorted(set(collector.calls)),
                    github_tests=list(test_by_symbol.get(full_name, [])),
                )
            )
        results.append(
            ClassInfo(
                module_name=module_name,
                class_name=class_name,
                full_name=f"{module_name}.{class_name}",
                lineno=getattr(class_node, "lineno", None),
                end_lineno=getattr(class_node, "end_lineno", None),
                methods=methods,
            )
        )
    return results


def github_tests(function_info: FunctionCallInfo) -> list[RichGithubTestRef]:
    """Return GitHub test references that call a specific rich symbol."""
    return function_info.github_tests


def build_call_graph(
    infos: list[FunctionCallInfo],
    class_infos: list[ClassInfo],
) -> nx.DiGraph:
    """
    Build a directed graph with:
      - class nodes
      - function/method nodes
      - edges:
          class -> method        (structure)
          caller -> callee       (calls)
    """
    G = nx.DiGraph()

    # -----------------------------
    # 1. Add class nodes
    # -----------------------------
    for cls in class_infos:
        G.add_node(
            cls.full_name,
            node_type="class",
            module=_sanitize_str(cls.module_name),
            class_name=_sanitize_str(cls.class_name),
            lineno=_sanitize_int(cls.lineno),
            end_lineno=_sanitize_int(cls.end_lineno),
        )

    # -----------------------------
    # 2. Add function/method nodes
    # -----------------------------
    for info in infos:
        G.add_node(
            info.full_name,
            node_type=info.kind,  # "function", "method", etc.
            module=_sanitize_str(info.module_name),
            qualname=_sanitize_str(info.qualname),
            class_name=_sanitize_str(info.class_name),
            lineno=_sanitize_int(info.lineno),
            end_lineno=_sanitize_int(info.end_lineno),
        )

        # -----------------------------
        # 3. Add class → method edges
        # -----------------------------
        if info.class_name is not None:
            class_full = f"{info.module_name}.{info.class_name}"

            if not G.has_node(class_full):
                # fallback safety (should already exist)
                G.add_node(
                    class_full,
                    node_type="class",
                    module=_sanitize_str(info.module_name),
                    class_name=_sanitize_str(info.class_name),
                    lineno=-1,
                    end_lineno=-1,
                )

            G.add_edge(
                class_full,
                info.full_name,
                edge_type="has_method",
            )

    # -----------------------------
    # 4. Add call edges
    # -----------------------------
    for info in infos:
        for callee in info.calls:
            if not G.has_node(callee):
                # optional: skip or add external node
                G.add_node(callee, node_type="external")

            G.add_edge(
                info.full_name,
                callee,
                edge_type="calls",
            )

    return G


def print_graph_summary(G: nx.DiGraph) -> None:
    console = rich.get_console()
    console.rule("[bold blue]Graph summary[/bold blue]")
    console.print(f"Nodes: {G.number_of_nodes()}")
    console.print(f"Edges: {G.number_of_edges()}")

    in_degrees = sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:10]
    out_degrees = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:10]

    console.print("\n[bold]Top 10 most-called nodes (highest in-degree)[/bold]")
    for node, deg in in_degrees:
        console.print(f"{deg:>3}  {node}")

    console.print("\n[bold]Top 10 most-calling nodes (highest out-degree)[/bold]")
    for node, deg in out_degrees:
        console.print(f"{deg:>3}  {node}")


def export_graph(G: nx.DiGraph, path: str):
    try:
        nx.write_graphml(G, path)
    except TypeError as e:
        print("GraphML export failed, retrying without attributes:", e)

        G_clean = nx.DiGraph()
        G_clean.add_edges_from(G.edges())
        nx.write_graphml(G_clean, path)


if __name__ == "__main__":
    function_infos = collect_rich_only_call_infos()
    class_infos = collect_rich_class_infos()

    G = build_call_graph(function_infos, class_infos)
