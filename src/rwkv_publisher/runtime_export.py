from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_FORMAT_VERSION = 4
EXPORT_FORMAT_VERSION = 3
STATE_SOURCE = "kernel_tilelang_state.py"
DECODE_SOURCE = "kernel_tilelang_decode.py"
RUNTIME_SOURCE_FILES = (
    "configuration_rwkv7.py",
    "custom_ops.py",
    "kernel_dispatch.py",
    "modeling_rwkv7.py",
    "state.py",
    "tilelang_decode.py",
    "kernel_tilelang_state.py",
    "kernel_tilelang_decode.py",
)
RUNTIME_MODULE_FILES = (
    "configuration_rwkv7.py",
    "custom_ops.py",
    "kernel_dispatch.py",
    "modeling_rwkv7.py",
    "state.py",
    "tilelang_decode.py",
)
RUNTIME_MODULE_ORDER = (
    "configuration_rwkv7",
    "state",
    "custom_ops",
    "kernel_dispatch",
    "tilelang_decode",
    "modeling_rwkv7",
)
FLAT_RUNTIME_FILES = (
    "runtime.py",
    "kernel.py",
)


@dataclass(frozen=True)
class RuntimeExport:
    files: dict[str, str]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class KernelExport:
    kernel: str
    provenance: dict[str, Any]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bound_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Import | ast.ImportFrom):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            return set()
        return {
            alias.asname or alias.name.split(".", 1)[0]
            for alias in node.names
            if alias.name != "*"
        }
    if isinstance(node, ast.Assign | ast.AnnAssign):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names: set[str] = set()
        for target in targets:
            names.update(
                child.id for child in ast.walk(target) if isinstance(child, ast.Name)
            )
        return names
    return set()


def _validate_kernel_tree(tree: ast.Module, filename: str, *, decode: bool) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Global | ast.Nonlocal):
            raise TypeError(f"{filename} uses unsupported namespace mutation")
        if isinstance(node, ast.Name) and node.id in {"__file__", "__name__"}:
            raise ValueError(f"{filename} depends on unsupported {node.id}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"globals", "locals"}
        ):
            raise ValueError(f"{filename} calls unsupported {node.func.id}()")
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            raise ValueError(f"{filename} mutates or inspects sys.modules")
        if isinstance(node, ast.ImportFrom) and node.level:
            allowed = (
                decode
                and node.level == 1
                and node.module == "kernel_tilelang_state"
                and [(alias.name, alias.asname) for alias in node.names]
                == [("cuda_arch_key", None)]
            )
            if not allowed:
                raise ValueError(
                    f"{filename} has unsupported relative import from {node.module!r}"
                )


def _source_inventory(
    source: str, filename: str, *, decode: bool
) -> tuple[tuple[str, ...], set[int]]:
    tree = ast.parse(source, filename=filename)
    _validate_kernel_tree(tree, filename, decode=decode)
    bindings: set[str] = set()
    removed_lines: set[int] = set()
    for node in tree.body:
        supported = isinstance(
            node,
            ast.Assign
            | ast.AnnAssign
            | ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef
            | ast.Import
            | ast.ImportFrom,
        ) or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not supported:
            raise TypeError(
                f"{filename} has unsupported top-level {type(node).__name__}"
            )
        bindings.update(_bound_names(node))
        remove = isinstance(node, ast.ImportFrom) and (
            node.module == "__future__"
            or (decode and node.level == 1 and node.module == "kernel_tilelang_state")
        )
        if remove:
            removed_lines.update(
                range(node.lineno, (node.end_lineno or node.lineno) + 1)
            )
    if decode:
        bindings.add("cuda_arch_key")
    return tuple(sorted(bindings)), removed_lines


def _kernel_factory(
    name: str,
    source: str,
    source_sha256: str,
    exports: tuple[str, ...],
    removed_lines: set[int],
    *,
    decode: bool,
) -> str:
    lines = source.splitlines()
    body = "\n".join(
        "" if index in removed_lines else line for index, line in enumerate(lines, 1)
    )
    future_imports = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise ValueError("could not preserve future import")
            future_imports.append(segment)
    executable_source = "\n".join([*future_imports, body, ""])
    output = [
        f"def _build_{name}_namespace("
        + ("state_cuda_arch_key" if decode else "")
        + "):",
        f"    source = {executable_source!r}",
        f"    filename = '<rwkv7_{name}_{source_sha256}>'",
        "    linecache.cache[filename] = (",
        "        len(source), None, source.splitlines(keepends=True), filename",
        "    )",
        f"    namespace = {{'__name__': 'inference.kernel.{name}'}}",
    ]
    if decode:
        output.append("    namespace['cuda_arch_key'] = state_cuda_arch_key")
    output.extend(
        [
            "    exec(compile(source, filename, 'exec'), namespace, namespace)  # noqa: S102",
            "",
            "    return SimpleNamespace(",
            *(f"        {export}=namespace[{export!r}]," for export in exports),
            "    )",
        ]
    )
    return "\n".join(output)


def build_kernel_export(runtime_source: Path) -> KernelExport:
    sources = {}
    inventories = {}
    removed = {}
    for filename, decode in ((STATE_SOURCE, False), (DECODE_SOURCE, True)):
        path = runtime_source / filename
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"kernel source does not exist: {path}")
        source = path.read_text(encoding="utf-8")
        exports, removed_lines = _source_inventory(source, filename, decode=decode)
        sources[filename] = source
        inventories[filename] = exports
        removed[filename] = removed_lines
    source_hashes = {
        filename: _sha256_text(source) for filename, source in sources.items()
    }
    state_factory = _kernel_factory(
        "state",
        sources[STATE_SOURCE],
        source_hashes[STATE_SOURCE],
        inventories[STATE_SOURCE],
        removed[STATE_SOURCE],
        decode=False,
    )
    decode_factory = _kernel_factory(
        "decode",
        sources[DECODE_SOURCE],
        source_hashes[DECODE_SOURCE],
        inventories[DECODE_SOURCE],
        removed[DECODE_SOURCE],
        decode=True,
    )
    kernel = "\n".join(
        [
            '"""Generated TileLang kernel export. Do not edit; regenerate it."""',
            "",
            "import linecache",
            "from types import SimpleNamespace",
            "",
            f"EXPORT_FORMAT_VERSION = {EXPORT_FORMAT_VERSION}",
            f"SOURCE_SHA256 = {source_hashes!r}",
            "",
            state_factory,
            "",
            "",
            decode_factory,
            "",
            "",
            "state = _build_state_namespace()",
            "decode = _build_decode_namespace(state.cuda_arch_key)",
            "",
            '__all__ = ["state", "decode"]',
            "",
        ]
    )
    kernel_sha256 = _sha256_text(kernel)
    provenance = {
        "format_version": EXPORT_FORMAT_VERSION,
        "output": "inference/kernel.py",
        "output_sha256": kernel_sha256,
        "sources": {
            filename: {
                "sha256": source_hashes[filename],
                "namespace": namespace,
                "exports": list(inventories[filename]),
            }
            for filename, namespace in (
                (STATE_SOURCE, "state"),
                (DECODE_SOURCE, "decode"),
            )
        },
    }
    return KernelExport(kernel=kernel, provenance=provenance)


def _offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(node: ast.AST, offsets: list[int]) -> tuple[int, int]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if lineno is None or col_offset is None or end_lineno is None:
        raise ValueError("import node has no source span")
    start = offsets[lineno - 1] + col_offset
    end = offsets[end_lineno - 1] + (end_col_offset or 0)
    return start, end


def _import_names(node: ast.ImportFrom) -> str:
    values = [
        f"{alias.name} as {alias.asname}" if alias.asname else alias.name
        for alias in node.names
    ]
    return ", ".join(values)


def _replacement(node: ast.ImportFrom, indent: str) -> str | None:
    module = node.module or ""
    if node.level == 0 and module == "rwkv7_pytorch.tilelang_decode":
        return f"from .tilelang_decode import {_import_names(node)}"
    if node.level != 1 or module not in {
        "kernel_tilelang_state",
        "kernel_tilelang_decode",
    }:
        return None
    namespace = "state" if module.endswith("state") else "decode"
    alias = f"_kernel_{namespace}"
    lines = [f"from .kernel import {namespace} as {alias}"]
    for item in node.names:
        target = item.asname or item.name
        lines.append(f"{target} = {alias}.{item.name}")
    return ("\n" + indent).join(lines)


def transform_runtime_source(source: str, filename: str) -> str:
    tree = ast.parse(source, filename=filename)
    offsets = _offsets(source)
    replacements: list[tuple[int, int, str]] = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        indent = lines[node.lineno - 1][: node.col_offset]
        replacement = _replacement(node, indent)
        if replacement is not None:
            start, end = _node_span(node, offsets)
            replacements.append((start, end, replacement))
    transformed = source
    for start, end, replacement in sorted(replacements, reverse=True):
        transformed = transformed[:start] + replacement + transformed[end:]
    ast.parse(transformed, filename=filename)
    transformed_tree = ast.parse(transformed, filename=filename)
    forbidden: list[str] = []
    for node in ast.walk(transformed_tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(("rwkv7_pytorch", "kernel_tilelang")):
                forbidden.append(module)
    if forbidden:
        raise ValueError(
            f"runtime transform left forbidden imports in {filename}: {forbidden}"
        )
    if "rwkv7_pytorch." in transformed:
        raise ValueError(
            f"runtime transform left absolute package references in {filename}"
        )
    return transformed


def _build_runtime_bundle(
    sources: dict[str, str], source_hashes: dict[str, str]
) -> str:
    return "\n".join(
        [
            '"""Generated RWKV-7 inference runtime. Do not edit; regenerate it."""',
            "",
            "import linecache",
            "import sys",
            "from types import ModuleType",
            "",
            f"RUNTIME_FORMAT_VERSION = {RUNTIME_FORMAT_VERSION}",
            f"SOURCE_SHA256 = {source_hashes!r}",
            f"_SOURCES = {sources!r}",
            "",
            "",
            "def _load_module(name):",
            "    source = _SOURCES[name]",
            "    module_name = f'{__package__}.{name}'",
            "    filename = f'<rwkv7_runtime_{name}_{SOURCE_SHA256[name]}>'",
            "    linecache.cache[filename] = (",
            "        len(source), None, source.splitlines(keepends=True), filename",
            "    )",
            "    module = ModuleType(module_name)",
            "    module.__file__ = filename",
            "    module.__package__ = __package__",
            "    sys.modules[module_name] = module",
            "    try:",
            "        exec(compile(source, filename, 'exec'), module.__dict__, module.__dict__)",
            "    except Exception:",
            "        sys.modules.pop(module_name, None)",
            "        raise",
            "    return module",
            "",
            "",
            f"_MODULE_ORDER = {RUNTIME_MODULE_ORDER!r}",
            "_MODULES = {name: _load_module(name) for name in _MODULE_ORDER}",
            "RWKV7Config = _MODULES['configuration_rwkv7'].RWKV7Config",
            "RWKV7LayerState = _MODULES['state'].RWKV7LayerState",
            "RWKV7State = _MODULES['state'].RWKV7State",
            "RWKV7ForCausalLM = _MODULES['modeling_rwkv7'].RWKV7ForCausalLM",
            "",
            '__all__ = ["RWKV7Config", "RWKV7ForCausalLM", "RWKV7LayerState", "RWKV7State"]',
            "",
        ]
    )


def build_flat_runtime(runtime_source: Path) -> RuntimeExport:
    runtime_source = runtime_source.resolve()
    for filename in RUNTIME_SOURCE_FILES:
        path = runtime_source / filename
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"runtime source does not exist: {path}")
    files = {}
    transforms = {}
    transformed_sources = {}
    source_hashes = {}
    for filename in RUNTIME_MODULE_FILES:
        source = (runtime_source / filename).read_text(encoding="utf-8")
        transformed = transform_runtime_source(source, filename)
        module_name = filename.removesuffix(".py")
        transformed_sources[module_name] = transformed
        source_hashes[module_name] = _sha256_text(source)
        transforms[filename] = {
            "source_sha256": _sha256_text(source),
            "output_sha256": _sha256_text(transformed),
        }
    files["runtime.py"] = _build_runtime_bundle(transformed_sources, source_hashes)
    kernel = build_kernel_export(runtime_source)
    files["kernel.py"] = kernel.kernel
    provenance = {
        "format_version": RUNTIME_FORMAT_VERSION,
        "files": {
            filename: {"sha256": _sha256_text(text)}
            for filename, text in sorted(files.items())
        },
        "transforms": transforms,
        "kernel": kernel.provenance,
    }
    return RuntimeExport(files=files, provenance=provenance)


def write_flat_runtime(runtime_source: Path, destination: Path) -> RuntimeExport:
    export = build_flat_runtime(runtime_source)
    destination.mkdir(parents=True, exist_ok=False)
    for filename, content in export.files.items():
        (destination / filename).write_text(content, encoding="utf-8")
    return export


__all__ = [
    "FLAT_RUNTIME_FILES",
    "RUNTIME_FORMAT_VERSION",
    "RUNTIME_MODULE_FILES",
    "RUNTIME_SOURCE_FILES",
    "RuntimeExport",
    "build_flat_runtime",
    "build_kernel_export",
    "transform_runtime_source",
    "write_flat_runtime",
]
