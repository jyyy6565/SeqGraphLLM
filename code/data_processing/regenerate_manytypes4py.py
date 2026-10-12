"""Regenerate the paper-facing ManyTypes4Py graph representations.

The source dataset stores tokenized Python code in ``processed_projects_cleaned``.
This script restores that code, builds one complete Joern CPG per selected source
file, extracts whole-file CFG/DFG/AST graphs, and derives every representation
consumed by the type-inference experiments.

Outputs are written under a separate output root.  Existing experiment data is
therefore not modified until the generated data has been validated and promoted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


GRAPH_TYPES = ("cfg", "ddg", "ast")
REPRESENTATIONS = (
    "GraphDot",
    "GraphText",
    "GraphCode",
    "CDEP",
    "DBP-1-GraphDot",
    "DBP-1-GraphText",
    "DBP-3-GraphDot",
    "DBP-3-GraphText",
    "DBP-5-GraphDot",
    "DBP-5-GraphText",
)


def _load_typeeval_helpers(workspace: Path):
    typeeval_dir = Path(__file__).resolve().parent
    if not typeeval_dir.is_dir():
        raise FileNotFoundError(f"TypeEvalPy helper directory not found: {typeeval_dir}")
    sys.path.insert(0, str(typeeval_dir))
    from cpg_parser import create_parser, extract_subgraphs  # type: ignore

    module_path = typeeval_dir / "code_to_graphs.py"
    spec = importlib.util.spec_from_file_location("typeeval_code_to_graphs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return create_parser, extract_subgraphs, module.cdep_compress_dot


def restore_code(tokenized_code: str) -> str:
    replacements = (
        ("[EOL]", "\n"),
        ("[string]", '"placeholder_string"'),
        ("[number]", "0"),
        ("[comment]", "# placeholder_comment"),
        ("[docstring]", '"""placeholder_docstring"""'),
    )
    code = tokenized_code
    for source, target in replacements:
        code = code.replace(source, target)
    return code


def _run(command: Sequence[str], timeout: int) -> None:
    result = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {detail[-2000:]}")


def _split_dot_statements(dot_content: str) -> Iterable[str]:
    """Split DOT statements on semicolons that occur outside quoted values."""
    body_start = dot_content.find("{")
    body_end = dot_content.rfind("}")
    body = dot_content[body_start + 1 : body_end if body_end > body_start else None]
    current: List[str] = []
    quoted = False
    escaped = False
    for character in body:
        if character == '"' and not escaped:
            quoted = not quoted
        if character == ";" and not quoted:
            statement = "".join(current).strip()
            if statement:
                yield statement
            current = []
            escaped = False
            continue
        current.append(character)
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False


def _normalize_dot_statement(statement: str) -> str:
    """Keep each statement on one line while preserving newlines inside values."""
    output: List[str] = []
    quoted = False
    escaped = False
    for character in statement:
        if character == '"' and not escaped:
            quoted = not quoted
        if character in "\r\n":
            output.append("\\n" if quoted else " ")
        else:
            output.append(character)
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return "".join(output).strip()


def extract_cpg_subgraphs(cpg_dot: str) -> Dict[str, str]:
    """Extract complete CFG/DFG/AST subgraphs without dropping multiline nodes."""
    node_re = re.compile(r'^\s*"([^"]+)"\s*\[(.*)\]\s*$', re.DOTALL)
    edge_re = re.compile(
        r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*\[(.*)\]\s*$', re.DOTALL
    )
    label_re = re.compile(r'\blabel\s*=\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
    nodes: Dict[str, str] = {}
    edges: List[Tuple[str, str, str, str]] = []

    for raw_statement in _split_dot_statements(cpg_dot):
        statement = _normalize_dot_statement(raw_statement)
        edge_match = edge_re.match(statement)
        if edge_match:
            source, target, attributes = edge_match.groups()
            label_match = label_re.search(attributes)
            edge_label = label_match.group(1) if label_match else ""
            edges.append((source, target, edge_label, statement))
            continue
        node_match = node_re.match(statement)
        if node_match:
            nodes[node_match.group(1)] = statement

    mapping = {"cfg": "CFG", "ddg": "REACHING_DEF", "ast": "AST"}
    result: Dict[str, str] = {}
    for graph_type, edge_label in mapping.items():
        selected_edges = [edge for edge in edges if edge[2] == edge_label]
        involved = {node for edge in selected_edges for node in edge[:2]}
        missing = involved.difference(nodes)
        if missing:
            raise RuntimeError(
                f"CPG {graph_type.upper()} extraction is missing {len(missing)} endpoint nodes"
            )
        lines = [f"digraph {graph_type} {{"]
        lines.extend(f"  {nodes[node]};" for node in nodes if node in involved)
        lines.extend(f"  {edge[3]};" for edge in selected_edges)
        lines.append("}")
        result[graph_type] = "\n".join(lines) + "\n"
    return result


def generate_dot_graphs(
    code: str,
    file_path: str,
    joern_dir: Path,
    extract_subgraphs,
    timeout: int,
) -> Dict[str, str]:
    parse_exe = joern_dir / ("joern-parse.bat" if os.name == "nt" else "joern-parse")
    export_exe = joern_dir / ("joern-export.bat" if os.name == "nt" else "joern-export")
    if not parse_exe.is_file() or not export_exe.is_file():
        raise FileNotFoundError(f"Joern CLI not found under {joern_dir}")

    suffix = Path(file_path).suffix or ".py"
    with tempfile.TemporaryDirectory(prefix="manytypes4py_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_file = temp_dir / f"source{suffix}"
        cpg_file = temp_dir / "cpg.bin"
        export_dir = temp_dir / "cpg-dot"
        source_file.write_text(code, encoding="utf-8")

        _run((str(parse_exe), str(source_file), "-o", str(cpg_file)), timeout)
        _run(
            (
                str(export_exe),
                str(cpg_file),
                "-o",
                str(export_dir),
                "--repr",
                "all",
                "--format",
                "dot",
            ),
            timeout,
        )

        exported = sorted(export_dir.glob("*.dot"))
        if len(exported) != 1:
            raise RuntimeError(f"Expected one complete CPG export, found {len(exported)}")
        cpg_dot = exported[0].read_text(encoding="utf-8", errors="replace")
        graphs = extract_cpg_subgraphs(cpg_dot)
    for graph_type, content in graphs.items():
        if not isinstance(content, str) or not content.lstrip().startswith("digraph"):
            raise RuntimeError(f"Joern produced no valid {graph_type.upper()} graph")
    return graphs


def _parse_representation(dot_content: str, graph_type: str, create_parser, mode: str) -> str:
    parser_type = "dfg" if graph_type == "ddg" else graph_type
    with tempfile.NamedTemporaryFile(
        suffix=f"-{parser_type}.dot", mode="w", encoding="utf-8", delete=False
    ) as temp_file:
        temp_file.write(dot_content)
        temp_path = Path(temp_file.name)
    try:
        parser = create_parser(parser_type, temp_path)
        parser.parse_dot_file()
        if mode == "text":
            return parser.explain_in_natural_language()
        if mode == "code":
            return parser.to_add_edge_script()
        raise ValueError(f"Unknown representation mode: {mode}")
    finally:
        temp_path.unlink(missing_ok=True)


def _dot_records(dot_content: str) -> Iterable[Tuple[str, str]]:
    """Yield complete DOT records, including node attributes containing newlines."""
    lines = dot_content.splitlines()
    index = 0
    node_start = re.compile(r'^\s*"([^"]+)"\s*\[')
    while index < len(lines):
        line = lines[index]
        match = node_start.match(line)
        if not match:
            yield "other", line
            index += 1
            continue
        record = [line]
        while not record[-1].rstrip().endswith("];") and index + 1 < len(lines):
            index += 1
            record.append(lines[index])
        yield "node", "\n".join(record)
        index += 1


def simplify_dot(dot_content: str, degree: int) -> str:
    """Apply the existing experiment's one-pass OR degree pruning semantics."""
    edge_re = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')
    node_re = re.compile(r'^\s*"([^"]+)"\s*\[')
    outgoing: Dict[str, set[str]] = {}
    incoming: Dict[str, set[str]] = {}
    edge_rows: List[Tuple[str, str, str]] = []

    for line in dot_content.splitlines():
        match = edge_re.match(line)
        if not match:
            continue
        source, target = match.groups()
        outgoing.setdefault(source, set()).add(target)
        incoming.setdefault(target, set()).add(source)
        edge_rows.append((source, target, line))

    seed_nodes = {
        node for node, neighbors in outgoing.items() if len(neighbors) >= degree
    }
    seed_nodes.update(
        node for node, neighbors in incoming.items() if len(neighbors) >= degree
    )
    kept_edges = {
        (source, target)
        for source, target, _ in edge_rows
        if source in seed_nodes or target in seed_nodes
    }
    kept_nodes = set(seed_nodes)
    for source, target in kept_edges:
        kept_nodes.add(source)
        kept_nodes.add(target)

    output: List[str] = []
    for record_type, line in _dot_records(dot_content):
        edge_match = edge_re.match(line)
        if edge_match:
            if edge_match.groups() in kept_edges:
                output.append(line)
            continue
        node_match = node_re.match(line) if record_type == "node" else None
        if node_match:
            if node_match.group(1) in kept_nodes:
                output.append(line)
            continue
        output.append(line)
    return "\n".join(output).rstrip() + "\n"


def derive_representations(dot_graphs, create_parser, cdep_compress_dot):
    result = {name: {} for name in REPRESENTATIONS}
    result["GraphDot"] = dict(dot_graphs)

    for graph_type, dot_content in dot_graphs.items():
        result["GraphText"][graph_type] = _parse_representation(
            dot_content, graph_type, create_parser, "text"
        )
        result["GraphCode"][graph_type] = _parse_representation(
            dot_content, graph_type, create_parser, "code"
        )

        cdep_dot = cdep_compress_dot(dot_content, min_degree=5)
        result["CDEP"][graph_type] = _parse_representation(
            cdep_dot, graph_type, create_parser, "text"
        )

        for degree in (1, 3, 5):
            dbp_dot = simplify_dot(dot_content, degree)
            result[f"DBP-{degree}-GraphDot"][graph_type] = dbp_dot
            result[f"DBP-{degree}-GraphText"][graph_type] = _parse_representation(
                dbp_dot, graph_type, create_parser, "text"
            )
    return result


def _write_json_atomic(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as temp_file:
        json.dump(data, temp_file, ensure_ascii=False, indent=2)
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def _repo_is_complete(output_root: Path, repo_file: str, expected_paths: Sequence[str]) -> bool:
    expected = set(expected_paths)
    for representation in REPRESENTATIONS:
        path = output_root / representation / repo_file
        if not path.is_file():
            return False
        try:
            keys = set(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return False
        if keys != expected:
            return False
    return True


def regenerate_repo(
    repo_file: str,
    selected_paths: Sequence[str],
    source_root: Path,
    output_root: Path,
    joern_dir: Path,
    create_parser,
    extract_subgraphs,
    cdep_compress_dot,
    timeout: int,
    raw_source: bool,
) -> Tuple[str, int, List[Tuple[str, str]], bool]:
    if _repo_is_complete(output_root, repo_file, selected_paths):
        return repo_file, len(selected_paths), [], True

    source_data = json.loads((source_root / repo_file).read_text(encoding="utf-8"))
    outputs: Dict[str, Dict[str, Dict[str, str]]] = {
        representation: {} for representation in REPRESENTATIONS
    }
    failures: List[Tuple[str, str]] = []

    for file_path in selected_paths:
        try:
            raw_code = source_data[file_path]["code"]
            code = raw_code if raw_source else restore_code(raw_code)
            dot_graphs = generate_dot_graphs(
                code, file_path, joern_dir, extract_subgraphs, timeout
            )
            representations = derive_representations(
                dot_graphs, create_parser, cdep_compress_dot
            )
            for representation, graphs in representations.items():
                outputs[representation][file_path] = graphs
        except Exception as exc:
            failures.append((file_path, str(exc)))

    for representation, data in outputs.items():
        _write_json_atomic(output_root / representation / repo_file, data)
    return repo_file, len(selected_paths) - len(failures), failures, False


def load_selection(manifest_path: Path, limit: int | None = None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest["selected_files"]
    if limit is not None:
        selected = selected[:limit]
    repositories: Dict[str, List[str]] = {}
    for item in selected:
        repo_file = f"{item['repo_name']}.json"
        repositories.setdefault(repo_file, []).append(item["file_path"])
    return repositories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace_default = Path(__file__).resolve().parents[2]
    dataset_default = workspace_default / "primevul_codegnn" / "dataset" / "manytypes4py"
    parser.add_argument("--dataset-dir", type=Path, default=dataset_default)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Source JSON directory; defaults to dataset-dir/processed_projects_cleaned",
    )
    parser.add_argument(
        "--raw-source",
        action="store_true",
        help="Treat source-dir code fields as original Python instead of tokenized sequences",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joern-dir", type=Path, default=workspace_default.parent / "joern-cli")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    source_root = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else dataset_dir / "processed_projects_cleaned"
    )
    manifest_path = dataset_dir / "selected_files_new.json"
    output_root.mkdir(parents=True, exist_ok=True)

    create_parser, extract_subgraphs, cdep_compress_dot = _load_typeeval_helpers(
        workspace_default
    )
    repositories = load_selection(manifest_path, args.limit)
    total_entries = sum(len(paths) for paths in repositories.values())
    print(
        f"Regenerating {total_entries} entries from {len(repositories)} repositories "
        f"with {args.workers} workers",
        flush=True,
    )

    completed_repos = 0
    completed_entries = 0
    skipped_repos = 0
    all_failures: List[Tuple[str, str, str]] = []
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                regenerate_repo,
                repo_file,
                paths,
                source_root,
                output_root,
                args.joern_dir.resolve(),
                create_parser,
                extract_subgraphs,
                cdep_compress_dot,
                args.timeout,
                args.raw_source,
            ): repo_file
            for repo_file, paths in repositories.items()
        }
        for future in concurrent.futures.as_completed(futures):
            repo_file = futures[future]
            try:
                _, success_count, failures, skipped = future.result()
            except Exception as exc:
                success_count = 0
                failures = [("<repository>", str(exc))]
                skipped = False
            with lock:
                completed_repos += 1
                completed_entries += success_count
                skipped_repos += int(skipped)
                all_failures.extend((repo_file, path, error) for path, error in failures)
                print(
                    f"[{completed_repos}/{len(repositories)} repos] "
                    f"entries={completed_entries}/{total_entries} failures={len(all_failures)} "
                    f"skipped_repos={skipped_repos} last={repo_file}",
                    flush=True,
                )

    if all_failures:
        print("Failures:", flush=True)
        for repo_file, file_path, error in all_failures:
            print(f"  {repo_file} :: {file_path} :: {error}", flush=True)
        return 1
    print("Regeneration completed without failures.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
