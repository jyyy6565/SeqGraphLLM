#!/usr/bin/env python3
"""ManyTypes4Py v2 表示重派生（PLAN.md Phase 3，决策 D9/D11）。

MT4Py 的 GraphDot 由新管线生成（0 悬空边、无多行节点丢失），因此**不重跑 Joern**，
只从 `GraphDot/<repo>.json` 重派生：

    GraphDot[path][graph]  (cfg/ddg/ast)
      │ ① 复合节点 CODE 缩写（同 regenerate_bigvul.abbreviate_code，D11 head 规则）
      ▼
    GraphDot-v2 ─② CPGParser（修复版）─▶ GraphText-v2 / GraphCode-v2
               ├③ dbc_v2                ─▶ DBC-v2-GraphDot / DBC-v2-GraphText
               └④ dbp_v2 light/medium/heavy ─▶ DBP-{lv}-v2-GraphDot / DBP-{lv}-v2-GraphText

输出：`<dataset>/<REP>/<repo>.json`，结构与旧表示一致 `{path: {cfg,ddg,ast: str}}`
（type_inference.get_code_graph 直接读 graph_dir/<repo>.json[path][graph]）。
每个 repo 的全部表示一次性原子写；已完整（全部 rep 文件存在且 path 集合一致）则跳过；
--force 强制重做。GraphDot 源为空壳/非法（非 digraph）的条目：写 failures.jsonl，
该 (path,graph) 在 v2 输出中缺失（不写错误文本）。

用法：
    python rederive_manytypes4py_v2.py [--dataset-dir ../dataset/manytypes4py] [--procs 6] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from cpg_parser import CPGParser  # noqa: E402
from graph_simplify_v2 import DBP_LEVELS, dbc_v2, dbp_v2, parse_dot  # noqa: E402
from regenerate_bigvul import abbreviate_node_stmt, now_str  # noqa: E402
from regenerate_manytypes4py import _normalize_dot_statement, _split_dot_statements  # noqa: E402

GRAPHS = ('cfg', 'ddg', 'ast')
REPS = (['GraphDot-v2', 'GraphText-v2', 'GraphCode-v2', 'DBC-v2-GraphDot', 'DBC-v2-GraphText']
        + [f'DBP-{lv}-v2-GraphDot' for lv in DBP_LEVELS]
        + [f'DBP-{lv}-v2-GraphText' for lv in DBP_LEVELS])


class StageError(Exception):
    pass


def rewrite_dot_heads(dot_text: str, graph_name: str, code_mode: str) -> str:
    """语句级重写：节点 CODE 按 head 规则缩写；边原样；输出单行语句的规范 DOT。"""
    if not dot_text or not dot_text.lstrip().startswith('digraph'):
        raise StageError('source dot is not a digraph')
    lines = [f'digraph {graph_name} {{']
    for raw in _split_dot_statements(dot_text):
        stmt = _normalize_dot_statement(raw)
        if '->' not in stmt.split('[', 1)[0] and code_mode == 'head':
            stmt = abbreviate_node_stmt(stmt)
        lines.append(f'  {stmt};')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def serialize(dot_text: str, mode: str, expect_nodes: Optional[int] = None) -> str:
    p = CPGParser()
    p.parse_dot(dot_text)
    if expect_nodes is not None and len(p.nodes) != expect_nodes:
        raise StageError(f'serializer node count {len(p.nodes)} != dot {expect_nodes}')
    return p.explain_in_natural_language() if mode == 'text' else p.to_add_edge_script()


def _selfcheck(dot_text: str, what: str, allow_empty: bool) -> None:
    g = parse_dot(dot_text)
    if not g.nodes and not allow_empty:
        raise StageError(f'{what}: simplified graph has no nodes')
    for s, t, _ in g.edges:
        if s not in g.nodes or t not in g.nodes:
            raise StageError(f'{what}: dangling edge {s}->{t}')


def derive_entry(dot_text: str, graph: str, code_mode: str) -> Dict[str, str]:
    dot_v2 = rewrite_dot_heads(dot_text, graph, code_mode)
    base_nodes = len(parse_dot(dot_v2).nodes)
    allow_empty = base_nodes == 0
    out = {'GraphDot-v2': dot_v2,
           'GraphText-v2': serialize(dot_v2, 'text', base_nodes),
           'GraphCode-v2': serialize(dot_v2, 'code', base_nodes)}
    dbc = dbc_v2(dot_v2, graph)
    _selfcheck(dbc, f'{graph}/dbc', allow_empty)
    out['DBC-v2-GraphDot'] = dbc
    out['DBC-v2-GraphText'] = serialize(dbc, 'text', len(parse_dot(dbc).nodes))
    for lv, frac in DBP_LEVELS.items():
        dbp = dbp_v2(dot_v2, frac)
        _selfcheck(dbp, f'{graph}/dbp-{lv}', allow_empty)
        out[f'DBP-{lv}-v2-GraphDot'] = dbp
        out[f'DBP-{lv}-v2-GraphText'] = serialize(dbp, 'text', len(parse_dot(dbp).nodes))
    return out


def _write_json_atomic(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp-', suffix='.json', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def repo_complete(root: str, repo_file: str, expected_paths) -> bool:
    expected = set(expected_paths)
    for rep in REPS:
        p = os.path.join(root, rep, repo_file)
        if not os.path.exists(p):
            return False
        try:
            data = json.load(open(p, encoding='utf-8'))
        except Exception:
            return False
        if set(data) != expected:
            return False
    return True


def process_repo(args) -> Dict:
    root, repo_file, force, code_mode = args
    res = {'repo': repo_file, 'entries': 0, 'skipped': False, 'failures': []}
    src_path = os.path.join(root, 'GraphDot', repo_file)
    try:
        src = json.load(open(src_path, encoding='utf-8'))
    except Exception as e:
        res['failures'].append({'repo': repo_file, 'path': '', 'graph': '', 'reason': f'GraphDot unreadable: {e}'})
        return res
    if not force and repo_complete(root, repo_file, src.keys()):
        res['skipped'] = True
        return res
    outputs: Dict[str, Dict[str, Dict[str, str]]] = {rep: {} for rep in REPS}
    for path, graphs in src.items():
        for rep in REPS:
            outputs[rep][path] = {}
        if not isinstance(graphs, dict):
            res['failures'].append({'repo': repo_file, 'path': path, 'graph': '', 'reason': 'value not dict'})
            continue
        for graph in GRAPHS:
            dot_text = graphs.get(graph)
            try:
                if not isinstance(dot_text, str):
                    raise StageError('graph missing in GraphDot')
                derived = derive_entry(dot_text, graph, code_mode)
            except StageError as e:
                res['failures'].append({'repo': repo_file, 'path': path, 'graph': graph, 'reason': str(e)})
                continue
            except Exception as e:
                res['failures'].append({'repo': repo_file, 'path': path, 'graph': graph,
                                        'reason': f'unexpected {type(e).__name__}: {e}'})
                continue
            for rep, text in derived.items():
                outputs[rep][path][graph] = text
            res['entries'] += 1
    for rep, data in outputs.items():
        _write_json_atomic(os.path.join(root, rep, repo_file), data)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-dir', default=os.path.join(CODE_DIR, 'dataset', 'manytypes4py'))
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--repos', nargs='*', default=None, help='只处理这些 <repo>.json')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--code-mode', choices=['head', 'full'], default='head')
    ap.add_argument('--log-dir', default=os.path.join(SCRIPT_DIR, 'logs', 'manytypes4py'))
    args = ap.parse_args()

    root = os.path.abspath(args.dataset_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    progress_log = os.path.join(args.log_dir, 'progress.log')
    failures_path = os.path.join(args.log_dir, 'failures.jsonl')

    repos = sorted(f for f in os.listdir(os.path.join(root, 'GraphDot')) if f.endswith('.json'))
    if args.repos:
        wanted = set(args.repos)
        repos = [r for r in repos if r in wanted]
    if args.limit:
        repos = repos[:args.limit]
    work = [(root, r, args.force, args.code_mode) for r in repos]
    print(f'[{now_str()}] repos={len(repos)} procs={args.procs} force={args.force} code_mode={args.code_mode}',
          file=sys.stderr, flush=True)
    with open(progress_log, 'a') as f:
        f.write(f'{now_str()} START repos={len(repos)} procs={args.procs} force={args.force} '
                f'code_mode={args.code_mode}\n')

    stats = collections.Counter()
    t0 = time.time()
    done = 0

    def flush(final=False):
        line = (f'{now_str()} {"DONE" if final else "PROG"} {done}/{len(repos)} entries={stats["entries"]} '
                f'skipped_repos={stats["skipped"]} failures={stats["failures"]} '
                f'elapsed={(time.time() - t0) / 60:.1f}min')
        print(line, file=sys.stderr, flush=True)
        with open(progress_log, 'a') as f:
            f.write(line + '\n')

    with mp.Pool(args.procs) as pool, open(failures_path, 'a') as ffail:
        for res in pool.imap_unordered(process_repo, work, chunksize=1):
            done += 1
            stats['entries'] += res['entries']
            stats['skipped'] += int(res['skipped'])
            for fl in res['failures']:
                stats['failures'] += 1
                fl['time'] = now_str()
                ffail.write(json.dumps(fl, ensure_ascii=False) + '\n')
            if res['failures']:
                ffail.flush()
            if done % 100 == 0:
                flush()
    flush(final=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
