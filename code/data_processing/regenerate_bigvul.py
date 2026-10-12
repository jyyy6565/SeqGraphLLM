#!/usr/bin/env python3
"""Big-Vul / PrimeVul 全量数据重建（PLAN.md Phase 1，决策 D9）。

一条龙：源码 → Joern CPG → 语句级抽子图 DOT → 修复版序列化 → v2 简化。
每个样本的每个 part（before/after）：

    <id>-<part>.c
      ① joern-parse（先按 .c；产出空壳则改 .cpp + MACRO_DEFINES 重解析；
         仍空壳则记 unrecoverable.txt）——判定逻辑同 scripts/rebuild_failed_graphs.py
         空壳判定：export 后 REACHING_DEF 边 == 0 且 CFG 边 <= 3
      ② joern-export --repr all --format dot   → <id>-CPG-v2-out/<part>/all/export.dot
      ③ 语句级抽子图（不丢多行节点；端点缺失即报错）
            cfg = CFG 边 / dfg = REACHING_DEF 边 / ast = AST 边
      ④ CPGParser（修复版）→ GraphText-v2 / GraphCode-v2
      ⑤ dbc_v2                → DBC-v2-{GraphDot,GraphText}
      ⑥ dbp_v2 light/medium/heavy → DBP-{light,medium,heavy}-v2-{GraphDot,GraphText}

工程要求（见 PLAN 3.1）：断点续跑、幂等原子写、Joern 超时 300s、失败写
failures.jsonl（绝不把错误文本写进产物）、按样本分片多进程、progress.log、
每个简化图产出后自检（parse_dot 非空 & GraphText 能过 CPGParser）。

用法：
    python regenerate_bigvul.py --dataset bigvul [--id-file ids] [--procs 8]
    python regenerate_bigvul.py --dataset primevul
    python regenerate_bigvul.py --dataset bigvul --ids 177736 177738   # 调试
    python regenerate_bigvul.py --dataset bigvul --force               # 无视已存在产物
默认 id 清单：bigvul = 全体样本目录 − code/ids.txt（= 6,342 实验子集）；
primevul = 全体 435。--all-ids 可强制取全体（用于 Big-Vul 4,558 排除样本的后台扩展重建）。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'scripts'))

from cpg_parser import CPGParser  # noqa: E402
from graph_simplify_v2 import DBP_LEVELS, dbc_v2, dbp_v2, parse_dot  # noqa: E402
from regenerate_manytypes4py import (  # noqa: E402
    _normalize_dot_statement,
    _split_dot_statements,
)
from rebuild_failed_graphs import MACRO_DEFINES  # noqa: E402  # 回退解析配方（可扩充）

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

GRAPHS = ('cfg', 'dfg', 'ast')
EDGE_LABEL = {'cfg': 'CFG', 'dfg': 'REACHING_DEF', 'ast': 'AST'}
PARTS = ('before', 'after')
JOERN_TIMEOUT = 300
STUB_CFG_MAX_EDGES = 3

REP_CPG = 'CPG-v2'
REP_DOT = 'GraphDot-v2'
REP_TEXT = 'GraphText-v2'
REP_CODE = 'GraphCode-v2'
REP_DBC_DOT = 'DBC-v2-GraphDot'
REP_DBC_TEXT = 'DBC-v2-GraphText'


def rep_dbp(level: str, fmt: str) -> str:
    return f'DBP-{level}-v2-Graph{fmt}'


DERIVED_REPS: List[Tuple[str, str]] = (
    [(REP_DOT, 'dot'), (REP_TEXT, 'txt'), (REP_CODE, 'graphcode.py'),
     (REP_DBC_DOT, 'dot'), (REP_DBC_TEXT, 'txt')]
    + [(rep_dbp(level, 'Dot'), 'dot') for level in DBP_LEVELS]
    + [(rep_dbp(level, 'Text'), 'txt') for level in DBP_LEVELS]
)

# Joern 导出的边语句形如  "a" -> "b" [label="CFG" ...]（label 总在属性首位）
EDGE_LABEL_RE = re.compile(r'->\s*"[^"]+"\s*\[\s*label="([A-Z_]+)"')


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def now_str() -> str:
    return _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def nonempty(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def rep_path(sdir: str, sid: str, rep: str, part: str, graph: str, suffix: str) -> str:
    return os.path.join(sdir, f'{sid}-{rep}-out', part, graph, f'{graph}.{suffix}')


def cpg_path(sdir: str, sid: str, part: str) -> str:
    return os.path.join(sdir, f'{sid}-{REP_CPG}-out', part, 'all', 'export.dot')


class StageError(Exception):
    def __init__(self, stage: str, reason: str):
        super().__init__(f'{stage}: {reason}')
        self.stage = stage
        self.reason = reason


# ---------------------------------------------------------------------------
# ① ② Joern
# ---------------------------------------------------------------------------

def edge_label_counts(cpg_dot: str) -> collections.Counter:
    return collections.Counter(EDGE_LABEL_RE.findall(cpg_dot))


def is_stub_cpg(cpg_dot: str) -> bool:
    c = edge_label_counts(cpg_dot)
    return c.get('REACHING_DEF', 0) == 0 and c.get('CFG', 0) <= STUB_CFG_MAX_EDGES


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _cleanup_source(text: str) -> str:
    """源码级清理：Big-Vul/PrimeVul 片段偶尔以块注释尾巴 ` */` 开头（截取函数时把上一条注释的
    结尾带进来了），会让 c2cpg 整段解析失败；去掉这种前导 `*/` 行。其他内容原样保留。"""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip() == '*/':
        return ''.join(lines[:i] + lines[i + 1:])
    return text


def run_joern(joern_cli: str, src_c: str, timeout: int) -> Tuple[Optional[str], Optional[str], str]:
    """回退策略解析：返回 (tag, cpg_dot_text, last_error)。

    策略（PLAN 3.1 ①，2026-08-15 加强）：
      1. 先按 .c 解析；若得到有数据流的 CPG（REACHING_DEF>0）→ 直接采用（tag 'c'）；
      2. 否则（空壳，或"有 CFG 但无 REACHING_DEF"的半解析）依次尝试 .c + MACRO_DEFINES
         （tag 'c-def'，签名/属性宏）与 .cpp + MACRO_DEFINES（tag 'cpp'，C++ 存成 .c）；
         在全部结果里选 (REACHING_DEF 边数, CFG 边数) 更大者，同分优先靠前者；
      3. 两次都是空壳 → 返回按 .c 的空壳（tag 'stub-c'），调用方记 unrecoverable
         （与旧数据对残缺 after 片段的处理一致）；
      None：Joern 本身失败/超时。"""
    joern_parse = os.path.join(joern_cli, 'joern-parse')
    joern_export = os.path.join(joern_cli, 'joern-export')
    last_err = ''
    candidates: List[Tuple[Tuple[int, int], str, str]] = []  # ((reaching_def, cfg), tag, text)
    with tempfile.TemporaryDirectory(prefix='bigvul-joern-') as tmp:
        define_args = [a for m in MACRO_DEFINES for a in ('--define', m)]
        attempts = [
            ('c', os.path.join(tmp, 'func.c'), []),
            ('c-def', os.path.join(tmp, 'func_def.c'), define_args),
            ('cpp', os.path.join(tmp, 'func.cpp'), define_args),
        ]
        src_text = _cleanup_source(open(src_c, encoding='utf-8', errors='replace').read())
        for tag, src, extra in attempts:
            with open(src, 'w', encoding='utf-8') as f:
                f.write(src_text)
            cpg = os.path.join(tmp, f'cpg_{tag}.bin')
            cmd = [joern_parse, src, '-o', cpg]
            if extra:
                cmd += ['--frontend-args'] + extra
            try:
                r = _run(cmd, timeout)
            except subprocess.TimeoutExpired:
                last_err = f'joern-parse timeout ({tag})'
                continue
            if r.returncode != 0 or not os.path.exists(cpg):
                last_err = f'joern-parse rc={r.returncode} ({tag}): {(r.stderr or r.stdout)[-300:]}'
                continue
            exp = os.path.join(tmp, f'export_{tag}')
            try:
                r = _run([joern_export, cpg, '--repr', 'all', '--format', 'dot', '-o', exp], timeout)
            except subprocess.TimeoutExpired:
                last_err = f'joern-export timeout ({tag})'
                continue
            dots = [f for f in os.listdir(exp)] if os.path.isdir(exp) else []
            dots = [f for f in dots if f.endswith('.dot')]
            if r.returncode != 0 or len(dots) != 1:
                last_err = f'joern-export rc={r.returncode} files={dots} ({tag}): {(r.stderr or r.stdout)[-300:]}'
                continue
            with open(os.path.join(exp, dots[0]), encoding='utf-8', errors='replace') as f:
                text = f.read()
            if not text.lstrip().startswith('digraph'):
                last_err = f'export not a digraph ({tag})'
                continue
            c = edge_label_counts(text)
            score = (c.get('REACHING_DEF', 0), c.get('CFG', 0))
            candidates.append((score, tag, text))
            if tag == 'c' and score[0] > 0:
                return 'c', text, ''  # 按 C 解析已有数据流，无需回退
    if not candidates:
        return None, None, last_err
    order = {'c': 2, 'c-def': 1, 'cpp': 0}
    score, tag, text = max(candidates, key=lambda x: (x[0], order[x[1]]))  # 同分优先靠前的尝试
    if is_stub_cpg(text):
        # 全是空壳：返回按 .c 的空壳（若有）
        for sc, tg, tx in candidates:
            if tg == 'c':
                return 'stub-c', tx, f'stub CPG (c and cpp)'
        return 'stub-c', text, 'stub CPG (cpp only)'
    return tag, text, ''


# ---------------------------------------------------------------------------
# ③ 语句级抽子图
# ---------------------------------------------------------------------------

_NODE_RE = re.compile(r'^\s*"([^"]+)"\s*\[(.*)\]\s*$', re.DOTALL)
_EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*\[(.*)\]\s*$', re.DOTALL)
_LABEL_RE = re.compile(r'\blabel\s*=\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
_CODE_RE = re.compile(r'\bCODE\s*=\s*"((?:\\.|[^"\\])*)"', re.DOTALL)

# --- 复合节点 CODE 缩写（决策 D11，--code-mode head，默认） -----------------
# Joern 给 METHOD / BLOCK / CONTROL_STRUCTURE 等复合节点的 CODE 是其**整段源码**
# （METHOD = 整个函数，BLOCK = 整个块体，if/for = 头部+整个体）。旧管线因逐行解析
# bug 把这些多行节点整个丢掉，所以论文旧数据里事实上没有它们；如今保留节点但
# 原样保留 CODE 会让每个图表示把函数源码重复 1.5–4 倍（pilot 实测 GraphText token
# cfg +23%、ast +44%），且提示词里源码总是单独给出。因此 head 模式把复合节点的
# CODE 缩成"头部"：
#   METHOD            → 签名（到第一个 '{' 之前）+ " { ... }"
#   CONTROL_STRUCTURE → 头部（到第一个 '{' 或换行之前）+ " { ... }"（有体时）
#   BLOCK             → "{ ... }"（以 '{' 开头时）；无花括号的多行块体 → 首行 + " ..."；
#                        <empty> 等单行保持不变
#   JUMP_TARGET       → 标签行（第一行）
#   TYPE_DECL / NAMESPACE_BLOCK / UNKNOWN → 同 CONTROL_STRUCTURE 规则
#   其他节点（CALL/LITERAL/RETURN…）多行 CODE → 空白折叠为单行（内容完整保留）
# 结果：所有节点 CODE 都是单行；子节点仍完整表达体内内容。full 模式则原样保留。
HEAD_KINDS_BRACE = {'METHOD', 'CONTROL_STRUCTURE', 'TYPE_DECL', 'NAMESPACE_BLOCK', 'UNKNOWN'}
_WS_RE = re.compile(r'\s+')


def dot_unescape(value: str) -> str:
    table = {'n': '\n', 'r': '\r', 't': '\t', '"': '"', '\\': '\\'}
    return re.sub(r'\\(.)', lambda m: table.get(m.group(1), m.group(0)), value, flags=re.DOTALL)


def dot_escape(value: str) -> str:
    return (value.replace('\\', '\\\\').replace('"', '\\"')
            .replace('\n', '\\n').replace('\r', '\\r'))


def abbreviate_code(kind: str, code: str) -> str:
    """按 head 规则缩写节点 CODE（见上方说明）。总是返回单行。"""
    if not code:
        return code
    if kind == 'BLOCK':
        if code.lstrip().startswith('{'):
            return '{ ... }'
        if '\n' in code:  # Python 等无花括号语言：块体 = 语句序列 → 首行 + " ..."
            return code.split('\n', 1)[0].strip() + ' ...'
        return code
    if kind == 'JUMP_TARGET':
        return code.split('\n', 1)[0].strip()
    if kind in HEAD_KINDS_BRACE:
        brace = code.find('{')
        if kind == 'METHOD':
            head = code if brace < 0 else code[:brace]
            has_body = brace >= 0
        else:
            nl = code.find('\n')
            cut = min(x for x in (brace, nl) if x >= 0) if (brace >= 0 or nl >= 0) else -1
            head = code if cut < 0 else code[:cut]
            has_body = brace >= 0 or nl >= 0
        head = _WS_RE.sub(' ', head).strip()
        if not head:  # 例如 CODE 直接以 '{' 开头
            head = code.split('\n', 1)[0].strip()
            return head if head else code[:60]
        return head + ' { ... }' if has_body else head
    return _WS_RE.sub(' ', code).strip() if '\n' in code else code


def abbreviate_node_stmt(stmt: str) -> str:
    """对一条（已单行化的）节点语句重写 CODE 属性。"""
    lm = _LABEL_RE.search(stmt)
    cm = _CODE_RE.search(stmt)
    if not lm or not cm:
        return stmt
    kind = lm.group(1)
    code = dot_unescape(cm.group(1))
    new_code = abbreviate_code(kind, code)
    if new_code == code:
        return stmt
    return stmt[:cm.start(1)] + dot_escape(new_code) + stmt[cm.end(1):]


def extract_subgraphs(cpg_dot: str, code_mode: str = 'head') -> Dict[str, str]:
    """从完整 CPG DOT 抽 cfg/dfg/ast 三个子图（节点语句单行化；head 模式缩写复合节点 CODE）。"""
    nodes: 'collections.OrderedDict[str, str]' = collections.OrderedDict()
    edges: List[Tuple[str, str, str, str]] = []
    for raw in _split_dot_statements(cpg_dot):
        stmt = _normalize_dot_statement(raw)
        em = _EDGE_RE.match(stmt)
        if em:
            s, t, attrs = em.groups()
            lm = _LABEL_RE.search(attrs)
            edges.append((s, t, lm.group(1) if lm else '', stmt))
            continue
        nm = _NODE_RE.match(stmt)
        if nm:
            nodes[nm.group(1)] = abbreviate_node_stmt(stmt) if code_mode == 'head' else stmt
    out: Dict[str, str] = {}
    for graph in GRAPHS:
        label = EDGE_LABEL[graph]
        sel = [e for e in edges if e[2] == label]
        involved = {n for e in sel for n in e[:2]}
        missing = involved.difference(nodes)
        if missing:
            raise StageError('extract', f'{graph}: {len(missing)} endpoint nodes missing')
        lines = [f'digraph {graph} {{']
        lines.extend(f'  {nodes[n]};' for n in nodes if n in involved)
        lines.extend(f'  {e[3]};' for e in sel)
        lines.append('}')
        out[graph] = '\n'.join(lines) + '\n'
    return out


# ---------------------------------------------------------------------------
# ④ ⑤ ⑥ 派生
# ---------------------------------------------------------------------------

def serialize(dot_text: str, mode: str, expect_nodes: Optional[int] = None) -> str:
    """CPGParser 序列化；expect_nodes 给定时校验节点数严格相等（跨表示一致性自检）。"""
    p = CPGParser()
    p.parse_dot(dot_text)
    if expect_nodes is not None and len(p.nodes) != expect_nodes:
        raise StageError('selfcheck', f'serializer node count {len(p.nodes)} != dot {expect_nodes}')
    return p.explain_in_natural_language() if mode == 'text' else p.to_add_edge_script()


def _selfcheck_dot(dot_text: str, what: str, allow_empty: bool) -> None:
    g = parse_dot(dot_text)
    if not g.nodes and not allow_empty:
        raise StageError('selfcheck', f'{what}: simplified graph has no nodes')
    ids = set(g.nodes)
    for s, t, _ in g.edges:
        if s not in ids or t not in ids:
            raise StageError('selfcheck', f'{what}: dangling edge {s}->{t}')


def derive_all(sdir: str, sid: str, part: str, dot_graphs: Dict[str, str], force: bool) -> None:
    """由子图 DOT 生成全部派生表示（目标已存在且非空则跳过，除非 force）。"""
    for graph, dot_text in dot_graphs.items():
        base_nodes = len(parse_dot(dot_text).nodes)
        allow_empty = base_nodes == 0  # 原图空（如 dfg 无 REACHING_DEF）时简化图也允许空
        targets: Dict[str, str] = {}

        def need(rep: str, suffix: str) -> bool:
            p = rep_path(sdir, sid, rep, part, graph, suffix)
            return force or not nonempty(p)

        if need(REP_DOT, 'dot'):
            targets[rep_path(sdir, sid, REP_DOT, part, graph, 'dot')] = dot_text
        if need(REP_TEXT, 'txt'):
            targets[rep_path(sdir, sid, REP_TEXT, part, graph, 'txt')] = serialize(dot_text, 'text', base_nodes)
        if need(REP_CODE, 'graphcode.py'):
            targets[rep_path(sdir, sid, REP_CODE, part, graph, 'graphcode.py')] = serialize(dot_text, 'code', base_nodes)

        if need(REP_DBC_DOT, 'dot') or need(REP_DBC_TEXT, 'txt'):
            dbc_dot = dbc_v2(dot_text, graph)
            _selfcheck_dot(dbc_dot, f'{graph}/dbc', allow_empty)
            dbc_txt = serialize(dbc_dot, 'text', len(parse_dot(dbc_dot).nodes))
            if need(REP_DBC_DOT, 'dot'):
                targets[rep_path(sdir, sid, REP_DBC_DOT, part, graph, 'dot')] = dbc_dot
            if need(REP_DBC_TEXT, 'txt'):
                targets[rep_path(sdir, sid, REP_DBC_TEXT, part, graph, 'txt')] = dbc_txt

        for level, frac in DBP_LEVELS.items():
            rd, rt = rep_dbp(level, 'Dot'), rep_dbp(level, 'Text')
            if need(rd, 'dot') or need(rt, 'txt'):
                dbp_dot = dbp_v2(dot_text, frac)
                _selfcheck_dot(dbp_dot, f'{graph}/dbp-{level}', allow_empty)
                dbp_txt = serialize(dbp_dot, 'text', len(parse_dot(dbp_dot).nodes))
                if need(rd, 'dot'):
                    targets[rep_path(sdir, sid, rd, part, graph, 'dot')] = dbp_dot
                if need(rt, 'txt'):
                    targets[rep_path(sdir, sid, rt, part, graph, 'txt')] = dbp_txt

        for path, text in targets.items():
            atomic_write(path, text)


# ---------------------------------------------------------------------------
# 单样本工作单元
# ---------------------------------------------------------------------------

def part_complete(sdir: str, sid: str, part: str) -> bool:
    if not nonempty(cpg_path(sdir, sid, part)):
        return False
    for rep, suffix in DERIVED_REPS:
        for graph in GRAPHS:
            if not nonempty(rep_path(sdir, sid, rep, part, graph, suffix)):
                return False
    return True


def process_sample(args) -> Dict:
    root, sid, joern_cli, timeout, force, force_derived, code_mode = args
    sdir = os.path.join(root, sid)
    result = {'id': sid, 'parts': {}, 'failures': [], 'unrecoverable': [], 'partial': []}
    for part in PARTS:
        info = {'status': 'ok', 'joern': 'cached'}
        try:
            src = os.path.join(sdir, f'{sid}-{part}.c')
            if not os.path.exists(src):
                raise StageError('source', 'source .c missing')
            if not (force or force_derived) and part_complete(sdir, sid, part):
                info['status'] = 'skipped'
                result['parts'][part] = info
                continue

            cpath = cpg_path(sdir, sid, part)
            if force or not nonempty(cpath):
                tag, cpg_dot, err = run_joern(joern_cli, src, timeout)
                if cpg_dot is None:
                    raise StageError('joern', err)
                if tag.startswith('stub'):
                    # 空壳 CPG：仍写出（派生表示为近空图），并记 unrecoverable 供论文披露/排除
                    result['unrecoverable'].append(part)
                elif edge_label_counts(cpg_dot).get('REACHING_DEF', 0) == 0:
                    # 半解析：有 CFG 但无数据流（c/cpp 都无法恢复，多为 C++ 类成员函数/构造函数）→ 记 partial
                    result['partial'].append(part)
                atomic_write(cpath, cpg_dot)
                info['joern'] = tag
            else:
                with open(cpath, encoding='utf-8', errors='replace') as f:
                    cpg_dot = f.read()

            dot_graphs = extract_subgraphs(cpg_dot, code_mode)
            derive_all(sdir, sid, part, dot_graphs, force or force_derived)
            info['nodes'] = {g: len(parse_dot(d).nodes) for g, d in dot_graphs.items()}
        except StageError as e:
            info['status'] = 'failed'
            result['failures'].append({'id': sid, 'part': part, 'stage': e.stage, 'reason': e.reason})
        except Exception as e:  # 未预期异常：记录 traceback 尾部
            info['status'] = 'failed'
            result['failures'].append({'id': sid, 'part': part, 'stage': 'unexpected',
                                       'reason': f'{type(e).__name__}: {e} | '
                                                 f'{traceback.format_exc()[-600:]}'})
        result['parts'][part] = info
    return result


# ---------------------------------------------------------------------------
# id 清单
# ---------------------------------------------------------------------------

def load_ids(root: str, dataset: str, id_file: Optional[str], ids: Optional[List[str]],
             all_ids: bool) -> List[str]:
    if ids:
        return [str(i) for i in ids]
    if id_file:
        with open(id_file) as f:
            return [l.strip() for l in f if l.strip()]
    every = sorted(d for d in os.listdir(root) if d.isdigit())
    if dataset == 'bigvul' and not all_ids:
        excl_path = os.path.join(CODE_DIR, 'ids.txt')
        excl = set(l.strip() for l in open(excl_path) if l.strip()) if os.path.exists(excl_path) else set()
        every = [i for i in every if i not in excl]
    return every


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['bigvul', 'primevul'], default='bigvul')
    ap.add_argument('--dataset-dir', default=None, help='默认 code/dataset/<dataset>')
    ap.add_argument('--id-file', default=None)
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--all-ids', action='store_true', help='bigvul 不排除 ids.txt')
    ap.add_argument('--joern', default=os.path.expanduser('~/bin/joern/joern-cli'))
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=JOERN_TIMEOUT)
    ap.add_argument('--force', action='store_true', help='无视已存在产物全部重做（含 Joern）')
    ap.add_argument('--force-derived', action='store_true',
                    help='只强制重做派生层（③-⑥），复用已有 CPG-v2/export.dot')
    ap.add_argument('--code-mode', choices=['head', 'full'], default='head',
                    help='复合节点 CODE 处理：head=缩成头部（默认，见源码说明）；full=原样保留')
    ap.add_argument('--log-dir', default=None, help='progress.log / failures.jsonl / unrecoverable.txt 目录；'
                                                  '默认 code/data_processing/logs/<dataset>')
    ap.add_argument('--tag', default='', help='日志文件名后缀（区分批次）')
    args = ap.parse_args()

    root = os.path.abspath(args.dataset_dir or os.path.join(CODE_DIR, 'dataset', args.dataset))
    log_dir = args.log_dir or os.path.join(SCRIPT_DIR, 'logs', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    suffix = f'-{args.tag}' if args.tag else ''
    progress_log = os.path.join(log_dir, f'progress{suffix}.log')
    failures_path = os.path.join(log_dir, f'failures{suffix}.jsonl')
    unrec_path = os.path.join(log_dir, f'unrecoverable{suffix}.txt')
    partial_path = os.path.join(log_dir, f'partial{suffix}.txt')

    ids = load_ids(root, args.dataset, args.id_file, args.ids, args.all_ids)
    work = [(root, sid, args.joern, args.timeout, args.force, args.force_derived, args.code_mode)
            for sid in ids]
    print(f'[{now_str()}] dataset={args.dataset} root={root} ids={len(ids)} procs={args.procs} '
          f'force={args.force} code_mode={args.code_mode}', file=sys.stderr, flush=True)
    with open(progress_log, 'a') as f:
        f.write(f'{now_str()} START ids={len(ids)} procs={args.procs} force={args.force} code_mode={args.code_mode}\n')

    stats = collections.Counter()
    t0 = time.time()
    done = 0

    def flush_progress(final: bool = False) -> None:
        elapsed = time.time() - t0
        line = (f'{now_str()} {"DONE" if final else "PROG"} {done}/{len(ids)} '
                f'ok_parts={stats["ok"]} skipped_parts={stats["skipped"]} '
                f'failed_parts={stats["failed"]} unrecoverable_parts={stats["unrec"]} partial_nodfg={stats["partial"]} '
                f'joern_c={stats["joern_c"]} joern_cdef={stats["joern_c_def"]} joern_cpp={stats["joern_cpp"]} '
                f'joern_stub={stats["joern_stub_c"]} '
                f'elapsed={elapsed / 60:.1f}min')
        print(line, file=sys.stderr, flush=True)
        with open(progress_log, 'a') as f:
            f.write(line + '\n')

    with mp.Pool(args.procs) as pool, \
            open(failures_path, 'a') as ffail, open(unrec_path, 'a') as funrec, \
            open(partial_path, 'a') as fpartial:
        for res in pool.imap_unordered(process_sample, work, chunksize=1):
            done += 1
            for part, info in res['parts'].items():
                stats[info['status']] += 1
                if info.get('joern') in ('c', 'c-def', 'cpp', 'stub-c'):
                    stats[f'joern_{info["joern"].replace("-", "_")}'] += 1
            for part in res['unrecoverable']:
                stats['unrec'] += 1
                funrec.write(f'{res["id"]}\t{part}\n')
            for part in res.get('partial', []):
                stats['partial'] += 1
                fpartial.write(f'{res["id"]}\t{part}\n')
            for fl in res['failures']:
                fl['time'] = now_str()
                ffail.write(json.dumps(fl, ensure_ascii=False) + '\n')
            if res['failures'] or res['unrecoverable'] or res.get('partial'):
                ffail.flush()
                funrec.flush()
                fpartial.flush()
            if done % 100 == 0:
                flush_progress()
    flush_progress(final=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
