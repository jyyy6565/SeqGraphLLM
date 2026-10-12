"""图简化器 v2 —— DBP / DBC 的统一重实现（RQ3 重跑用，替换旧版语义）。

设计动机见 docs/ROOT_CAUSE_REPORT.md 与 docs/PIPELINE_REVIEW.md：
旧版 DBP 用绝对度数阈值一次性过滤，在链式 CFG 上等于删除整图（空图率 63-95%），
在 AST 上反向保留 '<empty>' 结构桩；旧版 DBC 只做叶子折叠，对 CFG 是恒等变换。

v2 语义（文档化，跨图类型一致）：

DBP v2（度数剪枝，Degree-Based Pruning）
    按无向唯一邻居度数对节点排名，保留度数最高的 keep_fraction 比例
    （档位 light/medium/heavy = 50% / 25% / 10%），'<empty>' 节点不参与配额，
    输出诱导子图（两端都保留的边）。保底：非空图至少保留 3 个节点。

DBC v2（度数压缩，Degree-Based Compression）
    合并而非删除，保语义地缩图：
    - 叶子折叠（AST/DFG 有效）：度数=1 的节点并入其唯一邻居
      （邻居度数 >= fold_min_degree，默认 5），邻居标注 "[N nodes merged]"；
    - 链收缩（CFG 有效）：入度=出度=1 的连续直线链合并为一个 BLOCK 节点
      （basic block，编译器标准做法），代码按顺序拼接、保留首末行号。

BUDGET 模式（等 token 预算，回应 R1-4 的等预算对照）
    按度数从高到低贪心加入节点，直到序列化 token 估算达到预算上限。

输入输出均为 Joern 导出的单视图 DOT（CPG 属性风格），可与
cpg_parser.CPGParser 的 GraphText / GraphCode 序列化直接衔接。
空图输入输出 DOT 骨架（digraph 头 + 尾），绝不产出 0 字节文件。
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# DOT 解析（语句级，容忍带引号值内的换行/分号——沿用 regenerate_manytypes4py 的做法）
# ---------------------------------------------------------------------------

NODE_RE = re.compile(r'^\s*"([^"]+)"\s*\[(.*)\]\s*$', re.DOTALL)
EDGE_RE = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*(?:\[(.*)\])?\s*$', re.DOTALL)
CODE_ATTR_RE = re.compile(r'\bCODE\s*=\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
LABEL_ATTR_RE = re.compile(r'\blabel\s*=\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
LINE_ATTR_RE = re.compile(r'\bLINE_NUMBER\s*=\s*"([^"]*)"')

EMPTY_MARKERS = ('<empty>', '&lt;empty&gt;')


def _split_statements(dot_content: str) -> Iterable[str]:
    """按引号外的分号切分 DOT 语句体。"""
    body_start = dot_content.find('{')
    body_end = dot_content.rfind('}')
    if body_start < 0:
        return
    body = dot_content[body_start + 1: body_end if body_end > body_start else None]
    current: List[str] = []
    quoted = False
    escaped = False
    for ch in body:
        if ch == '"' and not escaped:
            quoted = not quoted
        if ch == ';' and not quoted:
            stmt = ''.join(current).strip()
            if stmt:
                yield stmt
            current = []
            escaped = False
            continue
        current.append(ch)
        escaped = (ch == '\\' and not escaped)


@dataclass
class Graph:
    name: str = 'graph'
    # id -> 原始节点语句（含全部属性，重建时原样保留）
    nodes: 'OrderedDict[str, str]' = field(default_factory=OrderedDict)
    # (src, dst, 原始边语句)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)

    # -- 结构查询 -----------------------------------------------------------
    def out_neighbors(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = defaultdict(set)
        for s, t, _ in self.edges:
            out[s].add(t)
        return out

    def in_neighbors(self) -> Dict[str, Set[str]]:
        inc: Dict[str, Set[str]] = defaultdict(set)
        for s, t, _ in self.edges:
            inc[t].add(s)
        return inc

    def undirected_degree(self) -> Dict[str, int]:
        """无向唯一邻居数。"""
        nbr: Dict[str, Set[str]] = defaultdict(set)
        for s, t, _ in self.edges:
            if s != t:
                nbr[s].add(t)
                nbr[t].add(s)
        return {nid: len(nbr.get(nid, ())) for nid in self.nodes}

    def node_code(self, nid: str) -> str:
        stmt = self.nodes.get(nid, '')
        m = CODE_ATTR_RE.search(stmt)
        return m.group(1) if m else ''

    def is_empty_stub(self, nid: str) -> bool:
        code = self.node_code(nid)
        return any(marker in code for marker in EMPTY_MARKERS) or code == ''


def parse_dot(dot_content: str) -> Graph:
    g = Graph()
    m = re.search(r'digraph\s+"?([^"{\s]*)"?\s*\{', dot_content or '')
    if m and m.group(1):
        g.name = m.group(1)
    if not dot_content or '{' not in dot_content:
        return g
    for stmt in _split_statements(dot_content):
        em = EDGE_RE.match(stmt)
        if em:
            g.edges.append((em.group(1), em.group(2), stmt))
            continue
        nm = NODE_RE.match(stmt)
        if nm:
            g.nodes[nm.group(1)] = stmt
    return g


def rebuild_dot(g: Graph, keep_nodes: Set[str],
                extra_node_stmts: Optional[List[str]] = None,
                edge_stmts: Optional[List[str]] = None) -> str:
    """由保留节点（诱导子图）+ 可选合成节点/边重建 DOT。永不返回空串。"""
    lines = [f'digraph {g.name} {{']
    for nid, stmt in g.nodes.items():
        if nid in keep_nodes:
            lines.append(f'  {stmt};')
    for stmt in (extra_node_stmts or []):
        lines.append(f'  {stmt};')
    if edge_stmts is None:
        for s, t, stmt in g.edges:
            if s in keep_nodes and t in keep_nodes:
                lines.append(f'  {stmt};')
    else:
        for stmt in edge_stmts:
            lines.append(f'  {stmt};')
    lines.append('}')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# DBP v2：按度数排名的配额剪枝
# ---------------------------------------------------------------------------

DBP_LEVELS = {'light': 0.50, 'medium': 0.25, 'heavy': 0.10}
MIN_KEEP = 3


def dbp_v2(dot_content: str, keep_fraction: float) -> str:
    """保留无向度数最高的 keep_fraction 节点；'<empty>' 桩不占配额；输出诱导子图。"""
    g = parse_dot(dot_content)
    n = len(g.nodes)
    if n == 0:
        return rebuild_dot(g, set())
    deg = g.undirected_degree()

    candidates = [nid for nid in g.nodes if not g.is_empty_stub(nid)]
    pool = candidates if candidates else list(g.nodes)
    quota = max(min(len(pool), MIN_KEEP), round(len(pool) * keep_fraction))
    # 度数降序，tie 按文档出现顺序（Python sort 稳定，pool 本身即文档序）
    ranked = sorted(pool, key=lambda nid: -deg.get(nid, 0))
    keep = set(ranked[:quota])
    return rebuild_dot(g, keep)


# ---------------------------------------------------------------------------
# DBC v2：叶子折叠 + 链收缩
# ---------------------------------------------------------------------------

def _fold_leaves(g: Graph, fold_min_degree: int) -> Tuple[Set[str], Dict[str, int]]:
    """迭代把度数=1 的节点并入唯一邻居（邻居度数>=fold_min_degree）。

    返回 (被合并节点集, 幸存节点的合并计数)。计数语义：值 = 该节点吸收的节点数。
    """
    merged: Set[str] = set()
    absorbed: Dict[str, int] = defaultdict(int)
    while True:
        nbr: Dict[str, Set[str]] = defaultdict(set)
        for s, t, _ in g.edges:
            if s in merged or t in merged or s == t:
                continue
            nbr[s].add(t)
            nbr[t].add(s)
        changed = False
        for nid in list(g.nodes):
            if nid in merged:
                continue
            neighbors = nbr.get(nid, set())
            if len(neighbors) != 1:
                continue
            neighbor = next(iter(neighbors))
            if neighbor in merged or len(nbr.get(neighbor, ())) < fold_min_degree:
                continue
            merged.add(nid)
            absorbed[neighbor] += 1 + absorbed.pop(nid, 0)
            changed = True
        if not changed:
            return merged, dict(absorbed)


def _annotate_merge_count(stmt: str, count: int) -> str:
    """在节点语句的 CODE（其次 label）属性后追加合并标注。"""
    note = f' [{count} nodes merged]'

    def _append(match: re.Match) -> str:
        return match.group(0)[:-1] + note + '"'

    new_stmt, replaced = CODE_ATTR_RE.subn(_append, stmt, count=1)
    if replaced:
        return new_stmt
    new_stmt, replaced = LABEL_ATTR_RE.subn(_append, stmt, count=1)
    return new_stmt if replaced else stmt


def _contract_chains(g: Graph, alive: Set[str]) -> Tuple[Set[str], List[str], List[str]]:
    """把 alive 节点中入度=出度=1 的直线链合并为 BLOCK 节点。

    返回 (链内被合并节点集, 合成节点语句, 全量边语句)。
    """
    out_n = {k: {t for t in v if t in alive} for k, v in g.out_neighbors().items() if k in alive}
    in_n = {k: {s for s in v if s in alive} for k, v in g.in_neighbors().items() if k in alive}

    def is_internal(nid: str) -> bool:
        return len(out_n.get(nid, ())) == 1 and len(in_n.get(nid, ())) == 1

    visited: Set[str] = set()
    chains: List[List[str]] = []
    for nid in g.nodes:
        if nid not in alive or nid in visited or not is_internal(nid):
            continue
        # 向上游走到链头
        head = nid
        while True:
            prev = next(iter(in_n[head]))
            if prev in alive and is_internal(prev) and prev not in visited and prev != nid:
                head = prev
            else:
                break
        # 从链头向下游收集
        chain = []
        cur = head
        while cur in alive and is_internal(cur) and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            cur = next(iter(out_n[cur]))
            if cur == head:  # 环保护
                break
        if len(chain) >= 2:
            chains.append(chain)

    if not chains:
        return set(), [], []

    chain_of: Dict[str, str] = {}
    extra_stmts: List[str] = []
    removed: Set[str] = set()
    for idx, chain in enumerate(chains):
        block_id = f'blk{idx}'
        codes = [g.node_code(nid) or '<stmt>' for nid in chain]
        first_line_m = LINE_ATTR_RE.search(g.nodes[chain[0]])
        last_line_m = LINE_ATTR_RE.search(g.nodes[chain[-1]])
        attrs = [f'label="BLOCK"']
        # node_code() 返回的是 DOT 属性原文（已转义，如 \" 与 \\），直接拼接即可；
        # 旧版再对 '"' 转义一次会产生 \\" 从而提前终止字符串、BLOCK 节点解析失败
        joined = '\\n'.join(codes)
        attrs.append(f'CODE="{joined}"')
        if first_line_m:
            attrs.append(f'LINE_NUMBER="{first_line_m.group(1)}"')
        if last_line_m and (not first_line_m or last_line_m.group(1) != first_line_m.group(1)):
            attrs.append(f'LINE_NUMBER_END="{last_line_m.group(1)}"')
        attrs.append(f'MERGED="{len(chain)}"')
        extra_stmts.append(f'"{block_id}" [{" ".join(attrs)}]')
        for nid in chain:
            chain_of[nid] = block_id
            removed.add(nid)

    # 重写边：链内部边丢弃，跨界边重定向到 BLOCK
    edge_stmts: List[str] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for s, t, stmt in g.edges:
        if s not in alive or t not in alive:
            continue
        ns = chain_of.get(s, s)
        nt = chain_of.get(t, t)
        if ns == nt and (s in removed or t in removed):
            continue  # 链内部边
        if ns == s and nt == t:
            edge_stmts.append(stmt)
            continue
        if (ns, nt) in seen_pairs:
            continue
        seen_pairs.add((ns, nt))
        lm = LABEL_ATTR_RE.search(stmt)
        label = lm.group(1) if lm else 'CFG'
        edge_stmts.append(f'"{ns}" -> "{nt}" [label="{label}"]')
    return removed, extra_stmts, edge_stmts


def dbc_v2(dot_content: str, graph_type: str, fold_min_degree: int = 5) -> str:
    """度数压缩 v2：AST/DFG 叶子折叠；CFG 链收缩（也做叶子折叠，通常无叶可折）。"""
    g = parse_dot(dot_content)
    if not g.nodes:
        return rebuild_dot(g, set())

    merged, absorbed = _fold_leaves(g, fold_min_degree)
    alive = {nid for nid in g.nodes if nid not in merged}

    extra_stmts: List[str] = []
    edge_stmts: Optional[List[str]] = None
    if graph_type.lower() == 'cfg':
        chain_removed, extra_stmts, edge_stmts = _contract_chains(g, alive)
        alive -= chain_removed
    if edge_stmts is None:
        edge_stmts = [stmt for s, t, stmt in g.edges if s in alive and t in alive]

    # 重建：幸存节点带合并计数标注
    lines = [f'digraph {g.name} {{']
    for nid, stmt in g.nodes.items():
        if nid not in alive:
            continue
        if nid in absorbed:
            stmt = _annotate_merge_count(stmt, absorbed[nid])
        lines.append(f'  {stmt};')
    for stmt in extra_stmts:
        lines.append(f'  {stmt};')
    for stmt in edge_stmts:
        lines.append(f'  {stmt};')
    lines.append('}')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# BUDGET 模式：等 token 预算剪枝
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // 4)


def budget_prune(dot_content: str, token_budget: int) -> str:
    """按度数降序贪心加入节点，直到诱导子图 DOT 的 token 估算达到预算。"""
    g = parse_dot(dot_content)
    if not g.nodes:
        return rebuild_dot(g, set())
    deg = g.undirected_degree()
    ranked = sorted(
        (nid for nid in g.nodes if not g.is_empty_stub(nid)),
        key=lambda nid: -deg.get(nid, 0),
    ) or list(g.nodes)

    edges_by_node: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for e in g.edges:
        edges_by_node[e[0]].append(e)
        edges_by_node[e[1]].append(e)

    keep: Set[str] = set()
    used = _count_tokens(f'digraph {g.name} {{}}')
    counted_edges: Set[int] = set()
    for nid in ranked:
        cost = _count_tokens(g.nodes[nid])
        pending: List[int] = []
        for e in edges_by_node.get(nid, ()):  # 该节点加入后新形成的诱导边
            eidx = id(e)
            other = e[1] if e[0] == nid else e[0]
            if eidx not in counted_edges and (other in keep or other == nid):
                cost += _count_tokens(e[2])
                pending.append(eidx)
        if used + cost > token_budget and len(keep) >= MIN_KEEP:
            continue
        keep.add(nid)
        counted_edges.update(pending)
        used += cost
        if used >= token_budget:
            break
    return rebuild_dot(g, keep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description='图简化器 v2（DBP/DBC/BUDGET）')
    ap.add_argument('input', help='输入 DOT 文件路径，- 表示 stdin')
    ap.add_argument('--mode', choices=['dbp', 'dbc', 'budget'], required=True)
    ap.add_argument('--graph-type', choices=['cfg', 'dfg', 'ddg', 'ast'], required=True)
    ap.add_argument('--level', choices=list(DBP_LEVELS), default='medium',
                    help='dbp 档位：light=保50%% / medium=保25%% / heavy=保10%%')
    ap.add_argument('--keep-fraction', type=float, default=None,
                    help='dbp 自定义保留比例（覆盖 --level）')
    ap.add_argument('--fold-min-degree', type=int, default=5, help='dbc 叶子折叠的邻居度数下限')
    ap.add_argument('--budget', type=int, default=2048, help='budget 模式的 token 预算')
    ap.add_argument('-o', '--output', default='-', help='输出路径，- 表示 stdout')
    args = ap.parse_args()

    content = sys.stdin.read() if args.input == '-' else open(args.input, encoding='utf-8').read()
    if args.mode == 'dbp':
        frac = args.keep_fraction if args.keep_fraction is not None else DBP_LEVELS[args.level]
        result = dbp_v2(content, frac)
    elif args.mode == 'dbc':
        result = dbc_v2(content, args.graph_type, args.fold_min_degree)
    else:
        result = budget_prune(content, args.budget)

    if args.output == '-':
        sys.stdout.write(result)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)


if __name__ == '__main__':
    main()
