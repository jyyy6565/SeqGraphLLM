"""预算内图裁剪（budget-fit）—— 替代对图序列化文本的尾部截断（2026-08-16 协议变更）。

动机：GraphText/GraphCode 都是"先全部 Nodes 再全部 Edges"，模型窗口装不下时尾部截断会先把边
全部砍掉，模型看到的其实是"节点清单"而不是图。budget-fit 改为：图放得下就原样给；放不下则用
graph_simplify_v2.budget_prune（无向度数降序贪心加节点，诱导子图，确定性）把 DOT 裁到预算内，
再序列化成目标格式（text / dot / graphcode）。裁剪程度以 kept_fraction（保留节点/总节点）记录。

对外接口：
    serialize_dot(dot_text, fmt)                       -> str
    fit_graph(dot_text, fmt, budget_tokens, full_text=None) -> FitResult
    fit_graphs([(dot_text, fmt, full_text|None), ...], total_budget) -> [FitResult]  # 多图按完整大小等比例分预算
    count_tokens(text) -> int                          # tiktoken cl100k_base，与 truncate_messages 一致

确定性：同一 (dot, fmt, budget) 永远给出同一输出（度数排名 tie 按文档序，迭代步长固定）。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cpg_parser import CPGParser  # noqa: E402
from graph_simplify_v2 import MIN_KEEP, budget_prune, parse_dot  # noqa: E402

# ---------------- token 计数 ----------------
# 默认 tiktoken(cl100k_base)，但它只是近似：CodeLlama/Qwen 的分词器对 C 代码通常给出更多 token，
# 低估会让请求真正超过 vLLM 的 max_model_len 而被服务端 400 拒绝（客户端记成"无效响应"，
# 系统性地伤害图增强配置）。设环境变量 SEQGRAPH_TOKENIZER=<模型目录> 即改用该模型的真实分词器。
def _spec() -> str:
    """每次读环境变量：主进程在解析完参数后才知道模型路径（子进程 fork 继承）。"""
    return os.environ.get('SEQGRAPH_TOKENIZER', '')

# 超过该字符数的文本必然远超任何模型窗口（32k token ≈ 12 万字符），只需量级，不必真的分词——
# 真分词一个 6 MB 的 dfg_dot 要好几秒，而它离预算差着两个数量级。
FAST_ESTIMATE_CHARS = 400_000
_HF_CACHE = {}


def _hf_tokenizer():
    spec = _spec()
    tok = _HF_CACHE.get(spec)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec, use_fast=True)
        tok.model_max_length = int(1e9)  # 关掉"超长"告警
        _HF_CACHE[spec] = tok
    return tok


try:
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')

    def _count_exact(text: str) -> int:
        if _spec():
            return len(_hf_tokenizer()(text, add_special_tokens=False)['input_ids'])
        return len(_ENC.encode(text, disallowed_special=()))
except Exception:  # pragma: no cover
    def _count_exact(text: str) -> int:
        if _spec():
            return len(_hf_tokenizer()(text, add_special_tokens=False)['input_ids'])
        return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    n = len(text)
    if n > FAST_ESTIMATE_CHARS:
        return max(1, n // 3)
    return _count_exact(text)


def encode(text: str):
    """与 count_tokens 同一分词器的 encode（供最终安全截断用）。"""
    if _spec():
        return _hf_tokenizer()(text, add_special_tokens=False)['input_ids']
    return _ENC.encode(text, disallowed_special=())


def decode(ids) -> str:
    if _spec():
        return _hf_tokenizer().decode(ids, skip_special_tokens=True)
    return _ENC.decode(ids)


def tokenizer_name() -> str:
    return _spec() or 'tiktoken:cl100k_base'

FORMATS = ('text', 'dot', 'graphcode')
MAX_ITER = 8
SHRINK = 0.92  # 超预算时的额外收缩系数
GROW = 0.97    # 不足预算时的放大保守系数
FILL = 0.85    # 达到预算的 85% 即视为填满


@dataclass
class FitResult:
    text: str            # 放进提示词的序列化文本
    tokens: int          # 其 token 数
    kept_fraction: float  # 保留节点数 / 原节点数（放得下 = 1.0；空图 = 1.0）
    kept_nodes: int
    total_nodes: int
    pruned: bool         # 是否发生了裁剪


def serialize_dot(dot_text: str, fmt: str) -> str:
    if fmt == 'dot':
        return dot_text
    p = CPGParser()
    p.parse_dot(dot_text)
    if fmt == 'text':
        return p.explain_in_natural_language()
    if fmt == 'graphcode':
        return p.to_add_edge_script()
    raise ValueError(f'unknown fmt {fmt!r}')


def fit_graph(dot_text: str, fmt: str, budget_tokens: int,
              full_text: Optional[str] = None) -> FitResult:
    """把一张图（DOT）装进 budget_tokens：放得下原样返回，否则度数贪心裁剪后序列化。

    full_text：若调用方已有该图完整序列化文本（如 GraphText-v2 文件内容），传入可省一次序列化。
    """
    total_nodes = len(parse_dot(dot_text).nodes)
    if full_text is None:
        full_text = serialize_dot(dot_text, fmt)
    full_tokens = count_tokens(full_text)
    if full_tokens <= budget_tokens or total_nodes == 0:
        return FitResult(full_text, full_tokens, 1.0, total_nodes, total_nodes, False)

    dot_tokens = count_tokens(dot_text)
    # 目标格式 token 与 DOT token 的比例，先按比例给 DOT 预算，再迭代校正：
    # 超预算 → 收缩；不足预算的 FILL 比例 → 放大再试；返回"放得下的最大"结果。
    ratio = dot_tokens / max(1, full_tokens)
    dot_budget = max(1, int(budget_tokens * ratio))
    best_fit: Optional[FitResult] = None
    last: Optional[FitResult] = None
    tried = set()
    for _ in range(MAX_ITER):
        if dot_budget in tried:
            break
        tried.add(dot_budget)
        pruned_dot = budget_prune(dot_text, dot_budget)
        text = serialize_dot(pruned_dot, fmt)
        toks = count_tokens(text)
        kept = len(parse_dot(pruned_dot).nodes)
        res = FitResult(text, toks, kept / total_nodes, kept, total_nodes, True)
        last = res
        if toks <= budget_tokens:
            if best_fit is None or res.kept_nodes > best_fit.kept_nodes:
                best_fit = res
            if toks >= FILL * budget_tokens or kept >= total_nodes:
                break
            dot_budget = max(dot_budget + 1, int(dot_budget * (budget_tokens / max(1, toks)) * GROW))
        else:
            if kept <= MIN_KEEP:
                break  # 已到保底节点数，无法再小；交给最终的安全截断
            dot_budget = max(1, int(dot_budget * (budget_tokens / toks) * SHRINK))
    return best_fit if best_fit is not None else last  # type: ignore[return-value]


def fit_graphs(items: Sequence[Tuple[str, str, Optional[str]]], total_budget: int) -> List[FitResult]:
    """多图：预算按各图完整序列化 token 数等比例分配（各图保留比例一致），逐图 fit。

    items: [(dot_text, fmt, full_text_or_None), ...]
    """
    fulls = []
    for dot_text, fmt, full_text in items:
        ft = full_text if full_text is not None else serialize_dot(dot_text, fmt)
        fulls.append(ft)
    sizes = [count_tokens(t) for t in fulls]
    total = sum(sizes)
    if total <= total_budget:
        return [fit_graph(d, f, total_budget, ft) for (d, f, _), ft in zip(items, fulls)]
    results = []
    for (dot_text, fmt, _), ft, sz in zip(items, fulls, sizes):
        share = int(total_budget * sz / max(1, total))
        results.append(fit_graph(dot_text, fmt, share, ft))
    return results
