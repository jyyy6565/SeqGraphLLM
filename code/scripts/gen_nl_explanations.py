#!/usr/bin/env python3
"""E4 消融：用被测模型为每个样本预生成"NL 解释"（PLAN.md Phase 5.2）。

对照目的：区分"图结构信息"与"多给了一段关于代码的自然语言"两种增益来源。
每个 (id, part, graph) 生成一段解释文本：功能概述 + 用文字描述控制流(cfg) / 数据依赖(dfg) /
语法结构(ast)，**不提图/节点/边**；长度对齐对应 GraphText-v2 **在目标模型窗口下实际放入的 token 数**（budget-fit 协议：
min(完整图 token, 该样本的图预算)，--window 指定窗口，默认 16384）（±20%，不满足则最多重试 2 次并调整目标长度提示）。

输出：<id>/<id>-NL-out/<part>/<graph>/<graph>.nl.txt（与 getText 的 `{graph}.{suffix}` 命名一致；配置名 cfg_nl 等）。
ManyTypes4Py：--dataset manytypes4py → <dataset>/NL/<repo>.json（{path: {cfg,ddg,ast: text}}）。

通过 OpenAI 兼容接口（vLLM）调用，参数与 E1 一致（temperature=0.2、top_p=1.0、seed=1）；
--provider anthropic 时用 Anthropic SDK（Claude 系不传采样参数）。

用法（在 GPU 集群上、vLLM 起好后）：
    python gen_nl_explanations.py --dataset bigvul --model /path/to/CodeLlama-7b-Instruct-hf \
        --api-bases http://127.0.0.1:8010/v1 http://127.0.0.1:8011/v1 --workers 16
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'data_processing'))
from regenerate_bigvul import atomic_write, nonempty  # noqa: E402

from graph_budget_fit import count_tokens as _count_tokens  # noqa: E402
GRAPH_ASPECT = {
    'cfg': 'the control flow: the order in which statements execute, branches, loops, early returns and error paths',
    'dfg': 'the data dependencies: which variables are defined where, how values flow between statements, '
           'and which computations depend on which inputs',
    'ddg': 'the data dependencies: which variables are defined where, how values flow between statements, '
           'and which computations depend on which inputs',
    'ast': 'the syntactic structure: how the code is organized into declarations, statements, expressions '
           'and nested blocks',
}
SYSTEM = ('You are an expert code analyst. Write plain-prose explanations of source code. '
          'Never mention graphs, nodes, edges, or any graph terminology. Do not output code, lists of '
          'identifiers, JSON, or markdown headings—only paragraphs of prose.')


def count(text: str) -> int:
    """用**被测模型自己的分词器**数 token（SEQGRAPH_TOKENIZER，在 main 里按 --model 设好）。

    原来用 tiktoken：它对 C 代码低估，于是 `提示词 + max_tokens` 算着没超窗、实际请求超了，
    被 vLLM 以 400 拒掉（实测整批全挂）。而且 NL 的目标长度本来就要和 budget-fit 里
    图占的 token 数对齐，两边必须用同一把尺子。
    """
    return _count_tokens(text)


def build_prompt(code: str, graph: str, target_tokens: int, lang: str) -> str:
    words = max(40, int(target_tokens * 0.75))
    return (f'Explain the following {lang} code in about {words} words. First give a short functional overview, '
            f'then describe {GRAPH_ASPECT[graph]}. Write continuous prose only.\n\nCode:\n{code}\n')


class Client:
    def __init__(self, provider, model, api_bases, temperature=0.2, top_p=1.0, seed=1):
        self.provider, self.model, self.api_bases = provider, model, api_bases
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self._i = 0
        self._lock = threading.Lock()

    def _next_base(self):
        with self._lock:
            b = self.api_bases[self._i % len(self.api_bases)]
            self._i += 1
        return b

    def complete(self, prompt: str, max_tokens: int) -> str:
        if self.provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic()
            r = client.messages.create(model=self.model, max_tokens=max_tokens, system=SYSTEM,
                                       messages=[{'role': 'user', 'content': prompt}],
                                       output_config={'effort': 'low'})
            if r.stop_reason == 'refusal':
                raise RuntimeError('refusal')
            return ''.join(b.text for b in r.content if b.type == 'text').strip()
        from openai import OpenAI
        client = OpenAI(api_key='EMPTY', base_url=self._next_base())
        r = client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p,
            seed=self.seed,
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}])
        return (r.choices[0].message.content or '').strip()


def generate(client: Client, code: str, graph: str, target: int, lang: str, window: int = 16384) -> str:
    """生成并把长度对齐到 target ±20%（最多 3 次尝试，逐次修正目标）。

    max_tokens 必须扣掉提示词本身占的窗口：目标长度是按"这段图在提示词里占了多少 token"定的，
    而生成时提示词里放的是代码+指令，`prompt + max_tokens` 很容易正好越过窗口上限，
    被 vLLM 以 400 拒掉（实测第一批 48 条全挂，都是差 1 个 token）。
    """
    best, best_gap = None, None
    ask = target
    for _ in range(3):
        prompt = build_prompt(code, graph, ask, lang)
        # 除了 user 段，还要给 system 消息和聊天模板本身留位置：服务端数出来的 prompt
        # 比脚本数的多几十个 token，差这一点就会被 400 拒掉。
        room = window - count(prompt) - count(SYSTEM) - 128
        cap = min(int(target * 1.6) + 64, room)
        if cap < 32:
            return best or ''                       # 代码本身几乎占满窗口，生成不了
        text = client.complete(prompt, max_tokens=cap)
        n = count(text)
        gap = abs(n - target) / max(1, target)
        if best is None or gap < best_gap:
            best, best_gap = text, gap
        if gap <= 0.2:
            break
        ask = max(30, int(ask * target / max(1, n)))  # 按实际/目标比例修正下一次的长度提示
    return best


def run_bigvul(args, client):
    root = os.path.join(CODE_DIR, 'dataset', args.dataset)
    ids = sorted(d for d in os.listdir(root) if d.isdigit())
    if args.id_file:
        wanted = set(l.strip() for l in open(args.id_file) if l.strip())
        ids = [i for i in ids if i in wanted]
    elif args.dataset == 'bigvul':
        excl = set(l.strip() for l in open(os.path.join(CODE_DIR, 'ids.txt')) if l.strip())
        ids = [i for i in ids if i not in excl]
    if args.limit:
        ids = ids[:args.limit]
    jobs = []
    graphs = tuple(args.graphs) if args.graphs else ('cfg', 'dfg', 'ast')
    for sid in ids:
        for part in ('before', 'after'):
            for graph in graphs:
                out = os.path.join(root, sid, f'{sid}-NL-out', part, graph, f'{graph}.nl.txt')
                if not args.force and nonempty(out):
                    continue
                jobs.append((sid, part, graph, out))
    print(f'{args.dataset}: {len(ids)} ids, {len(jobs)} NL texts to generate', file=sys.stderr)

    sys.path.insert(0, os.path.join(CODE_DIR, 'vulnerability_detection'))
    from vulnerability_detection import input_budget, QUERY_HEADER  # noqa: E402
    from utils.build_prompt import BuildPrompt  # noqa: E402
    prefix_msgs = BuildPrompt(current_file_path=os.path.join(CODE_DIR, 'vulnerability_detection'),
                              template_type='expertise', shot_number=0, graph_type='cfg',
                              datasets_name=args.dataset).build_prompt()
    prefix_tokens = sum(count(m['content']) for m in prefix_msgs)

    def work(job):
        sid, part, graph, out = job
        code = open(os.path.join(root, sid, f'{sid}-{part}.c'), encoding='utf-8', errors='replace').read()
        gt = os.path.join(root, sid, f'{sid}-GraphText-v2-out', part, graph, f'{graph}.txt')
        full = count(open(gt, encoding='utf-8').read()) if nonempty(gt) else 200
        graph_budget = input_budget(args.window) - prefix_tokens - count(QUERY_HEADER + 'Code:\n' + code + '\n') - 40
        target = max(60, min(full, graph_budget))
        text = generate(client, code, graph, target, 'C', args.window)
        atomic_write(out, text + '\n')
        return job

    _run_jobs(jobs, work, args.workers)


def run_mt4py(args, client):
    root = os.path.join(CODE_DIR, 'dataset', 'manytypes4py')
    src_dir = os.path.join(root, 'processed_projects_cleaned')
    gt_dir = os.path.join(root, 'GraphText-v2')
    out_dir = os.path.join(root, 'NL')
    sys.path.insert(0, os.path.join(CODE_DIR, 'data_processing'))
    from regenerate_manytypes4py import restore_code
    want_graphs = tuple(args.graphs) if args.graphs else ('cfg', 'ddg', 'ast')
    repos = sorted(f for f in os.listdir(gt_dir) if f.endswith('.json'))
    if args.limit:
        repos = repos[:args.limit]

    def work(repo):
        out = os.path.join(out_dir, repo)
        if not args.force and nonempty(out):
            return repo
        gt = json.load(open(os.path.join(gt_dir, repo), encoding='utf-8'))
        src = json.load(open(os.path.join(src_dir, repo), encoding='utf-8'))
        result = {}
        for path, graphs in gt.items():
            code = restore_code(src.get(path, {}).get('code', ''))
            result[path] = {}
            for graph in want_graphs:
                if graph not in graphs:
                    continue
                result[path][graph] = generate(client, code, graph, count(graphs[graph]), 'Python', args.window)
        atomic_write(out, json.dumps(result, ensure_ascii=False, indent=2))
        return repo

    _run_jobs(repos, work, args.workers)


def _run_jobs(jobs, fn, workers):
    done = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f'FAIL {futs[fut]}: {type(e).__name__}: {e}', file=sys.stderr)
            if (done + failed) % 200 == 0:
                print(f'  {done + failed}/{len(jobs)} done={done} failed={failed}', file=sys.stderr)
    print(f'finished: done={done} failed={failed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['bigvul', 'primevul', 'manytypes4py'], default='bigvul')
    ap.add_argument('--model', required=True, help='vLLM 模型路径 / 名称，或 Claude 模型 id')
    ap.add_argument('--api-bases', nargs='*', default=['http://127.0.0.1:8010/v1'])
    ap.add_argument('--provider', choices=['openai', 'anthropic'], default='openai')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--window', type=int, default=16384,
                    help='目标模型窗口：NL 长度对齐 min(完整图 token, 该样本在此窗口下的图预算)')
    ap.add_argument('--id-file')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    # 决策 3（8/17 用户确认）：只生成 E4 实际用到的图的 NL（VD=cfg、TI=ddg），
    # 否则 3 张图 × 全子集的长文本生成要多花几倍 GPU 时间。
    ap.add_argument('--graphs', nargs='+', default=None,
                    help='只为这些图生成 NL（bigvul/primevul 默认 cfg dfg ast；mt4py 默认 cfg ddg ast）')
    args = ap.parse_args()

    # token 计数用被测模型自己的分词器（与 VD/TI 客户端、budget-fit 同一把尺子）
    if not os.environ.get('SEQGRAPH_TOKENIZER') and os.path.isdir(args.model):
        os.environ['SEQGRAPH_TOKENIZER'] = args.model
    # **在主线程里先把分词器加载好**：count() 里是懒加载 `from transformers import AutoTokenizer`，
    # 几十个工作线程同时第一次触发时会撞上 transformers 的 _LazyModule 竞态，
    # 报 `cannot import name 'AutoTokenizer'` 把整批任务打挂（8/19 在 MT4Py 那批上踩过）。
    count('warmup')
    client = Client(args.provider, args.model, args.api_bases, args.temperature, args.top_p, args.seed)
    if args.dataset == 'manytypes4py':
        run_mt4py(args, client)
    else:
        run_bigvul(args, client)


if __name__ == '__main__':
    main()
