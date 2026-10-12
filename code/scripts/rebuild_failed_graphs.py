"""重建 Joern 解析失败样本的 CPG（回应 F5：Big-Vul 42.3% 空 DFG）。

经验证的失败机制（见 docs/ROOT_CAUSE_REPORT.md）：
  - 约 87% 失败样本是 C++ 代码存成 .c → c2cpg 按 C 解析函数体失败 → 空壳 CPG
  - 其余主要是签名中的属性宏（TSRMLS_DC、__init、__user 等）
经验证的修复配方（24 样本抽验恢复 23/24）：
  1. 先按 .c 解析（保持与现有正常样本一致）
  2. 若产出空壳（无 REACHING_DEF 且 CFG 边 <= 3），改按 .cpp + 预定义宏重解析
  3. 仍空壳则记录为不可恢复（约 4%，多为前导空格+无返回类型的残缺片段）

注意：直接换 .cpp 全量重跑是错的——约 1/3 正常 C 样本含 C-only 语法
（如 void* 隐式转换），按 C++ 解析反而退化为空壳。必须用逐样本回退策略。

用法：
  python rebuild_failed_graphs.py --dataset ../dataset/bigvul --joern ~/bin/joern/joern-cli \
      [--limit 100] [--dry-run]

产出：每个恢复样本写 <id>/<id>-GraphSON-out-rebuilt/<part>/all/export.json，
并打印恢复统计。后续需用序列化器从 GraphSON 重新生成 GraphText/GraphDot/GraphCode。
"""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# 抽样验证过的属性宏预定义表（PHP TSRMLS / Linux kernel / Windows 调用约定）
MACRO_DEFINES = [
    'TSRMLS_DC', 'TSRMLS_CC',
    '__init', '__exit', '__user', '__force', '__percpu', '__rcu', '__iomem',
    '__read_mostly', '__must_check', '__maybe_unused', '__cold', '__weak',
    '__sched', '__ref', '__net_init', '__net_exit', '__devinit', '__devexit',
    'STDCALL', 'WINAPI', 'APIENTRY',
    # 2026-08-15 v2 重建 unrecoverable/partial 清单巡检新增（PLAN 3.3 回灌配方）：
    'UNUSED', 'CLASS', 'ZEXPORT', 'STATIC=static', 'pTHX_', 'pTHX', 'aTHX_', 'aTHX',
    'PJ_DEF(t)=t', 'PJ_DECL(t)=t', 'PJ_IDEF(t)=t', 'PJ_INLINE(t)=t',
    'IRC_PROTOCOL_CALLBACK(x)=int irc_protocol_cb_##x(void *ptr, const char *date, const char *nick, '
    'const char *address, const char *host, const char *command, int ignored, int argc, char **argv, char **argv_eol)',
    'INTERNAL_FUNCTION_PARAMETERS=int ht, void *return_value, void *return_value_ptr, void *this_ptr, int return_value_used',
    'INTERNAL_FUNCTION_PARAM_PASSTHRU=ht, return_value, return_value_ptr, this_ptr, return_value_used',
    'PHP_FUNCTION(n)=void zif_##n(int ht, void *return_value)',
    'ZEND_FUNCTION(n)=void zif_##n(int ht, void *return_value)',
    'PHP_METHOD(c,n)=void zim_##c##_##n(int ht, void *return_value)',
    '__rte_unused', '__rte_always_inline=inline', '__rte_noinline', 'R_OWN', 'R_NONNULL', 'R_NULLABLE', 'R_API',
    # 注意：不要加空值宏 'D='（实测会让 c2cpg 对整个文件产出 0 边的空 CPG）
]

NODE_RE = re.compile(r'^- Node\d+:', re.M)


def is_stub_export(export_json):
    """判断导出的 CPG 是否为解析失败的空壳（特征签名：AST≈11, CFG≈3, 无 REACHING_DEF）"""
    try:
        with open(export_json, encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        return True
    dd = d.get('@value', d)
    c = collections.Counter()
    for e in dd.get('edges', []):
        ev = e.get('@value', e)
        lab = ev.get('label')
        if isinstance(lab, dict):
            lab = lab.get('@value')
        c[lab] += 1
    return c.get('REACHING_DEF', 0) == 0 and c.get('CFG', 0) <= 3


def sample_is_failed(sample_dir, sid, part):
    """用现有 GraphText DFG 判断该样本是否为解析失败"""
    p = os.path.join(sample_dir, f'{sid}-GraphText-out', part, 'dfg', 'dfg.txt')
    if not os.path.exists(p):
        return False
    with open(p, encoding='utf-8', errors='replace') as f:
        return not NODE_RE.search(f.read())


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def rebuild_one(joern_cli, src_c, out_dir):
    """按回退策略重建一个函数的 CPG，返回 ('c'|'cpp'|None, export_path)"""
    joern_parse = os.path.join(joern_cli, 'joern-parse')
    joern_export = os.path.join(joern_cli, 'joern-export')
    with tempfile.TemporaryDirectory() as tmp:
        attempts = [
            ('c', os.path.join(tmp, 'func.c'), []),
            ('cpp', os.path.join(tmp, 'func.cpp'),
             [a for m in MACRO_DEFINES for a in ('--define', m)]),
        ]
        for tag, src, extra in attempts:
            shutil.copyfile(src_c, src)
            cpg = os.path.join(tmp, f'cpg_{tag}.bin')
            cmd = [joern_parse, src, '-o', cpg]
            if extra:
                cmd += ['--frontend-args'] + extra
            r = run(cmd)
            if r.returncode != 0 or not os.path.exists(cpg):
                continue
            exp = os.path.join(tmp, f'export_{tag}')
            r = run([joern_export, cpg, '--repr', 'all', '--format', 'graphson', '-o', exp])
            exp_json = os.path.join(exp, 'export.json')
            if r.returncode != 0 or not os.path.exists(exp_json):
                continue
            if not is_stub_export(exp_json):
                os.makedirs(out_dir, exist_ok=True)
                shutil.copyfile(exp_json, os.path.join(out_dir, 'export.json'))
                return tag, os.path.join(out_dir, 'export.json')
    return None, None


def main():
    ap = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--dataset', default=os.path.join(script_dir, '..', 'dataset', 'bigvul'))
    ap.add_argument('--joern', default=os.path.expanduser('~/bin/joern/joern-cli'))
    ap.add_argument('--limit', type=int, default=0, help='最多处理多少个失败样本；0 为全部')
    ap.add_argument('--dry-run', action='store_true', help='只扫描统计，不重建')
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    ids = sorted(d for d in os.listdir(root) if d.isdigit())
    failed = []
    for sid in ids:
        sd = os.path.join(root, sid)
        parts = [p for p in ('before', 'after') if sample_is_failed(sd, sid, p)]
        if parts:
            failed.append((sid, parts))
    print(f'扫描 {len(ids)} 个样本，发现解析失败 {len(failed)} 个')
    if args.dry_run:
        return

    if args.limit:
        failed = failed[:args.limit]
    stats = collections.Counter()
    for n, (sid, parts) in enumerate(failed, 1):
        sd = os.path.join(root, sid)
        for part in parts:
            src = os.path.join(sd, f'{sid}-{part}.c')
            if not os.path.exists(src):
                stats['no_source'] += 1
                continue
            out_dir = os.path.join(sd, f'{sid}-GraphSON-out-rebuilt', part, 'all')
            tag, _ = rebuild_one(args.joern, src, out_dir)
            stats[f'recovered_{tag}' if tag else 'unrecoverable'] += 1
        if n % 50 == 0:
            print(f'  进度 {n}/{len(failed)}  {dict(stats)}', file=sys.stderr)
    print(f'完成: {dict(stats)}')


if __name__ == '__main__':
    main()
