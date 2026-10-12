import os
import tempfile
import subprocess
import sys
import re
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import json
import argparse
from collections import defaultdict
import xml.etree.ElementTree as ET
import io
try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

from cpg_parser import create_parser, extract_subgraphs

# ============================================================================
# CDEP 图压缩算法 (基于文本的简化实现)
# ============================================================================

def cdep_compress_dot(dot_content: str, min_degree: int = 3) -> str:
    """
    使用简化的图压缩算法压缩 DOT 格式图。
    基于度数的节点合并策略。
    
    参数:
        dot_content: DOT 格式的图内容
        min_degree: 保留节点的最小度数阈值，度数小于此值的节点可能被合并
    
    返回:
        压缩后的 DOT 格式图
    """
    if not dot_content or not dot_content.strip():
        return ""
    
    try:
        # 解析 DOT 格式
        out_edges, in_edges, node_attrs, edge_lines = parse_dot_graph(dot_content)
        
        # 先计算所有节点的度数
        node_degrees = {}
        all_nodes = set(out_edges.keys()) | set(in_edges.keys())
        for node in all_nodes:
            total_degree = len(out_edges.get(node, set())) + len(in_edges.get(node, set()))
            node_degrees[node] = total_degree
        
        # 如果图太小或太稀疏，不进行压缩
        if len(all_nodes) < 5 or sum(node_degrees.values()) < 10:
            return dot_content
        
        # 记录哪些节点被合并了
        merged_nodes = set()
        merge_count = {}  # 记录每个节点合并了多少个其他节点
        
        # 迭代进行节点合并（只合并度数较低的节点）
        max_iterations = 100
        for iteration in range(max_iterations):
            changed = False
            
            # 只合并度为1的节点（叶子节点）- 更保守的策略
            degree_1_nodes = []
            for node in all_nodes:
                if node in merged_nodes:
                    continue
                total_degree = len(out_edges.get(node, set())) + len(in_edges.get(node, set()))
                if total_degree == 1:
                    degree_1_nodes.append(node)
            
            # 合并度为1的节点到其唯一的邻居（但要检查邻居度数）
            for node in degree_1_nodes:
                if node in merged_nodes:
                    continue
                    
                # 找到唯一的邻居
                out_set = set(out_edges.get(node, set()))
                in_set = set(in_edges.get(node, set()))
                neighbors = list(out_set.union(in_set))
                if len(neighbors) != 1:
                    continue
                    
                neighbor = neighbors[0]
                if neighbor in merged_nodes:
                    continue
                
                # 只合并到度数足够大的邻居节点
                neighbor_degree = len(out_edges.get(neighbor, set())) + len(in_edges.get(neighbor, set()))
                if neighbor_degree < min_degree:
                    continue
                
                # 标记该节点为已合并
                merged_nodes.add(node)
                merge_count[neighbor] = merge_count.get(neighbor, 1) + merge_count.get(node, 1)
                
                # 更新边关系
                if node in out_edges:
                    del out_edges[node]
                if node in in_edges:
                    del in_edges[node]
                    
                # 移除指向被合并节点的边
                for n in out_edges:
                    out_edges[n].discard(node)
                for n in in_edges:
                    in_edges[n].discard(node)
                
                changed = True
            
            if not changed:
                break
        
        # 如果没有任何节点被合并，返回原始内容
        if not merged_nodes:
            return dot_content
        
        # 构建压缩后的 DOT
        lines = []
        
        # 提取图的第一行（digraph xxx {）
        first_line = dot_content.split('\n')[0].strip()
        if first_line.startswith('digraph'):
            lines.append(first_line)
        else:
            lines.append("digraph {")
        
        # 添加保留的节点
        # 匹配各种格式的节点定义："id" [...] 或 "id"[...]
        node_pattern = re.compile(r'"([^"]+)"\s*\[([^\]]+)\]')
        added_nodes = set()  # 防止重复添加
        
        for line in dot_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('digraph') or line == '}':
                continue
                
            match = node_pattern.search(line)
            if match:
                node_id = f'"{match.group(1)}"'
                
                # 跳过已合并的节点
                if node_id in merged_nodes:
                    continue
                
                # 跳过已添加的节点（防止重复）
                if node_id in added_nodes:
                    continue
                    
                added_nodes.add(node_id)
                attrs = match.group(2).strip()
                
                # 如果节点合并了其他节点，添加合并计数
                if node_id in merge_count and merge_count[node_id] > 1:
                    # 查找 label 属性并修改
                    label_match = re.search(r'label\s*=\s*"([^"]*)"', attrs)
                    if label_match:
                        old_label = label_match.group(1)
                        new_label = f"{old_label} [{merge_count[node_id]} nodes merged]"
                        attrs = attrs.replace(f'label="{old_label}"', f'label="{new_label}"')
                    # 没有 label 的情况，添加一个
                    elif 'label' not in attrs:
                        attrs = f'label="Node [{merge_count[node_id]} nodes merged]" {attrs}'
                
                lines.append(f'  {node_id} [{attrs}];')
        
        # 添加保留的边
        edge_pattern = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"')
        
        for line in dot_content.split('\n'):
            line = line.strip()
            match = edge_pattern.search(line)
            if match:
                src = f'"{match.group(1)}"'
                dst = f'"{match.group(2)}"'
                # 只保留两端节点都未被合并的边
                if src not in merged_nodes and dst not in merged_nodes:
                    lines.append(f'  {line}')
        
        lines.append("}")
        return '\n'.join(lines)
        
    except Exception as e:
        return f"CDEP 压缩失败: {e}"

def _guess_graph_type_from_key(key: str) -> Optional[str]:
    """根据键名推测图类型，用于决定 NL 解析器类型。"""
    if not isinstance(key, str):
        return None
    k = key.lower()
    if k.startswith('cfg'):
        return 'cfg'
    if k.startswith('dfg') or k.startswith('ddg'):
        return 'dfg'
    if k.startswith('ast'):
        return 'ast'
    return None


def _ensure_results_dict(data: Dict) -> Dict:
    """确保 data['results'] 返回一个 dict。

    有些 JSON 使用列表或其他结构；优先返回 dict：
    - 如果 results 本身是 dict，直接返回
    - 如果 results 是 list，返回第一个 dict 元素（如果存在）
    - 否则返回空 dict
    """
    if not isinstance(data, dict):
        return {}
    res = data.get('results')
    if isinstance(res, dict):
        return res
    if isinstance(res, list):
        for item in res:
            if isinstance(item, dict):
                return item
    return {}

# --- 从 graph_degree_simplifier.py 引入的简化逻辑 ---

def parse_graph_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """解析自然语言图的一行，返回 (src_node, edge_type, dst_node)。

    兼容两种格式：
    1) 旧格式：
       [ID] Desc → EdgeType → [ID] Desc
    2) 新格式（Edges 段落中的句子）：
       - Control flows|Data flows|Syntax links from NodeX (Type) to NodeY (Type).
       - Relates between NodeX (Type) and NodeY (Type).
    """
    s = line.strip()
    if not s or s.startswith('Nodes:') or s.startswith('Edges:') or s.startswith('- Node'):
        return None, None, None

    # 1) 旧格式
    old_pat = r'\[(\d+)\] (.+?) \u2192 (.+?) \u2192 \[(\d+)\] (.+)'
    # 有些环境中箭头可能是直接的 Unicode 字符或被替换为 '→'
    old_pat_alt = r'\[(\d+)\] (.+?) → (.+?) → \[(\d+)\] (.+)'
    m = re.match(old_pat, s)
    if not m:
        m = re.match(old_pat_alt, s)
    if m:
        src_id, src_desc, edge_type, dst_id, dst_desc = m.groups()
        src = f"[{src_id}] {src_desc}"
        dst = f"[{dst_id}] {dst_desc}"
        return src, edge_type, dst

    # 2) 新格式：from/to
    new_from_pat = re.compile(
        r"^-\s*(Control flows|Data flows|Syntax links)\s+from\s+Node(\d+)\s*\([^)]*\)\s+to\s+Node(\d+)\s*\([^)]*\)\.?$",
        re.IGNORECASE,
    )
    m2 = new_from_pat.match(s)
    if m2:
        edge_phrase, src_num, dst_num = m2.groups()
        return f"Node{src_num}", edge_phrase, f"Node{dst_num}"

    # 3) 新格式：between/and（Relates）
    new_between_pat = re.compile(
        r"^-\s*Relates\s+between\s+Node(\d+)\s*\([^)]*\)\s+and\s+Node(\d+)\s*\([^)]*\)\.?$",
        re.IGNORECASE,
    )
    m3 = new_between_pat.match(s)
    if m3:
        src_num, dst_num = m3.groups()
        return f"Node{src_num}", 'Relates', f"Node{dst_num}"

    return None, None, None

def build_graph(graph_text: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """构建图的出度和入度关系"""
    out_edges = defaultdict(set)
    in_edges = defaultdict(set)
    
    for line in graph_text.split('\n'):
        if line.strip():
            src, _, dst = parse_graph_line(line)
            if src and dst:
                out_edges[src].add(dst)
                in_edges[dst].add(src)
    
    return out_edges, in_edges

def simplify_graph_by_degree(graph_text: str, min_out_degree: int, min_in_degree: int) -> str:
    """根据出度和入度限制简化自然语言格式的图（在原图基础上删除不保留项，保持行顺序，不重建）。"""
    if not isinstance(graph_text, str) or not graph_text.strip():
        return ""

    # 先统计度数以决定保留的节点集合
    out_edges, in_edges = build_graph(graph_text)

    keep_nodes: Set[str] = set()
    for node, outs in out_edges.items():
        if len(outs) >= min_out_degree:
            keep_nodes.add(node)
    for node, ins in in_edges.items():
        if len(ins) >= min_in_degree:
            keep_nodes.add(node)

    # 在原文逐行遍历：
    # - 对能解析成边的行：若 src/dst 有任一在 keep_nodes 中则保留，否则删除
    # - 对无法解析成边的行（如空行、说明行、Nodes 段的节点说明行）：原样保留，避免重建
    kept_lines: List[str] = []
    for raw in graph_text.split('\n'):
        line = raw.rstrip('\n')
        if not line.strip():
            kept_lines.append(raw)
            continue
        src, _, dst = parse_graph_line(line)
        if src is None and dst is None:
            # 非标准边描述，原样保留
            kept_lines.append(raw)
            continue
        # 边行：只在两端至少一个节点被保留时保留
        if (src in keep_nodes) or (dst in keep_nodes):
            kept_lines.append(raw)

    return '\n'.join(kept_lines)

def parse_dot_graph(dot_content: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str], List[str]]:
    """解析DOT格式图，返回出度关系、入度关系、节点描述和边关系"""
    out_edges = defaultdict(set)
    in_edges = defaultdict(set)
    node_attrs = {}
    edge_lines = []
    
    node_pattern = r'"(\d+)" \[label = <(.+?)> \]'
    for match in re.finditer(node_pattern, dot_content):
        node_id, label = match.groups()
        node_id_str = f'"{node_id}"'
        node_attrs[node_id_str] = label
    
    edge_pattern = r'"(\d+)" -> "(\d+)"'
    for match in re.finditer(edge_pattern, dot_content):
        src_id, dst_id = match.groups()
        src_id_str = f'"{src_id}"'
        dst_id_str = f'"{dst_id}"'
        
        out_edges[src_id_str].add(dst_id_str)
        in_edges[dst_id_str].add(src_id_str)
        edge_lines.append(f'  {src_id_str} -> {dst_id_str}')
    
    return out_edges, in_edges, node_attrs, edge_lines

def simplify_dot_graph(dot_content: str, min_out_degree: int, min_in_degree: int) -> str:
    """根据出度/入度简化 DOT：在原 DOT 文本上做行级删除，不重新构建图结构或改变顺序。"""
    if not isinstance(dot_content, str) or not dot_content.strip():
        return ""

    # 先用解析函数统计度数与节点属性，决定保留节点集合
    out_edges, in_edges, node_attrs, edge_lines = parse_dot_graph(dot_content)

    keep_nodes: Set[str] = set()
    for node, outs in out_edges.items():
        if len(outs) >= min_out_degree:
            keep_nodes.add(node)
    for node, ins in in_edges.items():
        if len(ins) >= min_in_degree:
            keep_nodes.add(node)

    # 若没有任何满足条件的节点，直接返回空串（与原逻辑兼容）
    if not keep_nodes:
        return ""

    # 逐行遍历原 DOT：
    # - 边行：如果其任一端点在 keep_nodes 中，则保留；否则删除
    # - 节点定义行：若该节点在 keep_nodes 中，保留；否则删除
    # - 其他配置/声明/花括号/注释行：原样保留
    kept_lines: List[str] = []

    # 用于判断是否是节点/边行的正则
    node_def_re = re.compile(r'^\s*"(\d+)"\s*\[.*\]')
    edge_re = re.compile(r'^\s*"(\d+)"\s*->\s*"(\d+)"')

    for raw in dot_content.split('\n'):
        line = raw.rstrip('\n')
        if not line.strip():
            kept_lines.append(raw)
            continue

        m_edge = edge_re.search(line)
        if m_edge:
            src_id = f'"{m_edge.group(1)}"'
            dst_id = f'"{m_edge.group(2)}"'
            if (src_id in keep_nodes) or (dst_id in keep_nodes):
                kept_lines.append(raw)
            # 否则删掉该边
            continue

        m_node = node_def_re.search(line)
        if m_node:
            n_id = f'"{m_node.group(1)}"'
            if n_id in keep_nodes:
                kept_lines.append(raw)
            # 否则删掉该节点定义
            continue

        # 其它行原样保留（如 digraph 头、属性设置、注释、结束花括号等）
        kept_lines.append(raw)

    return '\n'.join(kept_lines)

# --- GraphML 简化逻辑 ---
def simplify_graphml(graphml_content: str, min_out_degree: int, min_in_degree: int) -> str:
    """根据出度/入度简化 GraphML：删除未达阈值的节点及其相关边，保留 keys/data/图结构。"""
    if not isinstance(graphml_content, str) or not graphml_content.strip():
        return ""
    try:
        root = ET.fromstring(graphml_content)
    except Exception:
        return ""

    # 获取 graph 节点（通常只有一个）
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    graph = root.find(f'.//{ns}graph') or root.find('.//graph')
    if graph is None:
        return ""

    node_tag = f'{ns}node' if ns else 'node'
    edge_tag = f'{ns}edge' if ns else 'edge'

    nodes = [n for n in list(graph) if n.tag == node_tag]
    edges = [e for e in list(graph) if e.tag == edge_tag]
    node_ids = {n.get('id') for n in nodes}
    out_deg = {nid: 0 for nid in node_ids}
    in_deg = {nid: 0 for nid in node_ids}
    for e in edges:
        s = e.get('source')
        t = e.get('target')
        if s in out_deg:
            out_deg[s] += 1
        if t in in_deg:
            in_deg[t] += 1

    keep_nodes = {nid for nid in node_ids if (out_deg.get(nid, 0) >= min_out_degree) or (in_deg.get(nid, 0) >= min_in_degree)}
    if not keep_nodes:
        return ""

    # 深拷贝并删除不需要的节点/边
    try:
        root2 = ET.fromstring(graphml_content)
    except Exception:
        return ""
    graph2 = root2.find(f'.//{ns}graph') or root2.find('.//graph')
    if graph2 is None:
        return ""

    for child in list(graph2):
        if child.tag == node_tag:
            if child.get('id') not in keep_nodes:
                graph2.remove(child)
        elif child.tag == edge_tag:
            s = child.get('source')
            t = child.get('target')
            if (s not in keep_nodes) or (t not in keep_nodes):
                graph2.remove(child)

    try:
        return ET.tostring(root2, encoding='unicode')
    except Exception:
        return ""

# --- 原有 code_to_graphs.py 逻辑 ---


def code_to_graphs(
    code_content: str,
    graph_types: Union[str, List[str]] = ["cfg", "ddg", "ast"],
    language: str = "c",
    # 是否保留原始图
    include_original: bool = True,
    output_format: str = "graphml",  # 新增：'dot' | 'graphml'
    generate_nl: bool = True,  # 新增：是否在 DOT 模式下生成 NL
) -> Dict[str, Dict[str, str]]:
    """
    直接从代码字符串生成指定类型的图结构，同时返回DOT格式和自然语言描述
    
    参数:
        code_content: 要分析的源代码字符串
        graph_types: 图表示类型，可以是字符串或列表，支持 'cfg', 'ddg', 'ast'
        language: 代码语言，默认为 'c'
    
    返回:
        嵌套字典，结构为 {
            "dot": {"cfg": "...", "ddg": "...", "ast": "..."},
            "nl": {"cfg": "...", "ddg": "...", "ast": "..."}
        }
    """
    # 将单个图类型转换为列表
    if isinstance(graph_types, str):
        graph_types = [graph_types.lower()]
    else:
        graph_types = [gt.lower() for gt in graph_types]
    
    # 验证图类型 - 现在使用 dfg 而不是 ddg
    valid_types = {"cfg", "dfg", "ast"}
    mapped_types = []
    for gt in graph_types:
        if gt == "ddg":  # 向后兼容：ddg映射到dfg
            mapped_types.append("dfg")
        elif gt in valid_types:
            mapped_types.append(gt)
        else:
            raise ValueError(f"不支持的图类型: {gt}。支持的类型有: {', '.join(valid_types)}")
    
    graph_types = mapped_types
    
    # Joern CLI 工具路径（Windows 下使用 .bat 脚本）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    joern_cli_dir = os.path.join(script_dir, "joern-cli")
    if not os.path.exists(joern_cli_dir):
        joern_cli_dir = "G:\\study\\项目\\CodeGraph\\joern-cli"
    joern_parse_path = os.path.join(joern_cli_dir, "joern-parse")
    joern_export_path = os.path.join(joern_cli_dir, "joern-export")
    if os.name == 'nt':  # Windows
        joern_parse_path += ".bat"
        joern_export_path += ".bat"

    
    # 创建临时文件和目录存储代码
    with tempfile.NamedTemporaryFile(suffix=f".{language}", delete=False) as temp_code_file:
        # 写入代码到临时文件
        temp_code_file.write(code_content.encode('utf-8'))
        temp_code_file_path = temp_code_file.name
    
    # 创建临时目录用于存储图输出
    temp_output_dir = tempfile.mkdtemp()
    temp_cpg_path = tempfile.mktemp(suffix=".bin")
    
    fmt = (output_format or 'graphml').lower()
    container_key = 'dot' if fmt == 'dot' else 'graphml'
    results = {
        container_key: {},
        "nl": {}
    }
    
    try:
        # 步骤1：使用 joern-parse 解析代码，超时120秒
        code_unix = temp_code_file_path.replace('\\', '/')
        cpg_unix = temp_cpg_path.replace('\\', '/')
        parse_result = subprocess.run(
            [joern_parse_path, code_unix, "-o", cpg_unix],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120
        )
        
        # 步骤2：导出完整CPG
        cpg_output_dir = os.path.join(temp_output_dir, "cpg")
        out_unix = cpg_output_dir.replace('\\', '/')
        
        if fmt not in {"dot", "graphml"}:
            fmt = "graphml"
        export_result = subprocess.run(
            [joern_export_path, cpg_unix, "-o", out_unix, "--repr", "all", "--format", fmt],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120
        )
        
        # 读取生成的CPG文件
        cpg_files = os.listdir(cpg_output_dir)
        
        if not cpg_files:
            for graph_type in graph_types:
                results[container_key][graph_type] = f"无法生成CPG图"
                results["nl"][graph_type] = f"无法生成CPG图"
            return results
        
        # 取第一个生成的CPG文件
        cpg_file_path = os.path.join(cpg_output_dir, cpg_files[0])
        
        # 读取CPG内容
        with open(cpg_file_path, 'r', encoding='utf-8') as f:
            cpg_content = f.read()
        
        # 从CPG中提取不同类型的子图（统一走 cpg_parser 的入口）
        extracted_graphs = extract_subgraphs(cpg_content, graph_types, fmt)
        
        # 对每种图类型生成DOT和自然语言描述
        for graph_type in graph_types:
            if graph_type in extracted_graphs:
                sub_content = extracted_graphs[graph_type]
                if include_original:
                    results[container_key][graph_type] = sub_content
                # 仅当输出为 DOT 且开启 generate_nl 时，生成 NL
                if fmt == 'dot' and generate_nl:
                    temp_subgraph_file = os.path.join(temp_output_dir, f"{graph_type}_temp.dot")
                    with open(temp_subgraph_file, 'w', encoding='utf-8') as f:
                        f.write(sub_content)
                    try:
                        parser = create_parser(graph_type, temp_subgraph_file)
                        parser.parse_dot_file()
                        results["nl"][graph_type] = parser.explain_in_natural_language()
                    except Exception as e:
                        results["nl"][graph_type] = f"生成{graph_type.upper()}自然语言描述时发生错误: {str(e)}"
                else:
                    results["nl"][graph_type] = ""
            else:
                results[container_key][graph_type] = f"无法从CPG提取{graph_type.upper()}图"
                results["nl"][graph_type] = f"无法从CPG提取{graph_type.upper()}图"
            
    except subprocess.CalledProcessError as e:
        print(f"Joern处理错误: {e.stderr}")
        for graph_type in graph_types:
            if graph_type not in results[container_key]:
                results[container_key][graph_type] = f"处理{graph_type.upper()}时发生错误: {e.stderr}"
                results["nl"][graph_type] = f"处理{graph_type.upper()}时发生错误: {e.stderr}"
    except Exception as e:
        print(f"处理图时发生错误: {str(e)}")
        for graph_type in graph_types:
            if graph_type not in results[container_key]:
                results[container_key][graph_type] = f"处理{graph_type.upper()}时发生错误: {str(e)}"
                results["nl"][graph_type] = f"处理{graph_type.upper()}时发生错误: {str(e)}"
    finally:
        # 清理临时文件
        if os.path.exists(temp_code_file_path):
            os.remove(temp_code_file_path)
        if os.path.exists(temp_cpg_path):
            os.remove(temp_cpg_path)
        # 递归删除临时目录
        for root, dirs, files in os.walk(temp_output_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        if os.path.exists(temp_output_dir):
            os.rmdir(temp_output_dir)
    
    return results


def compress_dot_text(dot_text: str, min_degree: int = 5) -> str:
    """
    使用 CDEP 算法压缩图（字符串 -> 字符串），可单独调用，无需生成 CPG。
    
    参数:
        dot_text: DOT 格式的图内容
        min_degree: 保留节点的最小度数阈值，默认5
    
    返回:
        压缩后的 DOT 格式图
    """
    return cdep_compress_dot(dot_text, min_degree=min_degree)

def process_python_files_dataset(
    dataset_path: str,
    graph_types: Union[str, List[str]] = ["cfg", "dfg", "ast"],
    skip_existing: bool = True,
    include_original: bool = True,
    output_format: str = 'graphml',
    generate_nl: bool = False,
) -> Dict[str, str]:
    """
    递归处理数据集文件夹中的所有Python文件，生成对应的JSON图文件
    
    参数:
        dataset_path: 数据集根目录路径
        graph_types: 图表示类型，可以是字符串或列表，支持 'cfg', 'ddg', 'ast'
        skip_existing: 是否跳过已存在的输出文件，默认为True
    
    返回:
        处理结果字典，包含成功和失败的文件列表
    """
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    # 将单个图类型转换为列表
    if isinstance(graph_types, str):
        graph_types = [graph_types.lower()]
    else:
        graph_types = [gt.lower() for gt in graph_types]
    
    # 收集所有Python文件
    python_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"找到 {len(python_files)} 个Python文件")
    
    # 处理结果统计
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    # 使用进度条（如果可用）
    iterator = tqdm(python_files, desc="处理Python文件") if _has_tqdm else python_files
    
    for py_file in iterator:
        try:
            # 检查是否跳过已存在的文件
            base_name = os.path.splitext(py_file)[0]
            json_file = f"{base_name}.json"
            
            if skip_existing and os.path.exists(json_file):
                results["skipped"].append(py_file)
                if not _has_tqdm:
                    print(f"跳过已存在: {py_file}")
                continue
            
            # 读取Python文件内容
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()
            except UnicodeDecodeError:
                # 尝试其他编码
                with open(py_file, 'r', encoding='gbk') as f:
                    code_content = f.read()
            
            # 如果文件为空或只有注释，跳过
            if not code_content.strip() or all(line.strip().startswith('#') or not line.strip() 
                                              for line in code_content.split('\n')):
                results["skipped"].append(py_file)
                if not _has_tqdm:
                    print(f"跳过空文件: {py_file}")
                continue
            
            # 生成图表示（可选压缩）
            graph_results = code_to_graphs(
                code_content,
                graph_types,
                language="py",
                include_original=include_original,
                output_format=output_format,
                generate_nl=generate_nl,
            )
            
            # 添加源文件信息
            output_data = {
                "source_file": py_file,
                "graph_types": graph_types,
                "results": graph_results
            }
            
            # 写入JSON文件
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            results["success"].append(py_file)
            if not _has_tqdm:
                print(f"成功处理: {py_file}")
                
        except Exception as e:
            results["failed"].append(f"{py_file}: {str(e)}")
            if not _has_tqdm:
                print(f"处理失败 {py_file}: {str(e)}")
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"成功: {len(results['success'])} 个文件")
    print(f"跳过: {len(results['skipped'])} 个文件") 
    print(f"失败: {len(results['failed'])} 个文件")
    
    if results["failed"]:
        print("\n失败的文件:")
        for failed in results["failed"][:10]:  # 只显示前10个失败项
            print(f"  - {failed}")
        if len(results["failed"]) > 10:
            print(f"  ... 还有 {len(results['failed']) - 10} 个失败项")
    
    return results

def simplify_and_update_json(json_file: str, degrees: List[int]):
    """
    读取包含图数据的JSON文件，简化图，并将简化结果作为新的图类型添加回原文件。
    
    参数:
        json_file: 输入的JSON文件路径
        degrees: 用于简化的度数列表 (例如 [1, 3, 5])
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "results" not in data:
            return  # 跳过没有'results'键的json

        results = data.get("results", {})
        if "dot" not in results or "nl" not in results:
            return

        # 获取原始图类型，避免重复简化
        original_graph_types = [gt for gt in results["dot"].keys() if "_simp" not in gt]
        
        for degree in degrees:
            for graph_type in original_graph_types:
                # 定义简化后的图类型名称
                simp_graph_type = f"{graph_type}_{degree}_{degree}_simp"

                # 简化 dot 图
                dot_content = results["dot"].get(graph_type, "")
                simplified_dot = simplify_dot_graph(dot_content, degree, degree)
                results["dot"][simp_graph_type] = simplified_dot
                
                # 基于简化后的 DOT 重新生成 NL（避免对 NL 做文本级匹配简化）
                try:
                    if simplified_dot and simplified_dot.strip():
                        # 写入临时文件并用解析器生成 NL
                        with tempfile.NamedTemporaryFile(suffix='.dot', delete=False, mode='w', encoding='utf-8') as tf:
                            tf.write(simplified_dot)
                            tmp_path = tf.name
                        # 猜测图类型（若键名非标准，如包含后缀，尽力推断）
                        gtype = _guess_graph_type_from_key(graph_type) or graph_type
                        parser4 = create_parser(gtype, tmp_path)
                        parser4.parse_dot_file()
                        regen_nl = parser4.explain_in_natural_language()
                        results["nl"][simp_graph_type] = regen_nl
                    else:
                        results["nl"][simp_graph_type] = ""
                except Exception as e:
                    results["nl"][simp_graph_type] = f"从简化DOT生成NL失败: {e}"
                finally:
                    try:
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

        # 写回更新后的JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except json.JSONDecodeError:
        print(f"警告: 无法解析JSON文件: {json_file}")
    except Exception as e:
        print(f"处理 {json_file} 时发生错误: {e}")


def process_and_simplify_dataset(dataset_path: str, degrees: List[int] = [1, 3, 5]):
    """
    遍历数据集中的所有JSON文件，并应用图简化。
    
    参数:
        dataset_path: 数据集根目录
        degrees: 度数列表
    """
    json_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.json') and file != "processing_results.json":
                json_files.append(os.path.join(root, file))
    
    print(f"找到 {len(json_files)} 个JSON文件进行简化")
    
    iterator = tqdm(json_files, desc="简化图") if _has_tqdm else json_files
    
    for json_file in iterator:
        simplify_and_update_json(json_file, degrees)

    print("所有JSON文件简化完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理C/Python代码以生成代码图表示并简化")
    parser.add_argument("--dataset_path", default="./autogen_typeevalpy_benchmark/python_features", help="数据集根目录路径")
    parser.add_argument("--graph-types", nargs="+", default=["cfg", "dfg", "ast"], 
                      choices=["cfg", "dfg", "ast", "ddg"], help="要生成的图类型")
    parser.add_argument("--dataset-type", choices=["python", "bigvul"], default="python",
                      help="数据集类型：python 或 bigvul。选择 bigvul 时将按照 bigvul/<id>/<id>-after|<id>-before(.c) 处理并生成对应输出目录结构。")
    parser.add_argument("--no-skip", action="store_true", default=True, help="不跳过已存在的文件")
    parser.add_argument("--mode", nargs='+', choices=['generate', 'simplify', 'compress', 'translate', 'graphcode', 'all'], default=['graphcode'],
                      help="运行模式可多选: generate(仅从 joern 生成图), translate(将已有图生成 NL), simplify(简化), compress(压缩), graphcode(导出 Node/edge 脚本), all=generate+simplify")
    parser.add_argument("--output-format", choices=['dot', 'graphml'], default='graphml',
                      help="生成图的格式：dot 或 graphml。translate 仅支持 dot。")
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 3, 5], help="用于简化的度数列表")
    parser.add_argument("--compress-min-degree", type=int, default=5,
                        help="压缩时保留节点的最小度数阈值（默认5），度数低于此值的节点可能被合并")
    parser.add_argument("--no-original", action="store_true", default=False,
                        help="仅在 --mode compress 时使用：只输出压缩图，不包含原始图")

    args = parser.parse_args()

    try:
        # BigVul 数据集处理：根据 mode 控制输出目录
        if args.dataset_type == "bigvul":
            print("--- 处理 BigVul 数据集 ---")

            def _ensure_dir(p: str):
                os.makedirs(p, exist_ok=True)

            def _find_variant_file(id_dir: str, vid: str, variant: str) -> Optional[str]:
                """在 <id> 目录下寻找 <id>-<variant> 目录中的 .c 文件，或扁平的 <id>-<variant>.c 文件。"""
                cand_file = os.path.join(id_dir, f"{vid}-{variant}.c")
                if os.path.isfile(cand_file):
                    return cand_file

            def _write_dot(base_dir: str, variant: str, gt: str, dot_content: str):
                type_dir = os.path.join(base_dir, variant, gt)
                _ensure_dir(type_dir)
                out_dot_file = os.path.join(type_dir, f"{gt}.dot")
                with open(out_dot_file, 'w', encoding='utf-8') as f:
                    f.write(dot_content or "")

            def _write_text(base_dir: str, variant: str, gt: str, nl_content: str):
                type_dir = os.path.join(base_dir, variant, gt)
                _ensure_dir(type_dir)
                out_txt_file = os.path.join(type_dir, f"{gt}.txt")
                with open(out_txt_file, 'w', encoding='utf-8') as f:
                    f.write(nl_content or "")
            
            def _write_graph(base_dir: str, variant: str, gt: str, content: str, ext: str):
                type_dir = os.path.join(base_dir, variant, gt)
                _ensure_dir(type_dir)
                out_file = os.path.join(type_dir, f"{gt}{ext}")
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(content or "")
    
            def _read_text_safe(path: str) -> str:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except Exception:
                    return ""

            dataset_root = args.dataset_path
            if not os.path.isdir(dataset_root):
                raise ValueError(f"数据集路径不存在或不是目录: {dataset_root}")

            id_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
            id_dirs.sort()
            print(f"发现 {len(id_dirs)} 个 id 目录")

            # 规范化图类型（保持与 code_to_graphs 一致的映射规则）
            gtypes = []
            for gt in args.graph_types:
                gt_lower = gt.lower()
                if gt_lower == "ddg":
                    gtypes.append("dfg")
                elif gt_lower in {"cfg", "dfg", "ast"}:
                    gtypes.append(gt_lower)
            gtypes = list(dict.fromkeys(gtypes))

            iterator = tqdm(id_dirs, desc="处理BigVul") if _has_tqdm else id_dirs
            for vid in iterator:
                id_dir = os.path.join(dataset_root, vid)

                # 源文件 after / before
                variant_paths = {
                    "after": _find_variant_file(id_dir, vid, "after"),
                    "before": _find_variant_file(id_dir, vid, "before"),
                }
                if not variant_paths["after"] and not variant_paths["before"]:
                    if not _has_tqdm:
                        print(f"跳过(未找到 after/before 源文件): {vid}")
                    continue

                # 1) generate: 仅生成图（不再在此阶段生成 NL）
                if ('generate' in args.mode) or ('all' in args.mode):
                    fmt = (args.output_format or 'graphml').lower()
                    container_key = 'dot' if fmt == 'dot' else 'graphml'
                    out_dir_name = f"{vid}-dot-out" if fmt == 'dot' else f"{vid}-graphml-out"
                    file_ext = '.dot' if fmt == 'dot' else '.xml'
                    out_base = os.path.join(id_dir, out_dir_name)
                    for base in (out_base,):
                        _ensure_dir(os.path.join(base, "after"))
                        _ensure_dir(os.path.join(base, "before"))

                    for variant, c_path in variant_paths.items():
                        if not c_path:
                            continue
                        try:
                            with open(c_path, 'r', encoding='utf-8', errors='ignore') as f:
                                code = f.read()
                        except Exception as e:
                            print(f"读取失败 {c_path}: {e}")
                            continue

                        # 预检查缺失的图类型，避免不必要的解析
                        if args.no_skip:
                            missing_gts = list(gtypes)
                        else:
                            missing_gts = []
                            for gt in gtypes:
                                out_path = os.path.join(out_base, variant, gt, f"{gt}{file_ext}")
                                if not os.path.isfile(out_path):
                                    missing_gts.append(gt)

                        if not missing_gts:
                            if not _has_tqdm:
                                print(f"跳过(基线已完整): {vid}/{variant}")
                            continue

                        graphs = code_to_graphs(
                            code_content=code,
                            graph_types=missing_gts,
                            language="c",
                            include_original=True,
                            output_format=args.output_format,
                        )

                        for gt in missing_gts:
                            content = graphs.get(container_key, {}).get(gt, "")
                            _write_graph(out_base, variant, gt, content if isinstance(content, str) else str(content), file_ext)

                # 1.5) translate: 从已存在图生成 NL（优先 DOT，缺失再回退 GraphML）
                if ('translate' in args.mode) or ('all' in args.mode):
                    text_out_base = os.path.join(id_dir, f"{vid}-text-out")
                    _ensure_dir(os.path.join(text_out_base, "after"))
                    _ensure_dir(os.path.join(text_out_base, "before"))
                    for variant in ("after", "before"):
                        for gt in gtypes:
                            dot_file = os.path.join(id_dir, f"{vid}-dot-out", variant, gt, f"{gt}.dot")
                            xml_file = os.path.join(id_dir, f"{vid}-graphml-out", variant, gt, f"{gt}.xml")
                            picked = None
                            picked_suffix = ''
                            # 优先 DOT
                            if os.path.isfile(dot_file) and os.path.getsize(dot_file) > 0:
                                picked = dot_file
                                picked_suffix = '.dot'
                            elif os.path.isfile(xml_file) and os.path.getsize(xml_file) > 0:
                                picked = xml_file
                                picked_suffix = '.xml'
                            if not picked:
                                continue
                            try:
                                with open(picked, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                if not content.strip():
                                    continue
                                with tempfile.NamedTemporaryFile(suffix=picked_suffix, delete=False, mode='w', encoding='utf-8') as tf:
                                    tf.write(content)
                                    tmp_path = tf.name
                                parser_tr = create_parser(gt, tmp_path)
                                if picked_suffix == '.dot':
                                    parser_tr.parse_dot_file()
                                else:
                                    if hasattr(parser_tr, 'parse_graphml_file'):
                                        parser_tr.parse_graphml_file(tmp_path)
                                    else:
                                        raise RuntimeError('解析器不支持 GraphML 解析')
                                nl_desc = parser_tr.explain_in_natural_language()
                                _write_text(text_out_base, variant, gt, nl_desc)
                            except Exception as e:
                                print(f"NL 生成失败 {vid}/{variant}/{gt}: {e}")
                            finally:
                                try:
                                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                except Exception:
                                    pass

                # 2) simplify: 从已存在 dot/text 目录读取并生成 x-x-simp 与 x-x-simp_dot
                if ('simplify' in args.mode) or ('all' in args.mode):
                    # 读取来源：优先从基线输出读（支持 dot 与 graphml）；如不存在，回退生成一次（每个 gt 只回退一次）
                    dot_in_base = os.path.join(id_dir, f"{vid}-dot-out")
                    graphml_in_base = os.path.join(id_dir, f"{vid}-graphml-out")
                    text_in_base = os.path.join(id_dir, f"{vid}-text-out")

                    # 缓存基线，避免在不同度数下重复生成
                    # baseline_cache[(variant, gt)] = { 'fmt': 'dot'|'graphml', 'graph': str, 'nl': str }
                    baseline_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
                    for variant, c_path in variant_paths.items():
                        if not c_path:
                            continue
                        # 先找出缺失基线的图类型集合
                        missing_base_gts = []
                        for gt in gtypes:
                            in_dot_file = os.path.join(dot_in_base, variant, gt, f"{gt}.dot")
                            in_txt_file = os.path.join(text_in_base, variant, gt, f"{gt}.txt")
                            d_content = _read_text_safe(in_dot_file)
                            n_content = _read_text_safe(in_txt_file)
                            if d_content.strip() and n_content.strip():
                                baseline_cache[(variant, gt)] = { 'fmt': 'dot', 'graph': d_content, 'nl': n_content }
                                continue
                            # 尝试 graphml 基线（允许无 nl）
                            in_xml_file = os.path.join(graphml_in_base, variant, gt, f"{gt}.xml")
                            x_content = _read_text_safe(in_xml_file)
                            if x_content.strip():
                                baseline_cache[(variant, gt)] = { 'fmt': 'graphml', 'graph': x_content, 'nl': n_content }
                            else:
                                missing_base_gts.append(gt)

                        # 仅当缺失时才调用 code_to_graphs（显式生成 DOT 以便简化/翻译）
                        if missing_base_gts:
                            try:
                                with open(c_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    code = f.read()
                                graphs = code_to_graphs(
                                    code_content=code,
                                    graph_types=missing_base_gts,
                                    language="c",
                                    include_original=True,
                                    output_format='dot',
                                    generate_nl=True,
                                )
                                for gt in missing_base_gts:
                                    baseline_cache[(variant, gt)] = {
                                        'fmt': 'dot',
                                        'graph': graphs.get("dot", {}).get(gt, ""),
                                        'nl': graphs.get("nl", {}).get(gt, ""),
                                    }
                            except Exception as e:
                                print(f"回退生成失败 {vid} {variant}: {e}")

                    # 对每个度数 d 生成 <id>-d-d-simp-out 与 <id>-d-d-simp_dot-out
                    for d in args.degrees:
                        simp_text_out_base = os.path.join(id_dir, f"{vid}-{d}-{d}-simp-out")
                        simp_dot_out_base = os.path.join(id_dir, f"{vid}-{d}-{d}-simp_dot-out")
                        simp_graphml_out_base = os.path.join(id_dir, f"{vid}-{d}-{d}-simp_graphml-out")
                        for base in (simp_text_out_base, simp_dot_out_base, simp_graphml_out_base):
                            _ensure_dir(os.path.join(base, "after"))
                            _ensure_dir(os.path.join(base, "before"))

                        for variant, c_path in variant_paths.items():
                            for gt in gtypes:
                                base_info = baseline_cache.get((variant, gt)) or { 'fmt': '', 'graph': '', 'nl': '' }
                                fmt_base = base_info.get('fmt')
                                simp_nl = ""
                                if fmt_base == 'dot':
                                    dot_content = base_info.get('graph', '')
                                    simp_dot = simplify_dot_graph(dot_content or "", d, d)
                                    try:
                                        if simp_dot and simp_dot.strip():
                                            with tempfile.NamedTemporaryFile(suffix='.dot', delete=False, mode='w', encoding='utf-8') as tf:
                                                tf.write(simp_dot)
                                                tmp_path = tf.name
                                            parser4 = create_parser(gt, tmp_path)
                                            parser4.parse_dot_file()
                                            simp_nl = parser4.explain_in_natural_language()
                                        else:
                                            simp_nl = ""
                                    except Exception as e:
                                        simp_nl = f"从简化DOT生成NL失败: {e}"
                                    finally:
                                        try:
                                            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                                os.remove(tmp_path)
                                        except Exception:
                                            pass
                                    _ensure_dir(os.path.join(simp_text_out_base, variant, gt))
                                    _ensure_dir(os.path.join(simp_dot_out_base, variant, gt))
                                    out_simp_txt = os.path.join(simp_text_out_base, variant, gt, f"{gt}.txt")
                                    out_simp_dot = os.path.join(simp_dot_out_base, variant, gt, f"{gt}.dot")
                                    if (not args.no_skip) and os.path.isfile(out_simp_txt) and os.path.isfile(out_simp_dot):
                                        if not _has_tqdm:
                                            print(f"跳过已存在(简化 d={d}): {vid}/{variant}/{gt}")
                                        continue
                                    with open(out_simp_txt, 'w', encoding='utf-8') as f:
                                        f.write(simp_nl)
                                    with open(out_simp_dot, 'w', encoding='utf-8') as f:
                                        f.write(simp_dot)
                                elif fmt_base == 'graphml':
                                    xml_content = base_info.get('graph', '')
                                    simp_xml = simplify_graphml(xml_content or "", d, d)
                                    try:
                                        if simp_xml and simp_xml.strip():
                                            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w', encoding='utf-8') as tf:
                                                tf.write(simp_xml)
                                                tmp_path = tf.name
                                            parser4 = create_parser(gt, tmp_path)
                                            if hasattr(parser4, 'parse_graphml_file'):
                                                parser4.parse_graphml_file(tmp_path)
                                            else:
                                                raise RuntimeError('解析器不支持 GraphML 解析')
                                            simp_nl = parser4.explain_in_natural_language()
                                        else:
                                            simp_nl = ""
                                    except Exception as e:
                                        simp_nl = f"从简化GraphML生成NL失败: {e}"
                                    finally:
                                        try:
                                            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                                os.remove(tmp_path)
                                        except Exception:
                                            pass
                                    _ensure_dir(os.path.join(simp_text_out_base, variant, gt))
                                    _ensure_dir(os.path.join(simp_graphml_out_base, variant, gt))
                                    out_simp_txt = os.path.join(simp_text_out_base, variant, gt, f"{gt}.txt")
                                    out_simp_xml = os.path.join(simp_graphml_out_base, variant, gt, f"{gt}.xml")
                                    if (not args.no_skip) and os.path.isfile(out_simp_txt) and os.path.isfile(out_simp_xml):
                                        if not _has_tqdm:
                                            print(f"跳过已存在(简化 d={d} GraphML): {vid}/{variant}/{gt}")
                                        continue
                                    with open(out_simp_txt, 'w', encoding='utf-8') as f:
                                        f.write(simp_nl)
                                    with open(out_simp_xml, 'w', encoding='utf-8') as f:
                                        f.write(simp_xml)
                                else:
                                    if not _has_tqdm:
                                        print(f"跳过(无基线可简化): {vid}/{variant}/{gt}")

                # 3) compress: 生成压缩图（目录名即 graphtype，如 graphcomp 或 graphspar-<method>-d<delta>）
                if 'compress' in args.mode:
                    comp_dir_name = f"{vid}-cdep-out"
                    comp_out_base = os.path.join(id_dir, comp_dir_name)
                    for variant in ("after", "before"):
                        _ensure_dir(os.path.join(comp_out_base, variant))

                    # 来源：基线 dot；如缺失则回退生成
                    dot_in_base = os.path.join(id_dir, f"{vid}-dot-out")
                    text_in_base = os.path.join(id_dir, f"{vid}-text-out")

                    for variant, c_path in variant_paths.items():
                        for gt in gtypes:
                            # 先做目标输出的跳过判断，避免无效计算
                            comp_dot_path = os.path.join(comp_out_base, variant, gt, f"{gt}.dot")
                            comp_txt_path = os.path.join(comp_out_base, variant, gt, f"{gt}.txt")
                            if (not args.no_skip) and os.path.isfile(comp_dot_path) and os.path.isfile(comp_txt_path):
                                if not _has_tqdm:
                                    print(f"跳过已存在(压缩): {vid}/{variant}/{gt}")
                                continue
                            in_dot_file = os.path.join(dot_in_base, variant, gt, f"{gt}.dot")
                            # in_txt 可选，用于当需要回退生成 NL
                            in_txt_file = os.path.join(text_in_base, variant, gt, f"{gt}.txt")
                            base_dot = _read_text_safe(in_dot_file)
                            if not base_dot.strip() and c_path:
                                # 回退生成基线
                                try:
                                    with open(c_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        code = f.read()
                                    graphs = code_to_graphs(
                                        code_content=code,
                                        graph_types=[gt],
                                        language="c",
                                        include_original=True,
                                        output_format='dot',
                                        generate_nl=True,
                                    )
                                    base_dot = graphs.get("dot", {}).get(gt, base_dot)
                                except Exception as e:
                                    print(f"压缩回退生成失败 {vid} {variant} {gt}: {e}")

                            # 压缩 DOT
                            try:
                                cdot = compress_dot_text(base_dot or "", min_degree=args.compress_min_degree)
                            except Exception as ce:
                                cdot = f"压缩失败: {ce}"

                            # 为压缩 dot 生成 NL
                            nl_comp = ""
                            if cdot and not cdot.startswith("压缩失败"):
                                try:
                                    with tempfile.NamedTemporaryFile(suffix='.dot', delete=False, mode='w', encoding='utf-8') as tf:
                                        tf.write(cdot)
                                        tmp_path = tf.name
                                    parser3 = create_parser(gt, tmp_path)
                                    parser3.parse_dot_file()
                                    nl_comp = parser3.explain_in_natural_language()
                                except Exception as ne:
                                    nl_comp = f"生成压缩图NL描述时发生错误: {ne}"
                                finally:
                                    try:
                                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                            os.remove(tmp_path)
                                    except Exception:
                                        pass

                            # 压缩输出：单目录包含 dot 与 txt（在图类型子目录下）
                            _write_dot(comp_out_base, variant, gt, str(cdot) if cdot is not None else "")
                            _write_text(comp_out_base, variant, gt, nl_comp)

                # 4) graphcode: 读取已有图并导出 Node/edge 脚本（优先 DOT，缺失回退 GraphML）
                if 'graphcode' in args.mode:
                    dag_out_base = os.path.join(id_dir, f"{vid}-graphcode-out")
                    for variant in ("after", "before"):
                        for gt in gtypes:
                            dot_file = os.path.join(id_dir, f"{vid}-dot-out", variant, gt, f"{gt}.dot")
                            xml_file = os.path.join(id_dir, f"{vid}-graphml-out", variant, gt, f"{gt}.xml")
                            picked = None
                            picked_suffix = ''
                            if os.path.isfile(dot_file) and os.path.getsize(dot_file) > 0:
                                picked = dot_file
                                picked_suffix = '.dot'
                            elif os.path.isfile(xml_file) and os.path.getsize(xml_file) > 0:
                                picked = xml_file
                                picked_suffix = '.xml'
                            if not picked:
                                continue
                            try:
                                with open(picked, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                if not content.strip():
                                    continue
                                with tempfile.NamedTemporaryFile(suffix=picked_suffix, delete=False, mode='w', encoding='utf-8') as tf:
                                    tf.write(content)
                                    tmp_path = tf.name
                                parser_dag = create_parser(gt, tmp_path)
                                if picked_suffix == '.dot':
                                    parser_dag.parse_dot_file()
                                else:
                                    if hasattr(parser_dag, 'parse_graphml_file'):
                                        parser_dag.parse_graphml_file(tmp_path)
                                    else:
                                        raise RuntimeError('解析器不支持 GraphML 解析')
                                script = parser_dag.to_add_edge_script()
                                # 写出到 <vid>-graphcode-out/variant/gt/gt.graphcode.py
                                _ensure_dir(os.path.join(dag_out_base, variant, gt))
                                out_py = os.path.join(dag_out_base, variant, gt, f"{gt}.graphcode.py")
                                with open(out_py, 'w', encoding='utf-8') as f:
                                    f.write(script)
                            except Exception as e:
                                print(f"graphcode 导出失败 {vid}/{variant}/{gt}: {e}")
                            finally:
                                try:
                                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                except Exception:
                                    pass

            print("BigVul 处理完成。")
            sys.exit(0)

        # 传统 Python 数据集处理
        if ('generate' in args.mode) or ('all' in args.mode):
            print("--- 开始生成基础图(不进行压缩) ---")
            results = process_python_files_dataset(
                args.dataset_path,
                args.graph_types,
                skip_existing=not args.no_skip,
                # 生成阶段始终保留原始图
                include_original=True,
                output_format=args.output_format,
                generate_nl=False,
            )

            # 保存处理结果到JSON文件
            result_file = os.path.join(args.dataset_path, "processing_results.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n基础图生成结果已保存到: {result_file}")

        if ('simplify' in args.mode) or ('all' in args.mode):
            print("\n--- 开始简化图 ---")
            process_and_simplify_dataset(args.dataset_path, args.degrees)

        if ('translate' in args.mode):
            print("\n--- 将现有图翻译为自然语言 (优先 DOT，回退 GraphML) ---")
            # 遍历 JSON（Python 数据集路径），逐键优先从 dot 取，无则从 graphml 取
            json_files = []
            for root, _, files in os.walk(args.dataset_path):
                for file in files:
                    if file.endswith('.json') and file != 'processing_results.json':
                        json_files.append(os.path.join(root, file))
            for jf in (tqdm(json_files, desc="翻译NL") if _has_tqdm else json_files):
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    res = _ensure_results_dict(data)
                    dots = res.get('dot', {}) or {}
                    xmls = res.get('graphml', {}) or {}
                    nls = res.setdefault('nl', {})
                    all_keys = set(dots.keys()) | set(xmls.keys())
                    for k in all_keys:
                        v = dots.get(k)
                        picked_suffix = None
                        content = None
                        if isinstance(v, str) and v.strip():
                            content = v
                            picked_suffix = '.dot'
                        else:
                            v2 = xmls.get(k)
                            if isinstance(v2, str) and v2.strip():
                                content = v2
                                picked_suffix = '.xml'
                        if not content:
                            continue
                        # 解析并生成 NL
                        with tempfile.NamedTemporaryFile(suffix=picked_suffix, delete=False, mode='w', encoding='utf-8') as tf:
                            tf.write(content)
                            tmp_path = tf.name
                        try:
                            gtype = _guess_graph_type_from_key(k) or 'cfg'
                            parser_tr = create_parser(gtype, tmp_path)
                            if picked_suffix == '.dot':
                                parser_tr.parse_dot_file()
                            else:
                                if hasattr(parser_tr, 'parse_graphml_file'):
                                    parser_tr.parse_graphml_file(tmp_path)
                                else:
                                    continue
                            nls[k] = parser_tr.explain_in_natural_language()
                        finally:
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass
                    with open(jf, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"NL 翻译失败 {jf}: {e}")

        if 'compress' in args.mode:
            print("\n--- 开始压缩现有 JSON 中的 DOT ---")
            # 遍历 JSON，读取 dot 并生成压缩后的 dot，写回新键
            json_files = []
            for root, _, files in os.walk(args.dataset_path):
                for file in files:
                    if file.endswith('.json') and file != 'processing_results.json':
                        json_files.append(os.path.join(root, file))
            for jf in (tqdm(json_files, desc="压缩DOT") if _has_tqdm else json_files):
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        print(f"跳过(非dict结构): {jf}")
                        continue
                    res = data.get('results')
                    if not isinstance(res, dict):
                        print(f"跳过(无results或非dict): {jf}")
                        continue
                    dots = res.get('dot')
                    if not isinstance(dots, dict):
                        print(f"跳过(无dot或非dict): {jf}")
                        continue
                    nls = res.get('nl')
                    if not isinstance(nls, dict):
                        res['nl'] = {}
                        nls = res['nl']
                    new_dots = {}
                    new_nls = {}
                    for k, v in dots.items():
                        if not isinstance(v, str) or not v.strip():
                            continue
                        # 跳过已经是压缩后键的内容，避免重复
                        if k.startswith('cfg_comp_') or k.startswith('dfg_comp_') or k.startswith('ast_comp_'):
                            continue
                        try:
                            cdot = compress_dot_text(v, min_degree=args.compress_min_degree)
                            new_key = f"{k}_comp_cdep"
                            new_dots[new_key] = cdot
                            # 为压缩 DOT 生成 NL
                            gtype = _guess_graph_type_from_key(k)
                            if gtype:
                                # 写入临时文件供解析器
                                try:
                                    with tempfile.NamedTemporaryFile(suffix='.dot', delete=False, mode='w', encoding='utf-8') as tf:
                                        tf.write(cdot)
                                        tmp_path = tf.name
                                    parser3 = create_parser(gtype, tmp_path)
                                    parser3.parse_dot_file()
                                    nl3 = parser3.explain_in_natural_language()
                                    new_nls[new_key] = nl3
                                except Exception as ne:
                                    new_nls[new_key] = f"生成压缩图NL描述时发生错误: {ne}"
                                finally:
                                    try:
                                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                            os.remove(tmp_path)
                                    except Exception:
                                        pass
                        except Exception as ce:
                            new_key = f"{k}_comp_cdep"
                            new_dots[new_key] = f"压缩失败: {ce}"
                            new_nls[new_key] = f"压缩失败: {ce}"
                    dots.update(new_dots)
                    nls.update(new_nls)
                    with open(jf, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"压缩失败 {jf}: {e}")

        # graphcode for Python dataset: 从 JSON 产生 .graphcode.py（逐键优先 dot，回退 graphml）
        if 'graphcode' in args.mode:
            print("\n--- 导出 Node/edge 脚本 (graphcode) ---")
            json_files = []
            for root, _, files in os.walk(args.dataset_path):
                for file in files:
                    if file.endswith('.json') and file != 'processing_results.json':
                        json_files.append(os.path.join(root, file))
            for jf in (tqdm(json_files, desc="graphcode") if _has_tqdm else json_files):
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    res = _ensure_results_dict(data)
                    dots = res.get('dot', {}) or {}
                    xmls = res.get('graphml', {}) or {}
                    all_keys = set(dots.keys()) | set(xmls.keys())
                    # 将生成的 graphcode 脚本写回 JSON 的 results.graphcode 字段（按键存储）
                    # 确保 data 中的 results 指向我们可以修改的 dict
                    data_results = res
                    try:
                        # 如果原始 data['results'] 不是 dict（比如 list），覆盖回写为 dict
                        data['results'] = data_results
                    except Exception:
                        pass
                    graphcode_map: Dict[str, str] = data_results.setdefault('graphcode', {})
                    for k in all_keys:
                        # 逐键优先取 dot，再取 graphml
                        content = None
                        picked_suffix = None
                        v = dots.get(k)
                        if isinstance(v, str) and v.strip():
                            content = v
                            picked_suffix = '.dot'
                        else:
                            v2 = xmls.get(k)
                            if isinstance(v2, str) and v2.strip():
                                content = v2
                                picked_suffix = '.xml'
                        if not content:
                            continue
                        gtype = _guess_graph_type_from_key(k)
                        if gtype not in {"cfg", "dfg", "ast"}:
                            continue
                        with tempfile.NamedTemporaryFile(suffix=picked_suffix, delete=False, mode='w', encoding='utf-8') as tf:
                            tf.write(content)
                            tmp_path = tf.name
                        try:
                            parser_dag = create_parser(gtype or 'cfg', tmp_path)
                            if picked_suffix == '.dot':
                                parser_dag.parse_dot_file()
                            else:
                                if hasattr(parser_dag, 'parse_graphml_file'):
                                    parser_dag.parse_graphml_file(tmp_path)
                                else:
                                    continue
                            script = parser_dag.to_add_edge_script()
                            # 写回到 JSON 中
                            graphcode_map[k] = script
                        finally:
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass
                    # 写回 JSON 文件（覆盖原文件）
                    try:
                        with open(jf, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"写回 JSON 失败 {jf}: {e}")
                except Exception as e:
                    print(f"graphcode 导出失败 {jf}: {e}")

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

