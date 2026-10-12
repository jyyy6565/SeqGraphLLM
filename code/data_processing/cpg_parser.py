import re
import copy
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

class CPGParser:
    """统一的CPG解析器 - 简单直接"""
    
    def __init__(self, dot_file_path: str = ''):
        self.nodes = {}  # 存储节点信息
        self.edges = []  # 存储边信息
        self.dot_file_path = dot_file_path
        self.id_mapping = {}  # 用于存储原始ID到新ID的映射
        self.current_id = 0  # 用于生成新的连续ID
    
    @staticmethod
    def get_cpg_edge_types() -> Dict[str, str]:
        """CPG所有边类型的描述映射"""
        return {
            'ALIAS_OF': 'Alias Relationship',
            'ARGUMENT': 'Function Argument',
            'AST': 'Abstract Syntax Tree',
            'BINDS': 'Variable Binding',
            'CALL': 'Function Call',
            'CDG': 'Control Dependency',
            'CFG': 'Control Flow',
            'CONDITION': 'Conditional Branch',
            'CONTAINS': 'Contains Relationship',
            'DOMINATE': 'Dominance Relationship',
            'EVAL_TYPE': 'Type Evaluation',
            'PARAMETER_LINK': 'Parameter Link',
            'POST_DOMINATE': 'Post Dominance',
            'REACHING_DEF': 'Reaching Definition',
            'RECEIVER': 'Method Receiver',
            'REF': 'Reference',
            'SOURCE_FILE': 'Source File Link'
        }
    
    @staticmethod
    def get_graph_edge_mapping() -> Dict[str, List[str]]:
        """各图类型使用的边类型"""
        return {
            "ast": ["AST"],
            "cfg": ["CFG"],
            "dfg": ["REACHING_DEF"]
        }
    
    @staticmethod
    def get_cpg_node_types() -> Dict[str, str]:
        """CPG标准节点类型映射"""
        return {
            'BINDING': 'Binding',
            'BLOCK': 'Code Block',
            'CALL': 'Function Call',
            'CONTROL_STRUCTURE': 'Control Structure',
            'FIELD_IDENTIFIER': 'Field Identifier',
            'FILE': 'Source File',
            'IDENTIFIER': 'Identifier',
            'JUMP_TARGET': 'Jump Target',
            'LITERAL': 'Literal Value',
            'LOCAL': 'Local Variable',
            'META_DATA': 'Metadata',
            'METHOD': 'Method',
            'METHOD_REF': 'Method Reference',
            'METHOD_PARAMETER_IN': 'Input Parameter',
            'METHOD_PARAMETER_OUT': 'Output Parameter',
            'METHOD_RETURN': 'Method Return',
            'MODIFIER': 'Modifier',
            'NAMESPACE': 'Namespace',
            'NAMESPACE_BLOCK': 'Namespace Block',
            'RETURN': 'Return Statement',
            'TYPE': 'Type',
            'TYPE_DECL': 'Type Declaration'
        }
        
    def _clean_html(self, text: str) -> str:
        """清理HTML实体"""
        html_entities = {
            '&lt;': '<', '&gt;': '>', '&quot;': '"', '&amp;': '&', '&#39;': "'"
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        text = re.sub(r'<BR\s*/?>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def parse_dot_file(self) -> Tuple[List[Dict], List[Dict]]:
        """读取并解析DOT文件"""
        try:
            with open(self.dot_file_path, 'r', encoding='utf-8') as f:
                dot_content = f.read()
            return self.parse_dot(dot_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"DOT file not found: {self.dot_file_path}")
        except Exception as e:
            raise Exception(f"Error parsing DOT file: {str(e)}")

    @staticmethod
    def _strip_dot_comments(dot_content: str) -> str:
        """去掉引号外的 // 与 /* */ 注释。

        修复：旧版用正则全局删 // 到行尾，会把 CODE 里的 URL（'https://...'）后半段连同
        后续属性一起删掉，导致该节点语句损坏被丢弃（MT4Py 实测每图丢十几个节点）。
        """
        out = []
        i, n = 0, len(dot_content)
        quoted = False
        while i < n:
            ch = dot_content[i]
            if quoted:
                out.append(ch)
                if ch == '\\' and i + 1 < n:
                    out.append(dot_content[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    quoted = False
                i += 1
                continue
            if ch == '"':
                quoted = True
                out.append(ch)
                i += 1
                continue
            if dot_content.startswith('//', i):
                j = dot_content.find('\n', i)
                i = n if j < 0 else j
                continue
            if dot_content.startswith('/*', i):
                j = dot_content.find('*/', i + 2)
                i = n if j < 0 else j + 2
                continue
            out.append(ch)
            i += 1
        return ''.join(out)

    def parse_dot(self, dot_content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析DOT内容"""
        dot_content = self._strip_dot_comments(dot_content)
        
        # 检测CPG格式
        if 'label=' in dot_content and ';' in dot_content:
            return self._parse_cpg_format(dot_content)
        else:
            return self._parse_traditional_format(dot_content)

    # ============== GraphML 解析 ==============
    def _graphml_keymap(self, root: ET.Element) -> Dict[str, str]:
        """构建 keyId -> attr.name 的映射，用于把 <data key="..."> 映射成属性名。"""
        ns_uri = None
        if '}' in root.tag:
            ns_uri = root.tag.split('}')[0].strip('{')
        def _tag(t: str) -> str:
            return f"{{{ns_uri}}}{t}" if ns_uri else t
        keymap: Dict[str, str] = {}
        for k in root.findall(_tag('key')) if ns_uri else root.findall('key'):
            kid = k.get('id')
            aname = k.get('attr.name')
            if kid and aname:
                keymap[kid] = aname
        return keymap

    def parse_graphml(self, xml_content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析 GraphML 内容，填充 self.nodes/self.edges，与 DOT 路径统一结构。"""
        # reset
        self.nodes = {}
        self.edges = []
        self.id_mapping = {}
        self.current_id = 0

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise Exception(f"GraphML 解析失败: {e}")

        ns_uri = None
        if '}' in root.tag:
            ns_uri = root.tag.split('}')[0].strip('{')
        def _tag(t: str) -> str:
            return f"{{{ns_uri}}}{t}" if ns_uri else t

        keymap = self._graphml_keymap(root)
        graphs = list(root.findall(_tag('graph'))) if ns_uri else list(root.findall('graph'))
        if not graphs:
            graphs = list(root.iterfind('.//'+_tag('graph'))) if ns_uri else list(root.iterfind('.//graph'))
        if not graphs:
            return [], []
        g0 = graphs[0]

        # 解析节点
        for n in list(g0):
            if n.tag != _tag('node'):
                continue
            orig_id = n.get('id') or ''
            new_id = str(self.current_id)
            self.id_mapping[orig_id] = new_id
            self.current_id += 1

            # 收集 data
            attrs = {}
            for d in list(n):
                if d.tag == _tag('data'):
                    k = d.get('key') or ''
                    v = (d.text or '').strip()
                    name = keymap.get(k, k)
                    attrs[name] = v

            node_type = attrs.get('labelV', 'UNKNOWN')
            node_content = attrs.get('CODE', attrs.get('NAME', node_type))

            self.nodes[new_id] = {
                'ID': new_id,
                'original_id': orig_id,
                'label': node_content,
                'kind': node_type,
                'attributes': attrs,
            }

        # 解析边
        for e in list(g0):
            if e.tag != _tag('edge'):
                continue
            s = e.get('source')
            t = e.get('target')
            etype = 'UNKNOWN'
            eattrs = {}
            for d in list(e):
                if d.tag == _tag('data'):
                    k = d.get('key') or ''
                    v = (d.text or '').strip()
                    name = keymap.get(k, k)
                    eattrs[name] = v
                    if name == 'labelE':
                        etype = v or etype
            ns = self.id_mapping.get(s)
            nt = self.id_mapping.get(t)
            if ns is not None and nt is not None:
                self.edges.append({
                    'between': [ns, nt],
                    'edgeType': etype,
                    'attributes': eattrs,
                })

        return list(self.nodes.values()), self.edges

    def parse_graphml_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml = f.read()
            return self.parse_graphml(xml)
        except FileNotFoundError:
            raise FileNotFoundError(f"GraphML file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error parsing GraphML file: {str(e)}")
    
    @staticmethod
    def _split_dot_statements(dot_content: str):
        """按引号外的分号切分 DOT 语句体。

        修复：旧实现按行匹配节点/边，CODE 含换行的节点（多行调用、函数定义等）
        与属性值含 ']' 的节点/边会被静默丢弃（实测丢 ~3.4% 节点，偏向多行复杂
        语句）。语句级切分可正确处理引号内的换行、分号与方括号。
        """
        body_start = dot_content.find('{')
        body_end = dot_content.rfind('}')
        if body_start < 0:
            return
        body = dot_content[body_start + 1: body_end if body_end > body_start else None]
        current = []
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

    def _parse_cpg_format(self, dot_content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析CPG格式（语句级，容忍属性值内的换行/分号/方括号）"""
        # 节点 id 不限定为纯数字：DBC v2 链收缩会合成 "blk0" 这类 BLOCK 节点
        node_re = re.compile(r'^"([^"]+)"\s*\[(.*)\]\s*$', re.DOTALL)
        edge_re = re.compile(r'^"([^"]+)"\s*->\s*"([^"]+)"\s*(?:\[(.*)\])?\s*$', re.DOTALL)

        pending_edges = []
        for stmt in self._split_dot_statements(dot_content):
            edge_match = edge_re.match(stmt)
            if edge_match:
                pending_edges.append(edge_match)
                continue
            node_match = node_re.match(stmt)
            if not node_match:
                continue
            node_id = node_match.group(1)
            attrs = self._parse_attributes(node_match.group(2))

            new_id = str(self.current_id)
            self.id_mapping[node_id] = new_id
            self.current_id += 1

            # 原始行为：直接使用 label 作为节点类型（与 CPG 导出一致）
            node_type = attrs.get('label', 'UNKNOWN')
            node_content = attrs.get('CODE', attrs.get('NAME', node_type))

            self.nodes[new_id] = {
                'ID': new_id,
                'original_id': node_id,
                'label': node_content,
                'kind': node_type,
                'attributes': attrs
            }

        for match in pending_edges:
            src, dest = match.group(1), match.group(2)
            edge_attrs = self._parse_attributes(match.group(3) or '')
            edge_label = edge_attrs.get('label', 'UNKNOWN')

            new_src = self.id_mapping.get(src)
            new_dest = self.id_mapping.get(dest)
            if new_src is not None and new_dest is not None:
                self.edges.append({
                    'between': [new_src, new_dest],
                    'edgeType': edge_label,
                    'attributes': edge_attrs
                })

        return list(self.nodes.values()), self.edges
    
    def _parse_traditional_format(self, dot_content: str) -> Tuple[List[Dict], List[Dict]]:
        """解析传统格式"""
        # 解析节点
        node_pattern = r'"([^"]+)"\s*\[label\s*=\s*<((?:[^<>]|<\S[^<>]*>)+)>'
        for match in re.finditer(node_pattern, dot_content):
            node_id = match.group(1)
            label = match.group(2)
            
            new_id = str(self.current_id)
            self.id_mapping[node_id] = new_id
            self.current_id += 1
            
            # 处理标签内容
            br_parts = re.split(r'<BR\s*/?>', label, flags=re.IGNORECASE, maxsplit=1)
            if len(br_parts) > 1:
                node_type = self._clean_html(br_parts[0]).strip()
                node_content = self._clean_html(br_parts[1]).strip()
            else:
                cleaned_label = self._clean_html(label)
                if ',' in cleaned_label:
                    content = cleaned_label.strip('()')
                    node_type, node_content = content.split(',', 1)
                    node_type = node_type.strip()
                    node_content = node_content.strip()
                else:
                    node_type = 'unknown'
                    node_content = cleaned_label.strip()
            
            self.nodes[new_id] = {
                'ID': new_id,
                'original_id': node_id,
                'label': node_content,
                'kind': node_type
            }
        
        # 解析边
        edge_pattern = r'"([^"]+)"\s*->\s*"([^"]+)"'
        for match in re.finditer(edge_pattern, dot_content):
            src, dest = match.group(1), match.group(2)
            new_src = self.id_mapping.get(src)
            new_dest = self.id_mapping.get(dest)
            if new_src is not None and new_dest is not None:
                self.edges.append({
                    'between': [new_src, new_dest],
                    'edgeType': 'UNKNOWN'
                })
            
        return list(self.nodes.values()), self.edges
    
    _DOT_ESCAPES = {'n': '\n', 'r': '\r', 't': '\t', '"': '"', '\\': '\\'}

    @classmethod
    def _dot_unescape(cls, value: str) -> str:
        """DOT 引号串单遍反转义：\\n \\r \\t \\" \\\\ ；其他 \\x 原样保留（如 \\l）。"""
        return re.sub(r'\\(.)', lambda m: cls._DOT_ESCAPES.get(m.group(1), m.group(0)), value,
                      flags=re.DOTALL)

    def _parse_attributes(self, attr_str: str) -> Dict:
        """解析属性字符串"""
        attrs = {}
        if not attr_str:
            return attrs
        
        # key="value" 或 key=value；兼容 Joern 属性中的转义引号和换行。
        pattern = r'(\w+)\s*=\s*(?:"((?:\\.|[^"\\])*)"|([^\s\]]+))'
        for match in re.finditer(pattern, attr_str):
            key = match.group(1)
            quoted_value = match.group(2)
            unquoted_value = match.group(3)
            value = quoted_value if quoted_value is not None else unquoted_value
            if quoted_value is not None:
                # 单遍反转义：旧版链式 replace 会把 C 字面量 '\n'（DOT 中为 \\n）
                # 错误解成 反斜杠+换行
                value = self._dot_unescape(value)
            attrs[key] = value
            
        return attrs

    # ---------------- 新 NL 输出：先节点后边，英文，边括号用“解释后类型” ----------------

    def _format_line_info(self, attrs: Dict) -> str:
        """将行号信息格式化为简短英文: at line X / spans lines A–B / no line info"""
        line = attrs.get('LINE_NUMBER') or attrs.get('lineNumber')
        line_end = attrs.get('LINE_NUMBER_END') or attrs.get('lineNumberEnd')
        try:
            if line and line_end and str(line) != str(line_end):
                return f"spans lines {line}–{line_end}"
            if line:
                return f"at line {line}"
            return "no line info"
        except Exception:
            return f"at line {line}" if line else "no line info"

    def _friendly_node_type(self, raw_kind: str) -> str:
        kinds = self.get_cpg_node_types()
        return kinds.get(raw_kind, raw_kind if raw_kind else 'Unknown')

    def _node_phrase(self, node: Dict) -> str:
        """生成节点的简短自然语言短语，例如：assignment 'a = b' / call 'f(x)' / name 'x'。"""
        kind = (node.get('kind') or '').upper()
        attrs = node.get('attributes', {}) or {}
        code = (attrs.get('CODE') or node.get('label') or '').strip()
        name = (attrs.get('NAME') or '').strip()

        def quoted(text: str) -> str:
            return f"'{text}'" if text else "''"

        # CALL: 区分赋值与普通调用
        if kind == 'CALL':
            mfn = (attrs.get('METHOD_FULL_NAME') or '').lower()
            is_assign = ('assignment' in mfn) or (
                '=' in code and not any(op in code for op in ['==', '>=', '<=', '!='])
            )
            if is_assign:
                return f"assignment {quoted(code)}"
            return f"call {quoted(code if code else name)}"

        # IDENTIFIER/LOCAL/FIELD_IDENTIFIER → 当成 name
        if kind in {'IDENTIFIER', 'LOCAL', 'FIELD_IDENTIFIER'}:
            value = code or name
            return f"name {quoted(value)}"

        # METHOD: def 语句，偏向输出 assignment 'name = def name(...)'
        if kind == 'METHOD':
            if code.lower().startswith('def '):
                funcname = name
                if not funcname:
                    m = re.search(r'^def\s+([A-Za-z_][\w]*)', code)
                    funcname = m.group(1) if m else ''
                if funcname:
                    return f"assignment {quoted(f'{funcname} = {code}') }"
                return f"definition {quoted(code)}"
            return f"definition {quoted(code or name or 'function')}"

        # METHOD_REF: 函数值的引用（作为一等值被赋给变量或传参）
        if kind == 'METHOD_REF':
            refname = name
            if not refname and code:
                m = re.search(r'^def\s+([A-Za-z_][\w]*)', code)
                refname = m.group(1) if m else ''
            if refname:
                return f"function reference {quoted(refname)}"
            return f"function reference {quoted(code or name)}"

        # METHOD_RETURN: 更自然的表述
        if kind == 'METHOD_RETURN':
            t = attrs.get('TYPE_FULL_NAME')
            if t and t != 'ANY':
                return f"function return value {quoted(t)}"
            return "function return value"

        # RETURN
        if kind == 'RETURN':
            return f"return {quoted(code)}"

        # LITERAL
        if kind == 'LITERAL':
            return f"literal {quoted(code)}"

        # CONTROL_STRUCTURE
        if kind == 'CONTROL_STRUCTURE':
            return f"control {quoted(code)}"

        # 通用回退
        friendly = self._friendly_node_type(kind).lower()
        payload = code or name
        if payload:
            return f"{friendly} {quoted(payload)}"
        return friendly

    def _node_order(self) -> List[str]:
        """按照 DOT 中出现的顺序返回内部ID列表（依赖 dict 的插入顺序）。"""
        return [n['ID'] for n in self.nodes.values()]

    def explain_in_natural_language(self) -> str:
        """生成更自然且可机读的描述：先列节点（行信息），再列边（句式）。"""
        # 构建 NodeX 映射（按出现顺序）
        ordered_ids = self._node_order()
        id_to_idx = {nid: i + 1 for i, nid in enumerate(ordered_ids)}

        lines: List[str] = []

        # Nodes 部分（按 DOT 顺序，简短行信息）
        lines.append("Nodes:")
        for nid in ordered_ids:
            node = self.nodes[nid]
            info = self._format_line_info(node.get('attributes', {}))
            phrase = self._node_phrase(node)
            lines.append(f"- Node{id_to_idx[nid]}: {info} — {phrase}.")

        # Edges 部分（保持 DOT 顺序，句式固定，括号为解释后类型）
        lines.append("")
        lines.append("Edges:")

        if not self.edges:
            lines.append("(none)")
            return "\n".join(lines) + "\n"

        type_to_phrase = {
            'CFG': 'Control flows',
            'REACHING_DEF': 'Data flows',
            'AST': 'Syntax links'
        }

        for edge in self.edges:
            src_id, dest_id = edge['between']
            src_node = self.nodes.get(src_id)
            dest_node = self.nodes.get(dest_id)
            if not src_node or not dest_node:
                # 回退：查找（兼容旧结构）
                src_node = next((n for n in self.nodes.values() if n['ID'] == src_id), None)
                dest_node = next((n for n in self.nodes.values() if n['ID'] == dest_id), None)
            if not src_node or not dest_node:
                continue

            edge_type = edge.get('edgeType', '')
            phrase = type_to_phrase.get(edge_type, 'Relates')

            src_type = self._friendly_node_type(src_node.get('kind', ''))
            dst_type = self._friendly_node_type(dest_node.get('kind', ''))

            s_idx = id_to_idx.get(src_node['ID'], '?')
            d_idx = id_to_idx.get(dest_node['ID'], '?')

            if phrase in ('Control flows', 'Data flows', 'Syntax links'):
                lines.append(f"- {phrase} from Node{s_idx} ({src_type}) to Node{d_idx} ({dst_type}).")
            else:
                lines.append(f"- {phrase} between Node{s_idx} ({src_type}) and Node{d_idx} ({dst_type}).")

        return "\n".join(lines) + "\n"

    # ---------------- 图 -> 代码 (Node 变量 + add_edge 调用) ----------------
    def to_add_edge_script(self,
                           edge_whitelist: Optional[List[str]] = None,
                           friendly_edge: bool = True,
                           include_line_info: bool = True) -> str:
        """将当前已解析的图导出为简单的 Python 代码片段。

        输出示例:
            # Auto-generated CPG snippet
            # Nodes
            Node1 = "assignment 'a = b'"  # CALL id=0
            Node2 = "name 'a'"            # IDENTIFIER id=1
            # Edges
            add_edge(Node1, "control flow", Node2)

        参数:
            edge_whitelist: 仅保留这些 edgeType; None 表示全部
            friendly_edge:  True 使用人类友好描述 (如 control flow), False 使用原始 edgeType
            include_line_info: 若 True, 在节点注释中追加行号信息
        """
        ordered_ids = self._node_order()
        id_to_idx = {nid: i + 1 for i, nid in enumerate(ordered_ids)}

        edge_desc_map = self.get_cpg_edge_types()

        # 生成节点行
        node_lines: List[str] = []
        node_lines.append("# Nodes")
        for nid in ordered_ids:
            node = self.nodes[nid]
            phrase = self._node_phrase(node)
            kind = node.get('kind', 'UNKNOWN')
            comment_parts = [kind, f"id={nid}"]
            if include_line_info:
                comment_parts.append(self._format_line_info(node.get('attributes', {})))
            comment = ' / '.join(comment_parts)
            node_lines.append(f"Node{id_to_idx[nid]} = \"{phrase}\"  # {comment}")

        # 生成边行
        edge_lines: List[str] = []
        edge_lines.append("# Edges")
        def _edge_label(et: str) -> str:
            if not friendly_edge:
                return et
            # 使用描述（小写）若存在
            human = edge_desc_map.get(et, et)
            return human.lower()

        for e in self.edges:
            etype = e.get('edgeType', '')
            if edge_whitelist and etype not in edge_whitelist:
                continue
            s, t = e.get('between', [None, None])
            if s not in id_to_idx or t not in id_to_idx:
                continue
            edge_lines.append(
                f"add_edge(Node{id_to_idx[s]}, \"{_edge_label(etype)}\", Node{id_to_idx[t]})"
            )

        # 若无边, 也输出提示
        if len(edge_lines) == 1:
            edge_lines.append("# (no edges)")

        header = [
            "# Variables represent nodes; add_edge(src, relation, dst)",
            "",
        ]

        return "\n".join(header + node_lines + ["",] + edge_lines) + "\n"


def extract_subgraphs_from_cpg(cpg_content: str, requested_types: List[str]) -> Dict[str, str]:
    """从CPG中提取子图"""
    edge_mapping = CPGParser.get_graph_edge_mapping()
    result_graphs = {}
    
    for graph_type in requested_types:
        if graph_type not in edge_mapping:
            continue
            
        keep_edge_types = edge_mapping[graph_type]
        
        # 找到需要保留的边和节点
        edge_pattern = r'"(\d+)"\s*->\s*"(\d+)"\s*\[([^\]]*label\s*=\s*"([^"]*)"[^\]]*)\]\s*;'
        edges_to_keep = []
        involved_nodes = set()
        
        for match in re.finditer(edge_pattern, cpg_content):
            src_id = match.group(1)
            dest_id = match.group(2)
            edge_label = match.group(4)
            
            if edge_label in keep_edge_types:
                edges_to_keep.append(match.group(0))
                involved_nodes.add(src_id)
                involved_nodes.add(dest_id)
        
        # 找到需要保留的节点
        node_pattern = r'^\s*"(\d+)"\s*\[([^\]]+)\]\s*;$'
        nodes_to_keep = []
        
        for match in re.finditer(node_pattern, cpg_content, re.MULTILINE):
            node_id = match.group(1)
            if node_id in involved_nodes:
                nodes_to_keep.append(match.group(0))
        
        # 重建图
        lines = [f"digraph {graph_type} {{"]
        lines.extend(nodes_to_keep)
        lines.extend(edges_to_keep)
        lines.append("}")
        
        result_graphs[graph_type] = '\n'.join(lines)
    
    return result_graphs


def extract_subgraphs_from_graphml(cpg_xml: str, requested_types: List[str]) -> Dict[str, str]:
    """从 GraphML 格式的 CPG 中提取子图（ast/cfg/dfg）。

    规则：
    - ast: 仅保留 labelE=AST 的边以及其关联节点
    - cfg: 仅保留 labelE=CFG 的边以及其关联节点
    - dfg: 仅保留 labelE=REACHING_DEF 的边以及其关联节点
    - 同时复制 <key> 定义，保持 GraphML 的完整性
    """
    edge_mapping = CPGParser.get_graph_edge_mapping()

    # 解析 XML（兼容 namespace）
    try:
        root = ET.fromstring(cpg_xml)
    except ET.ParseError:
        return {}

    ns_uri = None
    if '}' in root.tag:
        ns_uri = root.tag.split('}')[0].strip('{')

    def _tag(t: str) -> str:
        return f"{{{ns_uri}}}{t}" if ns_uri else t

    keys = list(root.findall('key')) if not ns_uri else list(root.findall(_tag('key')))
    graphs = list(root.findall('graph')) if not ns_uri else list(root.findall(_tag('graph')))
    if not graphs:
        # 尝试深层查找
        graphs = list(root.iterfind('.//graph')) if not ns_uri else list(root.iterfind('.//'+_tag('graph')))
    if not graphs:
        return {}
    base_graph = graphs[0]

    # 收集节点
    node_by_id: Dict[str, ET.Element] = {}
    for n in list(base_graph):
        if n.tag == _tag('node'):
            nid = n.get('id')
            if nid:
                node_by_id[nid] = n

    # 收集边与 labelE
    edges: List[Tuple[ET.Element, Optional[str]]] = []
    for e in list(base_graph):
        if e.tag == _tag('edge'):
            label_e: Optional[str] = None
            for d in list(e):
                if d.tag == _tag('data') and d.get('key') in ('labelE', 'edge_label', 'label'):
                    label_e = (d.text or '').strip()
                    break
            edges.append((e, label_e))

    results: Dict[str, str] = {}
    for graph_type in requested_types:
        keep_edge_types = edge_mapping.get(graph_type, [])
        kept_edges: List[ET.Element] = []
        kept_nodes: set = set()
        for e, lab in edges:
            if lab in keep_edge_types:
                kept_edges.append(e)
                s = e.get('source')
                t = e.get('target')
                if s:
                    kept_nodes.add(s)
                if t:
                    kept_nodes.add(t)

        # 组装新的 GraphML 文档（深拷贝元素，避免移动原节点）
        new_root = ET.Element(_tag('graphml'))
        for k in keys:
            new_root.append(copy.deepcopy(k))
        new_g = ET.SubElement(new_root, _tag('graph'), attrib={
            'id': f'G_{graph_type}',
            'edgedefault': base_graph.get('edgedefault', 'directed'),
        })
        for nid in kept_nodes:
            n = node_by_id.get(nid)
            if n is not None:
                new_g.append(copy.deepcopy(n))
        for e in kept_edges:
            new_g.append(copy.deepcopy(e))

        try:
            ET.indent(new_root)  # Python 3.9+
        except Exception:
            pass
        results[graph_type] = ET.tostring(new_root, encoding='utf-8').decode('utf-8')

    return results


def extract_subgraphs(cpg_content: str, requested_types: List[str], fmt: str = 'dot') -> Dict[str, str]:
    """统一的子图提取入口：fmt in {'dot','graphml'}"""
    fmt_l = (fmt or 'dot').lower()
    if fmt_l == 'graphml':
        return extract_subgraphs_from_graphml(cpg_content, requested_types)
    # 默认按 DOT 处理
    return extract_subgraphs_from_cpg(cpg_content, requested_types)


# 简化的工厂函数
def create_parser(parser_type: str, dot_file_path: str = '') -> CPGParser:
    """创建CPG解析器 - 所有类型都用同一个解析器"""
    return CPGParser(dot_file_path)
