import json
import os
import sys
import sqlite3
import re
from typing import Dict, List, Tuple, Optional, Union
from openai import OpenAI
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager, Lock, freeze_support
from functools import partial
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'dataset', 'manytypes4py'))

# 修回实验（v2 数据，PLAN.md §4）：GraphDot-v2 由 GraphDot 经复合节点 CODE 缩写得到（D11）
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data_processing')))
from graph_budget_fit import fit_graph, count_tokens as _count_tokens  # noqa: E402  预算内图裁剪（budget-fit）

TI_MAX_OUTPUT_TOKENS = 2000   # budget-fit 预算里给输出预留的量（协议定值，不要动）
# 实际请求给的上限。与漏洞检测侧同一处理：预算按 2000 预留，但请求放宽，
# 免得模型把 JSON 写到一半被切断 —— 截断的响应 json.loads 必然失败，
# 旧行为是整条请求作废（全部变量记 -1），实测吃掉约 4% 的请求，且集中在个别文件上。
TI_REQUEST_MAX_OUTPUT_TOKENS = int(os.environ.get('TI_REQUEST_MAX_OUTPUT_TOKENS', 3000))
TI_SAFETY_RATIO = 0.9
TI_SYSTEM_PROMPT = ("You are a professional Python type inference expert. Use step-by-step reasoning to analyze code, paying special attention to variable definition methods, scope, and context. For class attributes, distinguish between class-level attributes (Type[class_name]) and instance-level attributes. For framework-specific code, consider the framework's type system rules and prioritize using generalized types. Avoid over-inferring specific implementation types; prefer generalized types like Any when uncertain. After carefully analyzing each variable, return the results in JSON format.")

# budget-fit 的图源：表示目录名 -> (DOT 目录名, 序列化格式)。放不下窗口时读 DOT、度数裁剪、再序列化。
GRAPH_DOT_SOURCE = {
    'GraphText-v2': ('GraphDot-v2', 'text'), 'GraphDot-v2': ('GraphDot-v2', 'dot'),
    'GraphCode-v2': ('GraphDot-v2', 'graphcode'),
    'DBC-v2-GraphText': ('DBC-v2-GraphDot', 'text'), 'DBC-v2-GraphDot': ('DBC-v2-GraphDot', 'dot'),
    # E4-TI 的伪图对照（8/19）。没有这一条的话 budget-fit 找不到 DOT 源就不裁剪，
    # 超窗的请求会被 vLLM 直接 400 → 记成无效响应，且只伤大图样本＝系统性偏差。
    'Shuffled-v2-GraphText': ('Shuffled-v2-GraphDot', 'text'),
    'Shuffled-v2-GraphDot': ('Shuffled-v2-GraphDot', 'dot'),
}
for _lv in ('light', 'medium', 'heavy'):
    GRAPH_DOT_SOURCE[f'DBP-{_lv}-v2-GraphText'] = (f'DBP-{_lv}-v2-GraphDot', 'text')
    GRAPH_DOT_SOURCE[f'DBP-{_lv}-v2-GraphDot'] = (f'DBP-{_lv}-v2-GraphDot', 'dot')


def ti_input_budget(max_model_len):
    return int((max_model_len - TI_MAX_OUTPUT_TOKENS) * TI_SAFETY_RATIO)


def ti_request_max_output(max_model_len):
    """请求实际允许的输出上限，按窗口夹住，保证 输入预算 + 输出上限 ≤ 窗口。

    8/21 复核发现的坑：输入预算是 (W-2000)*0.9，输出却固定按 3000 请求。
    32k / 16k 模型没事（30,691 与 15,945 都小于窗口），但 **Llama-3-8B 的 W=8192**：
    5,572 + 3,000 = 8,572 > 8,192 → vLLM 直接 400
    （"you requested 3000 output tokens and your prompt contains at least 5193 input tokens"），
    整条请求（十几个变量）全部记 -1。实测它 75.2% 的请求就是这么废掉的，
    带图配置 90-95% 全废、code 基线 36%，那一行 Table I 数据完全不可用。
    协议里的 2000 预留不动，只把**请求上限**夹到窗口内（8k → 2,620）。
    """
    return max(TI_MAX_OUTPUT_TOKENS,
               min(TI_REQUEST_MAX_OUTPUT_TOKENS,
                   int(max_model_len) - ti_input_budget(max_model_len)))


_PLACEHOLDER_PAIR = re.compile(r'"(TYPE_\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_type_response(content):
    """把模型返回解析成 {placeholder: type}。

    先按标准 JSON 解析；失败时（最常见的原因是输出被 max_tokens 切断，JSON 少了收尾括号）
    退回到正则逐对提取已经写完整的 `"TYPE_n": "..."`。截断只会伤到最后一两个变量，
    没理由把整条请求（十几个变量）都作废——与漏洞检测侧"保留被截断的输出"同一口径。
    返回 {} 表示确实什么都没解析出来。
    """
    if not isinstance(content, str):
        return content if isinstance(content, dict) else {}
    text = content.strip()
    if text.startswith('```'):      # 偶尔会带 ```json 围栏
        text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    out = {}
    for m in _PLACEHOLDER_PAIR.finditer(text):
        try:
            out[m.group(1)] = json.loads(f'"{m.group(2)}"')
        except json.JSONDecodeError:
            out[m.group(1)] = m.group(2)
    return out


def db_connect(db_path):
    """SQLite 连接（多进程并发写）：等锁而不是直接抛 "database is locked"。
    与漏洞检测侧 utils/mySql.py 同一套设置。"""
    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute('PRAGMA busy_timeout=120000')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn



ALL_GRAPH_REPRESENTATIONS = [
    'GraphText-v2',
    'GraphDot-v2',
    'GraphCode-v2',
    'DBC-v2-GraphText',
    'DBC-v2-GraphDot',
    'DBP-light-v2-GraphText',
    'DBP-light-v2-GraphDot',
    'DBP-medium-v2-GraphText',
    'DBP-medium-v2-GraphDot',
    'DBP-heavy-v2-GraphText',
    'DBP-heavy-v2-GraphDot',
]
# 论文旧数据（仅复现旧库）
LEGACY_GRAPH_REPRESENTATIONS = [
    'GraphText', 'GraphDot', 'GraphCode', 'CDEP',
    'DBP-1-GraphText', 'DBP-1-GraphDot', 'DBP-3-GraphText', 'DBP-3-GraphDot',
    'DBP-5-GraphText', 'DBP-5-GraphDot',
]


def build_experiments(args):
    """构造 (表示格式, 图目录, 图类型) 实验列表。"""
    if not args.run_all:
        graph_format = os.path.basename(os.path.normpath(args.graph_dir))
        return [
            ('Code' if graph_type == 'code' else graph_format, args.graph_dir, graph_type)
            for graph_type in args.graph_types
        ]

    experiments = [('Code', None, 'code')]
    missing_directories = []
    for graph_format in ALL_GRAPH_REPRESENTATIONS:
        graph_dir = os.path.join(args.dataset_dir, graph_format)
        if not os.path.isdir(graph_dir):
            missing_directories.append(graph_dir)
        experiments.extend(
            (graph_format, graph_dir, graph_type)
            for graph_type in ('ast', 'cfg', 'ddg')
        )
    if missing_directories:
        raise FileNotFoundError(
            "--run-all 缺少以下图目录:\n" + "\n".join(missing_directories)
        )
    return experiments

# 共享的Manager实例和资源
_manager = None

def get_shared_manager():
    """获取或创建共享的Manager实例"""
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager

def initialize_api_pool(api_bases, per_api_limit=1):
    """初始化API资源池（在主进程中调用），返回共享对象。

    per_api_limit：每个端点允许的并发请求数。旧版把 base 字符串直接当锁的键，
    同一个端点最多只有 1 个进程在发请求——单端点时整台 GPU 只被一个请求喂着。
    这里把每个端点展开成 `<base>#<i>` 若干个槽位，用时再把 `#i` 去掉。
    """
    manager = get_shared_manager()
    api_bases_lock = manager.Lock()
    slots = [f'{b}#{i}' for b in api_bases for i in range(max(1, per_api_limit))]
    api_bases_pool = manager.list(slots)
    api_bases_in_use = manager.dict()
    with api_bases_lock:
        for base in slots:
            api_bases_in_use[base] = False
    print(f"API资源池初始化完成: {len(slots)} 个槽位 / {len(api_bases)} 个端点")
    return manager, api_bases_lock, api_bases_pool, api_bases_in_use

def strip_slot(base):
    """槽位名 -> 真实 URL"""
    return base.split('#', 1)[0] if base else base


def get_api_base(timeout=600, api_bases_pool=None, api_bases_in_use=None, api_bases_lock=None):
    """从池中获取一个可用的API base（支持传入共享对象）"""
    if api_bases_pool is None or api_bases_lock is None or api_bases_in_use is None:
        raise RuntimeError("API资源池尚未初始化，请先调用initialize_api_pool")
    start_time = time.time()
    pid = os.getpid()
    while time.time() - start_time < timeout:
        with api_bases_lock:
            bases = list(api_bases_pool)
            random.shuffle(bases)
            for base in bases:
                if not api_bases_in_use[base]:
                    api_bases_in_use[base] = True
                    #print(f"进程 {pid} 获取到API base: {base}")
                    return base
        time.sleep(0.5)
    raise TimeoutError("无法获取可用的API base")

def release_api_base(base, api_bases_in_use=None, api_bases_lock=None):
    """释放API base（支持传入共享对象）"""
    if api_bases_in_use is None or api_bases_lock is None:
        return
    pid = os.getpid()
    with api_bases_lock:
        if base in api_bases_in_use:
            api_bases_in_use[base] = False
            #print(f"进程 {pid} 释放API base: {base}")

class TypeInferenceProcessor:
    def __init__(self, api_key: str, model: str, api_base: str = None, graph_dir: str = None, graph_type: str = None, graph_format: str = None, db_path: str = 'type_inference.db', initialize_database: bool = True,
                 temperature: float = 0.2, top_p: float = 1.0, seed: int = -1, max_model_len: int = 32768):
        self.api_key = api_key
        self.max_model_len = max_model_len  # budget-fit：模型窗口
        self._last_graph_info = {'graph_kept_fraction': 1.0, 'graph_tokens': 0}
        self.model = model
        self.api_base = api_base
        self.graph_dir = graph_dir
        self.graph_type = graph_type  # 可以是 'ast', 'cfg', 'ddg', 'tdg' 中的一个，code表示不加图
        self.graph_format = graph_format or ('Code' if graph_type == 'code' else 'legacy')
        self.db_path = db_path
        # E1 多种子协议：采样参数显式化；seed 进结果表主键
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

        if initialize_database:
            self.init_database()
    
    def init_database(self):
        """初始化SQLite数据库"""
        conn = db_connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建类型推断结果表。E1 多种子协议：seed 进主键。
        # 旧表（无 seed 或无 graph_format 列）改名为 type_inference_legacy 保留，不参与新实验。
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='type_inference'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor.execute("PRAGMA table_info(type_inference)")
            columns = {row[1] for row in cursor.fetchall()}
            if 'graph_format' not in columns or 'seed' not in columns:
                cursor.execute('DROP TABLE IF EXISTS type_inference_legacy')
                cursor.execute('ALTER TABLE type_inference RENAME TO type_inference_legacy')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS type_inference (
            dataset_file TEXT,           -- 数据集文件名
            file_path TEXT,              -- 文件路径
            class_name TEXT,             -- 类名（如果有）
            function_name TEXT,          -- 函数/方法名（如果有）
            variable_name TEXT,          -- 变量名
            original_type TEXT,          -- 原始变量类型
            inferred_type TEXT,          -- 推断变量类型
            is_correct INTEGER,          -- 推断是否正确：1严格正确，2最外层正确，0错误，-1未知
            graph_type TEXT,             -- 使用的图类型：ast, cfg, ddg, tdg，或code
            graph_format TEXT,           -- 图表示格式：GraphText, GraphDot, GraphCode, CDEP, DBP等
            seed INTEGER DEFAULT -1,     -- 随机种子（E1 多种子协议；-1 表示未设置）
            is_basic_type INTEGER,       -- 是否是基础类型：1是，0否
            compound_level INTEGER,      -- 复合层数：0基础类型, 1一层复合, 2两层复合, 3+三层以上
            graph_kept_fraction REAL DEFAULT 1.0,  -- budget-fit：图保留节点比例（放得下=1）
            graph_tokens INTEGER DEFAULT 0,        -- 实际放进提示词的图 token 数
            PRIMARY KEY (dataset_file, file_path, class_name, function_name, variable_name, graph_type, graph_format, seed)
        )
        ''')
        cursor.execute("PRAGMA table_info(type_inference)")
        cols = {row[1] for row in cursor.fetchall()}
        if 'graph_kept_fraction' not in cols:
            cursor.execute("ALTER TABLE type_inference ADD COLUMN graph_kept_fraction REAL DEFAULT 1.0")
        if 'graph_tokens' not in cols:
            cursor.execute("ALTER TABLE type_inference ADD COLUMN graph_tokens INTEGER DEFAULT 0")

        conn.commit()
        conn.close()
    
    def get_compound_level(self, type_str: str) -> int:
        """
        计算变量类型的复合层数
        例如:
        - int -> 0 (基础类型)
        - List[int] -> 1 (一层复合)
        - Dict[str, List[int]] -> 2 (两层复合)
        - List[Dict[str, List[int]]] -> 3 (三层复合)
        
        参数:
            type_str: 变量类型字符串
            
        返回:
            复合层数
        """
        # 类型为空的情况
        if not type_str:
            return -1  # 表示无效类型
        
        # 计算方括号的嵌套层数
        level = 0
        max_level = 0
        for char in type_str:
            if char == '[':
                level += 1
                max_level = max(max_level, level)
            elif char == ']':
                level -= 1
        
        # 返回检测到的层级数
        return max_level
    
    def is_basic_type(self, type_str: str) -> bool:
        """
        判断是否为基础类型（如 int, str, bool 等）
        与零层的非基础类型（如自定义类）不同
        
        参数:
            type_str: 变量类型字符串
            
        返回:
            是否为基础类型
        """
        if not type_str:
            return False
            
        basic_types = {
            'builtins.int', 'builtins.float', 'builtins.str', 'builtins.bool', 
            'builtins.bytes', 'builtins.None', 'builtins.NoneType', 'int', 
            'float', 'str', 'bool', 'bytes', 'None', 'Any', 'typing.Any',
            'Tuple[()]', 'tuple', 'builtins.tuple', 'object', 'builtins.object'
        }
        return type_str in basic_types
    
    def count_successful_inferences(self, db_path: str, dataset_file: str, file_path: str, graph_type: Optional[str], graph_format: str, seed: int = -1) -> int:
        """统计数据库中已成功处理的变量数量"""
        conn = db_connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
            SELECT COUNT(*) FROM type_inference
            WHERE dataset_file = ? AND file_path = ? AND graph_type = ?
              AND graph_format = ? AND seed = ? AND inferred_type != "-1"
            ''', (dataset_file, file_path, graph_type, graph_format, seed))
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def count_total_variables(self, data: Dict) -> int:
        """统计文件中的总变量数量"""
        total = 0
        
        def count_vars(d: Dict) -> int:
            nonlocal total
            for k, v in d.items():
                if k == 'variables':
                    total += len(v)
                elif isinstance(v, dict):
                    count_vars(v)
            return total
        
        return count_vars(data)
    
    def replace_types_with_placeholders(self, data: Dict) -> Tuple[Dict, List[Tuple[str, str, str, str, str, str]]]:
        """将变量类型替换为占位符，并记录替换信息"""
        placeholders = []
        placeholder_counter = 0
        
        def replace_type(value: str, var_name: str, context: Dict) -> str:
            nonlocal placeholder_counter
            if isinstance(value, str) and value.strip():
                placeholder = f"TYPE_{placeholder_counter}"
                placeholder_counter += 1
                # 记录变量上下文信息
                class_name = context.get('class_name', None)
                function_name = context.get('function_name', None)
                placeholders.append((placeholder, value, class_name, function_name, var_name, ""))
                return placeholder
            return value
        
        def process_dict(d: Dict, context: Dict = None) -> Dict:
            if context is None:
                context = {}
            
            result = {}
            for k, v in d.items():
                if k == 'variables':
                    result[k] = {var_name: replace_type(var_type, var_name, context) 
                                for var_name, var_type in v.items()}
                elif k == 'class':
                    result[k] = {}
                    for class_name, class_data in v.items():
                        class_context = {**context, 'class_name': class_name}
                        result[k][class_name] = process_dict(class_data, class_context)
                elif k == 'function':
                    result[k] = {}
                    for func_name, func_data in v.items():
                        func_context = {**context, 'function_name': func_name}
                        result[k][func_name] = process_dict(func_data, func_context)
                elif isinstance(v, dict):
                    result[k] = process_dict(v, context)
                else:
                    result[k] = v
            return result
        
        processed_data = process_dict(data)
        return processed_data, placeholders
    
    def get_code_graph(self, dataset_file: str, file_path: str) -> Optional[Dict]:
        """获取文件的代码图数据"""
        if not self.graph_dir or not self.graph_type or self.graph_type == "code":
            return None
            
        # 获取图数据文件路径
        graph_file = os.path.join(self.graph_dir, dataset_file)
        
        try:
            if os.path.exists(graph_file):
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    # 检查图数据中是否包含当前文件的信息
                    if file_path in graph_data:
                        file_graphs = graph_data[file_path]
                        # 返回指定类型的图
                        if self.graph_type in file_graphs:
                            return file_graphs[self.graph_type]
        except Exception as e:
            print(f"Error reading graph data for {file_path}: {str(e)}")
        
        return None
    
    def fit_code_graph(self, dataset_file: str, file_path: str, data: Dict, placeholders, code_graph):
        """把图装进模型窗口预算：放得下原样；放不下读对应 DOT 用 graph_budget_fit 裁剪后再序列化。
        返回 (code_graph_text, info)。无图 / 无 DOT 源（NL、旧表示）时不裁剪（原样返回，仅记 tokens）。"""
        info = {'graph_kept_fraction': 1.0, 'graph_tokens': 0}
        if not code_graph or self.graph_type == 'code':
            return code_graph, info
        graph_tokens = _count_tokens(code_graph)
        info['graph_tokens'] = graph_tokens
        base_prompt = self.create_prompt(file_path, data, placeholders, '')
        budget = ti_input_budget(self.max_model_len) - _count_tokens(TI_SYSTEM_PROMPT) - _count_tokens(base_prompt) - 64
        if graph_tokens <= budget:
            return code_graph, info
        src = GRAPH_DOT_SOURCE.get(str(self.graph_format))
        if src is None or not self.graph_dir:
            return code_graph, info  # 无 DOT 源：交给服务端/后续截断处理
        dot_rep, fmt = src
        dot_dir = os.path.join(os.path.dirname(os.path.normpath(self.graph_dir)), dot_rep)
        try:
            with open(os.path.join(dot_dir, dataset_file), 'r', encoding='utf-8') as f:
                dot_text = json.load(f)[file_path][self.graph_type]
        except Exception as e:
            print(f"budget-fit: 读取 DOT 失败 {dot_rep}/{dataset_file}[{file_path}][{self.graph_type}]: {e}")
            return code_graph, info
        res = fit_graph(dot_text, fmt, max(1, budget), full_text=code_graph)
        info = {'graph_kept_fraction': res.kept_fraction, 'graph_tokens': res.tokens}
        return res.text, info

    def create_prompt(self, file_path: str, data: Dict, placeholders: List[Tuple[str, str, str, str, str, str]], code_graph: str) -> str:
        """Creates a prompt for the language model"""
        # Create detailed context information for placeholders
        placeholder_info = []
        for placeholder, original_type, class_name, function_name, var_name, _ in placeholders:
            context = f"- {placeholder}: variable '{var_name}'"
            if class_name:
                context += f", in class '{class_name}'"
            if function_name:
                context += f", in function '{function_name}'"
            placeholder_info.append(context)

        prompt = f"""Please analyze the variable types in the following Python code. The types in the code have been replaced with placeholders (TYPE_0, TYPE_1, etc.).
Return the Python type for each placeholder in JSON format.

IMPORTANT TYPE INFERENCE GUIDANCE:
1. Correctly identifying the outermost container type is most critical (e.g., List, Dict, Set, Tuple, Union, Optional)
2. For container types, try to infer the inner types when possible (e.g., List[int], Dict[str, float])
3. When uncertain about inner types but confident about the container, use Any for inner types (e.g., List[Any], Dict[str, Any])
4. For commonly used type patterns:
   - Use Optional[T] for variables that might be None
   - Use Union[T1, T2] for variables that could be multiple types
   - Distinguish between Tuple (fixed length with specific types) and List (variable length with consistent type)

Code structure:
{json.dumps(data, indent=2, ensure_ascii=False)}
"""

        # 如果有代码图数据，添加到提示中；E4 NL 消融（graph_format == 'NL'）注入的是
        # 自然语言解释而不是图，标题不得出现 "graph"
        if code_graph:
            if str(self.graph_format).upper() == 'NL':
                aspect = {'cfg': 'control flow', 'ddg': 'data dependencies', 'ast': 'syntactic structure'}
                prompt += f"""
Additional explanation of the code ({aspect.get(self.graph_type, self.graph_type)}):
{code_graph}
"""
            else:
                prompt += f"""
Code graph information ({self.graph_type}):
{code_graph}
"""

        prompt += f"""
Placeholder context information:
{chr(10).join(placeholder_info)}

Please analyze the code and return JSON containing the inferred types for each placeholder in the following format:
{{
    "TYPE_0": "List[Dict[str, int]]",
    "TYPE_1": "Optional[Union[str, int]]",
    "TYPE_2": "Callable[[int, str], bool]",
    ...
}}

Notes:
1. The response must be valid JSON format
2. For types, you can use either simple names (int, str, bool) or imports from typing module (List, Dict, Optional) without full qualification.
3. For class attributes, pay attention to whether it should be a Type[class_name] type
4. When uncertain about exact details but confident about the container type, prioritize getting the container right
5. Make sure to return types for all placeholders"""
        return prompt
    
    def call_llm(self, prompt: str, api_bases_pool=None, api_bases_in_use=None, api_bases_lock=None) -> Dict:
        """Calls the language model API"""
        temp_api_base = None
        
        # 尝试从池获取一个API base
        if not self.api_base:
            try:
                temp_api_base = get_api_base(api_bases_pool=api_bases_pool, api_bases_in_use=api_bases_in_use, api_bases_lock=api_bases_lock)
                api_base = temp_api_base
                #print(f"获取到API base: {api_base}")
            except TimeoutError as e:
                print(f"无法获取API base: {str(e)}")
                raise
        else:
            api_base = self.api_base
        
        try:
            # 在进程中创建client
            client = OpenAI(
                api_key=self.api_key,
                base_url=strip_slot(api_base),
                timeout=1800,
                max_retries=3,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TI_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                seed=(self.seed if self.seed is not None and self.seed >= 0 else None),
                max_tokens=ti_request_max_output(self.max_model_len),
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return content
        finally:
            # 释放API base
            if temp_api_base:
                release_api_base(temp_api_base, api_bases_in_use=api_bases_in_use, api_bases_lock=api_bases_lock)
    
    def extract_outer_type(self, type_str: str) -> str:
        """
        提取类型的最外层
        例如:
        - int -> int
        - List[int] -> List
        - Dict[str, List[int]] -> Dict
        - Union[int, str] -> Union
        - Optional[int] -> Optional
        """
        if not type_str or type_str == "-1":
            return ""
        
        normalized = type_str.strip()
        # 找到第一个方括号的位置
        bracket_pos = normalized.find('[')
        if bracket_pos > 0:
            return normalized[:bracket_pos]
        return normalized

    def is_type_correct(self, original_type: str, inferred_type: str, strict: bool = True) -> int:
        """判断推断的类型是否正确
        
        参数:
            original_type: 原始变量类型
            inferred_type: 推断变量类型
            strict: 是否使用严格模式判断，True表示完全匹配，False表示只匹配最外层类型
        
        返回值:
            1: 正确（完全相同，或在非严格模式下最外层类型相同）
            0: 错误
            -1: 无法判断（任一类型为-1或无效）
        """
        if original_type == "-1" or inferred_type == "-1":
            return -1
            
        # 规范化类型字符串，移除空格和前缀
        def normalize_type(type_str: str) -> str:
            # 移除所有空格
            type_str = re.sub(r'\s+', '', type_str)
            # 移除builtins.前缀
            type_str = re.sub(r'builtins\.', '', type_str)
            # 移除typing.前缀
            type_str = re.sub(r'typing\.', '', type_str)
            return type_str
            
        norm_original = normalize_type(original_type)
        norm_inferred = normalize_type(inferred_type)
        
        # 先进行严格匹配（完全相同）
        if norm_original == norm_inferred:
            return 1  # 严格匹配成功
        
        # 如果严格匹配失败，进行非严格匹配（最外层类型）
        outer_original = self.extract_outer_type(norm_original)
        outer_inferred = self.extract_outer_type(norm_inferred)
        
        # 如果最外层类型相同
        if outer_original == outer_inferred:
            return 2  # 非严格匹配成功
        
        # 处理特殊情况：Optional 和 Union 的兼容性
        if outer_original == "Optional" and outer_inferred == "Union":
            # 检查Union中是否包含None
            if "None" in norm_inferred or "NoneType" in norm_inferred:
                return 2  # 非严格匹配成功
        
        # 处理特殊情况：List 和 Sequence/Iterable 的兼容性
        if outer_original == "List" and outer_inferred in ["Sequence", "Iterable"]:
            return 2  # 非严格匹配成功
        
        # 处理特殊情况：Dict 和 Mapping 的兼容性
        if outer_original == "Dict" and outer_inferred == "Mapping":
            return 2  # 非严格匹配成功
            
        return 0  # 匹配失败
    
    def save_to_database(self, db_path: str, dataset_file: str, file_path: str, placeholders, inferred_types, graph_type: Optional[str], graph_format: str):
        """保存结果到数据库

        E1 多种子协议：每个 (变量, 配置, 种子) 一行，INSERT OR REPLACE。
        旧版按 is_correct 优先级"跨运行保留最好结果"的逻辑已删除——
        它会在多种子实验中抹平方差，与统计目的直接冲突。
        """
        conn = db_connect(db_path)
        cursor = conn.cursor()

        try:
            for placeholder, original_type, class_name, function_name, var_name, is_correct in placeholders:
                inferred_type = inferred_types.get(placeholder, "-1")

                # 判断是否是基础类型
                is_basic = 1 if self.is_basic_type(original_type) else 0

                # 获取复合层级
                compound_level = self.get_compound_level(original_type)

                cursor.execute('''
                INSERT OR REPLACE INTO type_inference
                (dataset_file, file_path, class_name, function_name, variable_name, original_type,
                 inferred_type, is_correct, graph_type, graph_format, seed, is_basic_type, compound_level,
                 graph_kept_fraction, graph_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (dataset_file, file_path, class_name, function_name, var_name, original_type,
                      inferred_type, is_correct, graph_type, graph_format, self.seed, is_basic, compound_level,
                      self._last_graph_info.get('graph_kept_fraction', 1.0), self._last_graph_info.get('graph_tokens', 0)))

            conn.commit()

        except Exception as e:
            print(f"Error saving to database for {file_path}: {str(e)}")
        finally:
            conn.close()
    
    def process_file(self, dataset_file: str, file_path: str, data: Dict, db_path: str, api_bases_pool=None, api_bases_in_use=None, api_bases_lock=None):
        """处理单个文件"""
        # 检查是否已经处理过所有变量
        successful_count = self.count_successful_inferences(
            db_path, dataset_file, file_path, self.graph_type, self.graph_format, self.seed
        )
        total_count = self.count_total_variables(data)
        
        if successful_count == total_count and total_count > 0:
            graph_type_str = self.graph_type if self.graph_type else "无图"
            #print(f"Skipping {file_path} (graph_type={graph_type_str}): all variables already processed successfully")
            return
        
        # 替换类型为占位符
        processed_data, placeholders = self.replace_types_with_placeholders(data)
        if len(placeholders) == 0:
            graph_type_str = self.graph_type if self.graph_type else "无图"
            #print(f"Skipping {file_path} (graph_type={graph_type_str}): no variables to infer")
            return
            
        # 获取代码图数据
        code_graph = self.get_code_graph(dataset_file, file_path)
        if code_graph is None and self.graph_type != "code":
            # 如果没有图数据且图类型不是code，则跳过
            graph_type_str = self.graph_type if self.graph_type else "无图"
            #print(f"Skipping {file_path} (graph_type={graph_type_str}): no code graph data available")
            return
            
        # budget-fit：图放不下窗口时按度数裁剪（协议 2026-08-16），记录 kept_fraction
        code_graph, self._last_graph_info = self.fit_code_graph(dataset_file, file_path, processed_data, placeholders, code_graph)

        # 创建提示
        prompt = self.create_prompt(file_path, processed_data, placeholders, code_graph)
        
        # 每个实验样本只请求一次，避免使用标签驱动的 pass@5 重试。
        max_tries = 1
        inferred_types = None
        result_placeholders = []
        # 初始化一个字典，记录每个placeholder的状态：-1表示未知，0表示错误，1表示严格正确，2表示非严格正确
        placeholder_status = {placeholder: -1 for placeholder, _, _, _, _, _ in placeholders}
        # 初始化一个字典，存储每个placeholder的最佳推断结果
        best_inferences = {placeholder: "-1" for placeholder, _, _, _, _, _ in placeholders}
        # 初始化一个集合，记录已经推断正确的placeholder
        correct_placeholders = set()
        
        for attempt in range(max_tries):
            try:
                #print(f"尝试 {attempt+1}/{max_tries} 推断 {file_path} (graph_type={self.graph_type if self.graph_type else '无图'})")
                inferred_types_response = self.call_llm(prompt, api_bases_pool=api_bases_pool, api_bases_in_use=api_bases_in_use, api_bases_lock=api_bases_lock)
                
                # 解析返回的JSON字符串（截断的 JSON 也尽量抢救出已写完的变量）
                try:
                    inferred_types = parse_type_response(inferred_types_response)

                    # 检查结果是否有效
                    if not isinstance(inferred_types, dict) or len(inferred_types) == 0:
                        print(f"收到无效的推断结果：{str(inferred_types_response)[:200]}")
                        continue
                        
                    # 检查每个变量的推断结果是否正确
                    all_correct = True
                    for placeholder, original_type, class_name, function_name, var_name, _ in placeholders:
                        # 如果该placeholder已经有正确的推断，则跳过
                        if placeholder in correct_placeholders:
                            continue
                            
                        inferred_type = inferred_types.get(placeholder, "-1")
                        if inferred_type != "-1":
                            is_correct = self.is_type_correct(original_type, inferred_type)
                            
                            # 更新状态：1表示严格正确，2表示非严格正确
                            if is_correct > 0:  # 正确的推断(1或2)
                                # 优先保存更好的结果（严格正确优于非严格正确）
                                if placeholder_status[placeholder] < is_correct:
                                    placeholder_status[placeholder] = is_correct
                                    best_inferences[placeholder] = inferred_type
                                # 只有是严格正确时才加入到correct_placeholders集合
                                if is_correct == 1:
                                    correct_placeholders.add(placeholder)
                            elif placeholder_status[placeholder] <= 0:  # 如果还没有任何正确的推断
                                placeholder_status[placeholder] = 0
                                # 只在还没有任何推断时更新
                                if best_inferences[placeholder] == "-1":
                                    best_inferences[placeholder] = inferred_type
                        
                        # 判断是否所有变量都已经有严格正确的推断
                        if placeholder not in correct_placeholders:
                            all_correct = False
                    
                    # print(f"尝试 {attempt+1}/{max_tries} - 正确推断数量: {len(correct_placeholders)}/{len(placeholders)}")
                    
                    # 如果所有变量都已经有严格正确的推断，则提前结束
                    if all_correct:
                        print(f"所有变量推断正确，提前结束尝试 {attempt+1}/{max_tries}")
                        break
                        
                except json.JSONDecodeError:
                    print(f"无法解析推断结果")
                    print(inferred_types_response)
                    continue
                    
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_tries} 失败：{str(e)}")
                time.sleep(1)  # 短暂延迟后重试
                continue
        
        # 准备最终结果
        for placeholder, original_type, class_name, function_name, var_name, _ in placeholders:
            is_correct = placeholder_status[placeholder]
            result_placeholders.append((placeholder, original_type, class_name, function_name, var_name, is_correct))
        
        # 保存最终结果到数据库
        self.save_to_database(
            db_path, dataset_file, file_path, result_placeholders,
            best_inferences, self.graph_type, self.graph_format
        )

# 适用于多进程的处理函数
def process_file_worker(args, api_key, model, api_base, db_path, graph_dir, graph_type, graph_format=None, api_bases_pool=None, api_bases_in_use=None, api_bases_lock=None):
    """处理单个文件的工作函数"""
    dataset_file, file_path, file_data = args
    pid = os.getpid()
    #print(f"进程 {pid} 开始处理文件: {file_path}")
    
    # 在每个工作进程中创建处理器实例
    processor = TypeInferenceProcessor(
        api_key=api_key,
        model=model,
        api_base=api_base,
        graph_dir=graph_dir,
        graph_type=graph_type,
        graph_format=graph_format,
        db_path=db_path
    )
    
    try:
        processor.process_file(dataset_file, file_path, file_data, db_path, api_bases_pool=api_bases_pool, api_bases_in_use=api_bases_in_use, api_bases_lock=api_bases_lock)
        return file_path, True
    except Exception as e:
        graph_type_str = graph_type if graph_type else "无图"
        print(f"\n进程 {pid} 处理文件出错 {file_path} {str(e)}")
        return file_path, False

def parse_args():
    parser = argparse.ArgumentParser(description='Python类型推断工具')
    # 定义默认api_bases，使其可被重用
    default_api_bases = [f'http://127.0.0.1:{port}/v1' for port in range(8010, 8013)]
    parser.add_argument('--api-key', default='EMPTY', help='OpenAI API密钥')
    parser.add_argument('--api-bases', nargs='+', default=default_api_bases, help='OpenAI API基础URL列表（可选）')
    parser.add_argument('--model', default='/mnt/gold/lbt/LLM_model/Qwen2.5_7B_Instruct', help='使用的模型名称')
    parser.add_argument('--db-path', default=os.path.join(SCRIPT_DIR, 'results', 'type_inference.db'), help='SQLite数据库文件路径')
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR, help='数据集根目录')
    parser.add_argument('--input-dir', default=None, help='输入数据目录；默认是 <dataset-dir>/processed_projects_cleaned')
    parser.add_argument('--graph-dir', default=None, help='单实验代码图目录；默认是 <dataset-dir>/GraphText')
    parser.add_argument('--selected-files-path', default=None, help='待处理文件清单；默认是 <dataset-dir>/selected_files_new.json')
    parser.add_argument('--graph-types', nargs='+', choices=['ast', 'cfg', 'ddg', 'tdg', 'code'], default=['ast', 'cfg', 'ddg', 'code'], help='要测试的图类型列表，例如: --graph-types ast cfg dfg tdg')
    parser.add_argument('--run-all', action='store_true', help='运行代码基线及全部图表示实验；覆盖 --graph-dir 和 --graph-types')
    parser.add_argument('--per-api-limit', type=int, default=8,
                        help='每个 API 端点允许的并发请求数（决定实际并发；旧版固定为 1）')
    parser.add_argument('--processes', type=int, default=len(default_api_bases), help=f'并行处理的进程数（默认：{len(default_api_bases)}，与API基础URL数量相同）')
    # E1 多种子协议
    parser.add_argument('--temperature', type=float, default=0.2, help='采样温度（E1 协议默认 0.2）')
    parser.add_argument('--top-p', type=float, default=1.0, help='top_p（沿用原实验设置）')
    parser.add_argument('--max-model-len', type=int, default=32768,
                        help='模型上下文窗口（budget-fit：图按 (窗口-2000)*0.9 的输入预算裁剪；Qwen2.5 系 32768）')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='随机种子列表；每个种子完整跑一遍全部实验配置')
    return parser.parse_args()

def main():
    # 添加多进程支持
    freeze_support()
    
    args = parse_args()
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    if args.input_dir is None:
        args.input_dir = os.path.join(args.dataset_dir, 'processed_projects_cleaned')
    if args.graph_dir is None:
        args.graph_dir = os.path.join(args.dataset_dir, 'GraphText')
    if args.selected_files_path is None:
        args.selected_files_path = os.path.join(args.dataset_dir, 'selected_files_new.json')
    args.db_path = os.path.abspath(args.db_path)
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    # token 计数用被测模型自己的分词器（--model 是权重目录时自动启用；子进程 fork 继承环境变量）。
    # tiktoken 对代码普遍低估，会让 budget-fit 算出的预算偏大、请求超窗被 vLLM 400 拒绝。
    if not os.environ.get('SEQGRAPH_TOKENIZER') and os.path.isdir(args.model):
        os.environ['SEQGRAPH_TOKENIZER'] = args.model
    print(f"token 计数分词器: {os.environ.get('SEQGRAPH_TOKENIZER') or 'tiktoken:cl100k_base'}")
    args.experiments = build_experiments(args)
    print(f"实验配置数: {len(args.experiments)}")
    for graph_format, graph_dir, graph_type in args.experiments:
        print(f"  {graph_format}/{graph_type}: {graph_dir or 'code only'}")

    first_graph_format, first_graph_dir, first_graph_type = args.experiments[0]
    TypeInferenceProcessor(
        api_key=args.api_key,
        model=args.model,
        graph_dir=first_graph_dir,
        graph_type=first_graph_type,
        graph_format=first_graph_format,
        db_path=args.db_path,
    )
    
    # 初始化共享的API资源池
    manager, api_bases_lock, api_bases_pool, api_bases_in_use = initialize_api_pool(
        args.api_bases, per_api_limit=args.per_api_limit)
    
    # 获取所有需要处理的文件 - 从selected_files.json读取
    all_files = []
    try:
        selected_files_path = args.selected_files_path
        with open(selected_files_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            selected_files = data.get('selected_files', [])
            print(f"从{selected_files_path}中读取了{len(selected_files)}个文件")
            processed_count = 0
            
            # 从selected_files.json中提取文件信息
            for file_info in selected_files:      
                repo_name = file_info.get('repo_name', '')
                file_path = file_info.get('file_path', '')
                
                if not repo_name or not file_path:
                    continue
                
                dataset_file = f"{repo_name}.json"
                
                # 如果该文件已存在于input_dir中
                full_dataset_path = os.path.join(args.input_dir, dataset_file)
                if os.path.exists(full_dataset_path):
                    try:
                        with open(full_dataset_path, 'r', encoding='utf-8') as dataset_file:
                            dataset_data = json.load(dataset_file)
                            if file_path in dataset_data:
                                file_data = dataset_data[file_path]
                                all_files.append((repo_name + ".json", file_path, file_data))
                                processed_count += 1
                            else:
                                print(f"警告: 数据集文件 {full_dataset_path} 中找不到路径 {file_path}")
                    except Exception as e:
                        print(f"读取数据集文件出错 {full_dataset_path}: {str(e)}")
                else:
                    print(f"警告: 找不到数据集文件 {full_dataset_path}")
                    
            print(f"成功加载 {len(all_files)} 个文件进行处理")
    except Exception as e:
        print(f"读取selected_files.json出错: {str(e)}")
        print("将回退到扫描目录模式...")
        
        # 回退到原来的方式：扫描目录
        for filename in os.listdir(args.input_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(args.input_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for file_path, file_data in data.items():
                            all_files.append((filename, file_path, file_data))
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
    
    # 改为按文件处理：处理每个文件的所有图类型
    run_inference_by_file(all_files, args, api_bases_pool, api_bases_in_use, api_bases_lock)

def run_inference_by_file(all_files, args, api_bases_pool, api_bases_in_use, api_bases_lock):
    """按文件处理所有图类型"""
    db_path = args.db_path
    
    # 设置进程池
    pool = multiprocessing.Pool(processes=args.processes)
    
    # 使用进程池并行处理文件
    with tqdm(total=len(all_files), desc="Processing files") as pbar:
        for result in pool.imap_unordered(partial(process_file_by_graph_types, 
                                                  args=args, 
                                                  db_path=db_path, 
                                                  api_bases_pool=api_bases_pool, 
                                                  api_bases_in_use=api_bases_in_use, 
                                                  api_bases_lock=api_bases_lock), all_files):
            file_path, success = result
            pbar.update(1)
    
    # 关闭进程池
    pool.close()
    pool.join()

def process_file_by_graph_types(file_info, args, db_path, api_bases_pool, api_bases_in_use, api_bases_lock):
    """处理单个文件的所有图类型"""
    dataset_file, file_path, file_data = file_info
    pid = os.getpid()
    #print(f"进程 {pid} 开始处理文件: {file_path}")
    
    try:
        # 配置在外层、种子在内层：同一 (文件, 配置) 的 5 个种子提示词逐字节相同，
        # 连着发给**同一个端点**才能命中 vLLM 的 prefix cache（prefill 只算一次）。
        # 反过来（种子在外层）两次相同提示词之间隔着 34 个配置，缓存必被挤掉——
        # 实测命中率只有 2%，而 TI 是 prefill 主导的。整组种子共用一个端点槽位。
        def run_seed(seed, graph_format, graph_dir, graph_type, slot):
            TypeInferenceProcessor(
                api_key=args.api_key,
                model=args.model,
                api_base=slot,
                graph_dir=graph_dir,
                graph_type=graph_type,
                graph_format=graph_format,
                db_path=db_path,
                initialize_database=False,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
                max_model_len=args.max_model_len,
            ).process_file(dataset_file, file_path, file_data, db_path,
                           api_bases_pool=api_bases_pool,
                           api_bases_in_use=api_bases_in_use,
                           api_bases_lock=api_bases_lock)

        for graph_format, graph_dir, graph_type in args.experiments:
            slot = get_api_base(api_bases_pool=api_bases_pool,
                                api_bases_in_use=api_bases_in_use,
                                api_bases_lock=api_bases_lock)
            try:
                # 先热后并发（与漏洞检测侧同一招）：第 1 个种子单独发出去，把这段提示词的 KV
                # 写进 prefix cache；其余种子再并发发向**同一个端点**，共用那份 KV。
                # 光靠"顺序发同一提示词"是不够的：KV 只有 9 万 token，别的请求会在两次之间把它挤掉
                # （实测命中率仅 2%→9%）；并发发时这些块被活跃请求钉住，prefill 真的只算一次。
                seeds = list(args.seeds)
                if seeds:
                    run_seed(seeds[0], graph_format, graph_dir, graph_type, slot)
                    rest = seeds[1:]
                    if rest:
                        with ThreadPoolExecutor(max_workers=len(rest)) as ex:
                            list(ex.map(lambda s: run_seed(s, graph_format, graph_dir,
                                                           graph_type, slot), rest))
            finally:
                release_api_base(slot, api_bases_in_use=api_bases_in_use,
                                 api_bases_lock=api_bases_lock)
        return file_path, True
    except Exception as e:
        print(f"\n进程 {pid} 处理文件出错 {file_path}: {str(e)}")
        return file_path, False

if __name__ == '__main__':
    main()
