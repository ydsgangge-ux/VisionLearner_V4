# sentence_parser.py - 语言学驱动的句法解析器
"""
基于语言学句法结构的意图理解模块。
不依赖LLM，纯规则+分词。

核心思路：
  任何语言的疑问句都包含两件事：
    1. 查询对象（主语/宾语的中心语）= 用户想了解的是什么
    2. 查询维度（谓语核心词）       = 用户想了解哪个方面

解析流程：
  语言检测 → 句型判定 → 去除虚词层 →
  提取中心语(object) + 维度词(dimension) → 归一化
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParseResult:
    """句法解析结果"""
    object: str        # 查询对象（中心语），如 "蠢"、"光合作用"、"list"
    dimension: str     # 查询维度，归一化后，如 "reading"、"meaning"、""
    dimension_raw: str # 维度原始词，如 "读音"、"pronounce"
    intent_type: str   # query / learn / quiz / progress / general
    lang: str          # zh / en / mixed
    sentence_type: str # interrogative / imperative / declarative
    search_query: str  # 最终检索词 = object + dimension_raw（供RAG使用）
    raw: str           # 原始输入


# ═══════════════════════════════════════════════════════════════
# 语言检测
# ═══════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    检测文本语言。
    返回：'zh' / 'en' / 'mixed'
    """
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_chars  = len(re.findall(r'[a-zA-Z]', text))
    total = zh_chars + en_chars
    if total == 0:
        return 'zh'
    zh_ratio = zh_chars / total
    if zh_ratio > 0.7:
        return 'zh'
    if zh_ratio < 0.3:
        return 'en'
    return 'mixed'


# ═══════════════════════════════════════════════════════════════
# 句型判定
# ═══════════════════════════════════════════════════════════════

# 中文疑问词
ZH_INTERROGATIVE = re.compile(
    r'什么|怎么|怎样|如何|哪|几|多少|为什么|为何|是否|吗|呢|？|\?'
)

# 中文祈使句触发词（句首动词）
ZH_IMPERATIVE = re.compile(
    r'^(告诉我|帮我|帮助我|解释|介绍|说说|讲讲|列出|给我|请问|请)'
)

# 中文陈述式问句（"我想知道/了解"）
ZH_DECLARATIVE_QUERY = re.compile(
    r'我想(知道|了解|学|学习)|我要(知道|了解)'
)

# 英文疑问词
EN_INTERROGATIVE = re.compile(
    r'\b(what|how|why|which|where|when|who|whose|whom|is|are|do|does|did|can|could|would)\b',
    re.IGNORECASE
)

# 英文祈使句触发词（句首动词）
EN_IMPERATIVE = re.compile(
    r'^(tell me|explain|describe|define|show me|help me|give me|list|please)',
    re.IGNORECASE
)


def detect_sentence_type(text: str, lang: str) -> str:
    """
    判断句型：interrogative（疑问）/ imperative（祈使）/ declarative（陈述）
    """
    t = text.strip()
    if lang in ('zh', 'mixed'):
        if ZH_IMPERATIVE.search(t):
            return 'imperative'
        if ZH_DECLARATIVE_QUERY.search(t):
            return 'declarative'
        if ZH_INTERROGATIVE.search(t):
            return 'interrogative'
    if lang in ('en', 'mixed'):
        if EN_IMPERATIVE.search(t):
            return 'imperative'
        if EN_INTERROGATIVE.search(t):
            return 'interrogative'
    return 'declarative'


# ═══════════════════════════════════════════════════════════════
# 意图类型判定（独立于语言）
# ═══════════════════════════════════════════════════════════════

INTENT_PATTERNS = [
    ('learn',    re.compile(r'我想学|开始学|学习一下|帮我学|learn|study', re.I)),
    ('quiz',     re.compile(r'测试我|考考我|出题|测验|检验|quiz me|test me', re.I)),
    ('progress', re.compile(r'进度|学了多少|完成了多少|还差多少|progress|how much', re.I)),
]

def detect_intent_type(text: str) -> str:
    for intent, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return 'query'


# ═══════════════════════════════════════════════════════════════
# 维度词表（谓语核心词 → 归一化维度）
# 规律：动词/疑问短语 揭示"想知道什么方面"
# ═══════════════════════════════════════════════════════════════

# 格式：(归一化维度, [中文触发词], [英文触发词])
DIMENSION_TABLE = [
    ('reading',  ['怎么读','读音','拼音','发音','怎么念','念什么','读什么','读法'],
                 ['pronounce','pronunciation','read','reading','sound']),

    ('strokes',  ['几画','笔画','笔顺','怎么写','字形','结构','部首','偏旁'],
                 ['stroke','strokes','radical','write','writing','structure']),

    ('meaning',  ['什么意思','是什么意思','含义','意思','意义','解释','表示什么',
                  '代表什么','指什么','意味着'],
                 ['mean','means','meaning','definition','define','definition of',
                  'what is','what does','signify']),

    ('usage',    ['怎么用','如何用','用法','组词','造句','例句','搭配','使用'],
                 ['use','usage','example','sentence','how to use','collocate']),

    ('memory',   ['怎么记','如何记','记忆','记法','联想','助记'],
                 ['memorize','remember','mnemonic','memory trick']),

    ('compare',  ['区别','不同','对比','比较','有什么差','差异','辨析'],
                 ['difference','compare','versus','vs','distinguish']),

    ('related',  ['形近字','同音字','近义词','反义词','相关','同义'],
                 ['similar','synonym','antonym','related','homophone']),

    ('origin',   ['来历','由来','历史','词源','起源','怎么来的'],
                 ['origin','etymology','history','come from']),

    ('principle', ['原理','原则','机制','怎么工作','如何工作','为什么'],
                  ['principle','mechanism','how does','why does','theory']),
]

def normalize_dimension(text: str, lang: str) -> Tuple[str, str]:
    """
    从文本中提取并归一化维度词。
    返回：(归一化维度, 原始维度词)
    例：('reading', '读音')
    """
    text_lower = text.lower()

    # 优先匹配更长的词（按长度降序），避免"什么意思"被"什么"截断
    if lang in ('zh', 'mixed'):
        candidates = []
        for dim, zh_words, _ in DIMENSION_TABLE:
            for w in zh_words:
                if w in text:
                    candidates.append((len(w), dim, w))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1], candidates[0][2]

    if lang in ('en', 'mixed'):
        candidates = []
        for dim, _, en_words in DIMENSION_TABLE:
            for w in en_words:
                if re.search(r'\b' + re.escape(w) + r'\b', text_lower):
                    candidates.append((len(w), dim, w))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1], candidates[0][2]

    return '', ''


# ═══════════════════════════════════════════════════════════════
# 虚词过滤层
# 规律：疑问词本身、虚主语、语气词 不是查询对象
# ═══════════════════════════════════════════════════════════════

# 中文虚词集合
ZH_STOP_WORDS = set(
    '的了吗呢啊哦嗯嘛哈吧呀我你他她它们'
    '这那是有在什么怎么如何几多为什么为何'
    '哪里哪个请帮告诉介绍解释说讲给列出'
    '想知道想了解想学我想'
)

# 中文虚主语（出现在句首，忽略）
ZH_DUMMY_SUBJECTS = {'我', '你', '他', '她', '它', '我们', '你们', '他们'}

# 英文停用词（疑问词 + 虚主语 + 助动词）
EN_STOP_WORDS = set(
    'i you he she it we they me him her us them '
    'what how why which where when who whose whom '
    'is are was were be been being '
    'do does did can could would should may might must '
    'the a an of in on at to for with by from '
    'tell explain describe define show help give list please'
    .split()
)

def _strip_zh_noise(text: str) -> str:
    """去除中文句子的问句噪声，返回干净的名词性内容"""
    # 去句尾语气词
    text = re.sub(r'[吗呢啊哦嗯嘛哈吧呀？\?！!]+$', '', text).strip()
    # 去句首祈使前缀
    text = ZH_IMPERATIVE.sub('', text).strip()
    # 去陈述式前缀
    text = ZH_DECLARATIVE_QUERY.sub('', text).strip()
    return text

def _strip_en_noise(text: str) -> str:
    """去除英文句子的问句噪声"""
    text = EN_IMPERATIVE.sub('', text, count=1).strip()
    # 去 "I want to know" 类前缀
    text = re.sub(r'^(i want to (know|understand|learn)|can you (tell me|explain))\s*',
                  '', text, flags=re.IGNORECASE).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# 中心语提取
# 规律：最长的名词性短语 = 查询对象
#       修饰语(的/of) 揭示维度
# ═══════════════════════════════════════════════════════════════

def extract_object_zh(text: str, dimension_raw: str) -> str:
    """
    中文中心语提取。
    策略：
      1. 引号内容优先（明确标注的术语）
      2. "X的Y" 结构：X是对象，Y是维度
      3. "X字/X词" 结构
      4. 去掉维度词、疑问词、语气词后，取剩余最长名词性片段
    """
    # 1. 引号优先
    quoted = re.findall(r'[「」""\'\'「」](.{1,20})[「」""\'\'「」]', text)
    if quoted:
        return quoted[0].strip()

    # 2. "X的Y" 结构，且Y是维度词
    if dimension_raw:
        m = re.search(r'([\u4e00-\u9fff]{1,10})[的]' + re.escape(dimension_raw), text)
        if m:
            candidate = m.group(1).strip()
            if candidate not in ZH_STOP_WORDS and len(candidate) >= 1:
                return candidate

    # 3. 比较句特殊处理："X和Y有什么区别" → 对象=X（第一个被比较项）
    if dimension_raw in ('区别', '不同', '对比', '比较', '差异', '辨析'):
        m = re.search(r'^([\u4e00-\u9fff\w]{1,10})[和与跟]', text)
        if m:
            return m.group(1)

    # 4. "X字" / "X词" 结构（如"蠢字怎么读" → 蠢）
    m = re.search(r'([\u4e00-\u9fff]{1,4})[字词]', text)
    if m:
        return m.group(1)

    # 4. 去掉维度词和疑问噪声词，提取剩余名词性内容
    clean = text
    if dimension_raw:
        clean = clean.replace(dimension_raw, '')

    # 去疑问词和语气词，以及方位介词（里/中/内/上/下/前/后）
    noise = re.compile(r'怎么|如何|什么|是否|为什么|为何|几|多少|[吗呢啊哦嗯嘛哈吧呀？\?里中内]')
    clean = noise.sub(' ', clean).strip()

    # 去句首祈使词
    clean = ZH_IMPERATIVE.sub('', clean).strip()

    # ── 去掉末尾的虚词（是/的/了/有/在/会/能/被/于）──
    # 例："认知革命是" → "认知革命"，"全球化的影响" → "全球化"
    # 先去掉词尾常见动词/助词结尾
    clean = re.sub(r'[是的了有在会能被于着过呢吧啊]+$', '', clean).strip()
    # 去掉"XX的YY"中"的YY"这类修饰尾巴（YY是常见泛化词）
    clean = re.sub(r'的(影响|原因|作用|目的|意义|结果|方式|方法|机制|过程|定义|概念|本质|逻辑|历史|现状|发展|变化|问题|挑战|趋势|特点|特征|优势|劣势|区别|联系)+$', '', clean).strip()

    # 提取连续汉字片段，过滤停用词（含单字介词/方位词）
    func = set('的了我你他她它们这那是有在和与及或里中上下前后左右内外到从被')
    segments = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z\d]{2,}', clean)
    candidates = [s for s in segments if s not in func]

    # 优先返回最长的（多字词/专有名词）
    if candidates:
        return max(candidates, key=len)

    # 兜底：单个汉字（如"蠢"单独出现）
    single = re.findall(r'[\u4e00-\u9fff]', clean)
    single = [c for c in single if c not in func]
    if single:
        return single[0]

    return ''


def extract_object_en(text: str, dimension_raw: str) -> str:
    """
    英文中心语提取。
    策略：
      1. 引号内容优先
      2. "X of Y" / "the Y of X" → X是对象
      3. 去掉维度词和停用词，取最长名词性片段
    """
    # 1. 引号优先（含中文词汇）
    quoted = re.findall(r'["\'「」](.{1,30})["\'「」]', text)
    if quoted:
        return quoted[0].strip()

    # 2. 中文字符（混合输入时，汉字/汉词往往是查询对象）
    zh_words = re.findall(r'[\u4e00-\u9fff]+', text)
    if zh_words:
        return max(zh_words, key=len)

    # 3. "X of Y" → Y是对象（如 "meaning of photosynthesis"）
    m = re.search(r'\bof\s+([\w\s]{2,30}?)(?:\s*[,?.]|$)', text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip('?.,')
        words = [w for w in candidate.split() if w.lower() not in EN_STOP_WORDS]
        if words:
            return ' '.join(words)

    # 4. 去停用词（扩展：加上 about/around/regarding 等介词）
    extended_stops = EN_STOP_WORDS | {'about', 'around', 'regarding', 'concerning',
                                       'into', 'onto', 'upon', 'within'}
    if dimension_raw:
        text = re.sub(r'\b' + re.escape(dimension_raw) + r'\b', '', text, flags=re.IGNORECASE)

    words = text.split()
    noun_chunks = []
    current_chunk = []
    for w in words:
        clean_w = w.strip('?.,!\'\"')
        if clean_w.lower() in extended_stops:
            if current_chunk:
                noun_chunks.append(' '.join(current_chunk))
                current_chunk = []
        else:
            if clean_w:
                current_chunk.append(clean_w)
    if current_chunk:
        noun_chunks.append(' '.join(current_chunk))

    if noun_chunks:
        return max(noun_chunks, key=len).strip('?.,')

    return ''


# ═══════════════════════════════════════════════════════════════
# 主解析器
# ═══════════════════════════════════════════════════════════════

class SentenceParser:
    """
    语言学驱动的句法解析器。

    不依赖LLM，纯规则处理。
    支持中文、英文、中英混合输入。

    用法：
        parser = SentenceParser()
        result = parser.parse("光合作用的原理是什么")
        # result.object = "光合作用"
        # result.dimension = "principle"
        # result.search_query = "光合作用 原理"
    """

    def __init__(self, current_topic: str = None):
        self.current_topic = current_topic  # 用于指代消解

    def set_topic(self, topic: str):
        self.current_topic = topic

    def parse(self, text: str) -> ParseResult:
        """
        完整解析一句话。
        返回 ParseResult。
        """
        raw = text.strip()

        # ── 第一步：指代消解 ──
        text = self._resolve_reference(raw)

        # ── 第二步：语言检测 ──
        lang = detect_language(text)

        # ── 第三步：意图类型 ──
        intent_type = detect_intent_type(text)

        # ── 第四步：句型判定 ──
        sentence_type = detect_sentence_type(text, lang)

        # ── 第五步：去噪（去疑问/祈使前缀）──
        if lang == 'zh':
            clean = _strip_zh_noise(text)
        elif lang == 'en':
            clean = _strip_en_noise(text)
        else:  # mixed：先去英文噪声，再去中文噪声
            # 特别处理 "英文词+里/中/内" 的上下文限定结构，如 "Python里"
            clean = re.sub(r'[a-zA-Z\d]+[里中内的]', ' ', text)
            clean = _strip_zh_noise(clean)
            clean = _strip_en_noise(clean)

        # ── 第六步：提取维度词 ──
        dimension, dimension_raw = normalize_dimension(text, lang)

        # ── 第七步：提取查询对象（中心语）──
        if lang == 'zh':
            obj = extract_object_zh(clean, dimension_raw)
        elif lang == 'en':
            obj = extract_object_en(clean, dimension_raw)
        else:  # mixed：中文2字以上片段优先（核心概念），英文往往是上下文
            # 先找中文2字以上的名词（如"列表"、"光合作用"）
            zh_obj = extract_object_zh(clean, dimension_raw)
            # 再找纯英文对象（如 photosynthesis）
            en_obj = extract_object_en(clean, dimension_raw)
            # 规则：有中文2字以上词 → 用中文；否则用英文
            if zh_obj and len(zh_obj) >= 2 and re.search(r'[\u4e00-\u9fff]', zh_obj):
                obj = zh_obj
            else:
                obj = en_obj or zh_obj

        # ── 第八步：合成检索词 ──
        # 方式B：object + dimension_raw 组合
        # 非查询意图（quiz/learn/progress）不需要object
        if intent_type != 'query':
            obj = ''
            search_query = raw
        elif obj and dimension_raw:
            search_query = f"{obj} {dimension_raw}"
        elif obj:
            search_query = obj
        else:
            search_query = raw  # 完全兜底：用原句

        return ParseResult(
            object=obj,
            dimension=dimension,
            dimension_raw=dimension_raw,
            intent_type=intent_type,
            lang=lang,
            sentence_type=sentence_type,
            search_query=search_query,
            raw=raw,
        )

    # ─── 指代消解 ───────────────────────────────────────────────

    ZH_PRONOUNS = {
        '它','这个','那个','这字','那字','这个字','那个字',
        '此字','该字','刚才','刚刚','上面','前面',
        '这词','这个词','那个词','这','那'
    }
    EN_PRONOUNS = {'it', 'this', 'that', 'this word', 'that word',
                   'this character', 'that character'}

    def _resolve_reference(self, text: str) -> str:
        if not self.current_topic:
            return text
        result = text
        # 中文代词（按长度降序，避免短词截断长词）
        for p in sorted(self.ZH_PRONOUNS, key=len, reverse=True):
            if p in result:
                result = result.replace(p, self.current_topic)
                break  # 一次只替换最长匹配
        # 英文代词（独立单词）
        for p in sorted(self.EN_PRONOUNS, key=len, reverse=True):
            result = re.sub(r'\b' + re.escape(p) + r'\b',
                            self.current_topic, result, flags=re.IGNORECASE)
        # 消解后清理：去掉 "蠢有几画" 里topic后面紧跟的系动词
        # 如 "蠢有几画" → "蠢几画"（"有"在这里是虚词）
        if self.current_topic:
            result = re.sub(
                re.escape(self.current_topic) + r'(有|是|被|为)',
                self.current_topic + ' ',
                result
            )
        return result


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = SentenceParser()

    test_cases = [
        # (输入, 期望object, 期望dimension, 当前话题)
        ("蠢字怎么读",              "蠢",      "reading",   None),
        ("它有几画",                "蠢",      "strokes",   "蠢"),
        ("这个字的部首是什么",       "蠢",      "strokes",   "蠢"),
        ("光合作用的原理是什么",     "光合作用", "principle", None),
        ("Python里列表怎么用",       "列表",    "usage",     None),
        ("蠢和舂有什么区别",         "蠢",      "compare",   None),
        ("How do you pronounce 蠢",  "蠢",      "reading",   None),
        ("What is the meaning of 光合作用", "光合作用", "meaning", None),
        ("tell me about photosynthesis",    "photosynthesis", "",  None),
        ("this word, how to use it",        "蠢",    "usage",     "蠢"),
        ("我想知道光合作用的原理",    "光合作用", "principle", None),
        ("测试我",                   "",        "",          None),
    ]

    print(f"\n{'输入':<35} {'对象':<15} {'维度':<12} {'检索词'}")
    print("─" * 80)
    for text, exp_obj, exp_dim, topic in test_cases:
        parser.set_topic(topic)
        r = parser.parse(text)
        obj_ok  = "OK" if r.object == exp_obj  else f"FAIL({exp_obj})"
        dim_ok  = "OK" if r.dimension == exp_dim else f"FAIL({exp_dim})"
        print(f"{text:<35} {r.object:<8}{obj_ok:<8} {r.dimension:<6}{dim_ok:<8} {r.search_query}")

    print("\nOK - 自测完成")
