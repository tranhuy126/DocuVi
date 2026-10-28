import re
import unicodedata


def clean_and_normalize_text(text: str) -> str:
    """Chuẩn hóa ký tự Unicode, ngoặc kép, gạch ngang, khoảng trắng và ký tự điều khiển ẩn."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Loại bỏ soft hyphen và zero-width space
    text = text.replace("\xad", "").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ")
    # Bảo vệ dấu gạch nối trong khoảng số (10-15 thay vì 10 - 15)
    text = re.sub(r"(\d+)\s*[–—]\s*(\d+)", r"\1-\2", text)
    text = text.replace("–", " - ")
    text = text.replace("\xa0", " ").replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def join_spans_into_line(spans: list[dict]) -> str:
    """Ghép các spans trong 1 dòng với khoảng cách khoảng trắng chính xác."""
    spans = sorted(spans, key=lambda s: s["bbox"][0])
    result = ""
    previous_span = None
    trailing_punctuation = ",.;:!?%)]}»’”\"'"
    leading_punctuation = "([{«“‘-/"

    for span in spans:
        text = span["text"]
        if not text:
            continue

        if previous_span is not None:
            prev_x1 = previous_span["bbox"][2]
            curr_x0 = span["bbox"][0]
            gap = curr_x0 - prev_x1
            font_size = max(previous_span["size"], span["size"])

            # Thêm khoảng trắng giữa 2 span nếu khoảng cách đủ lớn, không liền kề dấu câu trước hoặc sau
            if (
                gap > font_size * 0.15
                and result
                and not result.endswith((" ", "\t"))
                and not text.startswith((" ", "\t"))
                and text[0] not in trailing_punctuation
                and result[-1] not in leading_punctuation
            ):
                result += " "

        result += text
        previous_span = span

    return result.strip()


HYPHEN_PREFIXES = {
    "self", "cross", "state", "decision", "well", "user", "real",
    "multi", "non", "all", "high", "low", "long", "short", "large",
    "small", "fine", "co", "re", "pre", "post", "zero", "few",
    "inter", "intra", "semi", "ultra", "anti", "over", "under", "sub",
    "super", "out", "auto", "end", "first", "second", "third", "open",
    "cost", "time", "peer", "cloud", "rule", "knowledge", "transformer",
}

HYPHEN_SUFFIXES = {
    "based", "driven", "free", "oriented", "ready", "level", "scale",
    "specific", "aware", "rich", "wide", "centric", "friendly", "sensitive",
    "dependent", "making", "time", "quality", "saving", "proof", "related",
    "critical", "constrained", "bound", "like", "type",
}


def join_lines_into_paragraph(lines: list[dict]) -> str:
    """Nối các dòng trong đoạn văn bản, tự động gộp từ bị ngắt nối gạch ngang cuối dòng thông minh."""
    result = ""
    for line in lines:
        text = line["text"].strip()
        if not text:
            continue
        if not result:
            result = text
        elif result.endswith(("-", "\u2010")) and len(result) > 1 and result[-2].isalpha():
            # Phân tích tiền tố trước dấu gạch ngang và từ tiếp theo ở dòng mới
            prefix = result[:-1].split()[-1].lower() if result[:-1].split() else ""
            first_word = text.split()[0].lower() if text.split() else ""
            
            # Giữ gạch nối cho từ ghép thực sự:
            # 1. Tiền tố thuộc danh sách từ ghép (state-, self-, cross-)
            # 2. Hậu tố thuộc danh sách từ ghép (-based, -driven, -friendly)
            # 3. Từ tiếp theo viết hoa (non-English, cross-Lingual)
            should_keep_hyphen = (
                prefix in HYPHEN_PREFIXES
                or first_word in HYPHEN_SUFFIXES
                or (text[0].isupper() and len(prefix) >= 2)
            )

            if should_keep_hyphen:
                result = result + text
            else:
                # Nối liền từ bị bẻ dòng do ngắt trang (implemen-tation -> implementation)
                result = result[:-1] + text
        else:
            result += " " + text

    return clean_and_normalize_text(result)


URL_PATTERN = re.compile(
    r"^(?:https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[A-Za-z]{2,})$",
    re.IGNORECASE,
)
PAGE_NUM_PATTERN = re.compile(r"^\s*(\d+)\s*(?:of|/)\s*(\d+)\s*$", re.IGNORECASE)
NUMBER_OR_SYMBOL_PATTERN = re.compile(
    r"^[0-9\s.,;:\-+*/=><%#$€£¥()\[\]{}|\\•·~_–—^°]+$"
)

# Chỉ coi là biểu thức toán học khi có dấu so sánh (=, <, >, <=, >=) hoặc hàm toán / phép tính rõ ràng
# Tuyệt đối không nhận diện nhầm các từ ghép tiếng Anh (như state-of-the-art, end-to-end) thành toán học
MATH_EXPLICIT_OP = r"(?:<=|>=|!=|==|->|<-|[=<>≤≥±×÷∈∉∑∏∫≈~])"
MATH_TOKEN_CHARS = r"[a-zA-Z0-9_\(\)\[\]\{\}\\\^\.\',]"
MATH_EXPLICIT_PATTERN = re.compile(
    r"^\s*" + MATH_TOKEN_CHARS + r"+\s*" + MATH_EXPLICIT_OP + r"\s*[a-zA-Z0-9_\(\)\[\]\{\}\\\^\.\',/+\-* ]+\s*$"
)


def pre_filter_text(text: str) -> tuple[bool, str]:
    """
    Lọc các đoạn text không cần dịch (số trục biểu đồ, trang, URL, ký hiệu toán học đơn giản).
    Giúp không làm biến dạng các nhãn số và tiết kiệm đáng kể thời gian dịch.
    """
    text = clean_and_normalize_text(text)
    if not text:
        return False, ""

    if NUMBER_OR_SYMBOL_PATTERN.match(text):
        return False, text

    if not any(char.isalpha() for char in text):
        return False, text

    if URL_PATTERN.fullmatch(text):
        return False, text

    match = PAGE_NUM_PATTERN.fullmatch(text)
    if match:
        return False, f"{match.group(1)} / {match.group(2)}"

    # Trích dẫn kiểu [1], [12-14]
    if re.fullmatch(r"\[\s*\d+\s*(?:[-–,]\s*\d+\s*)*\]", text):
        return False, text

    # Kiểm tra từ ghép tiếng Anh thuần túy (VD: state-of-the-art, end-to-end): BẮT BUỘC DỊCH
    if re.fullmatch(r"^[a-zA-Z]+(?:-[a-zA-Z]+)+$", text.strip()):
        return True, text

    # Biểu thức toán học / so sánh thực sự (VD: f(x) = ax^2 + bx, p < 0.05)
    if len(text) < 60 and MATH_EXPLICIT_PATTERN.match(text):
        # Nếu chỉ có chữ và dấu gạch ngang thì không phải toán học
        if not re.search(r"[0-9=<>\+×÷±≤≥≈~]", text):
            return True, text
        return False, text

    return True, text


ABBR_PATTERN = re.compile(
    r"\b(Fig|Figs|Tab|Tabs|Sec|Eq|Eqs|Ref|Refs|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|e\.g|i\.e|et al|etc|vs|cf|Inc|Corp|Ltd|Co|Vol|No|pp|Dept|Univ|Approx|Est|Ph\.D|M\.Sc|B\.Sc|U\.S|U\.K)\.",
    re.IGNORECASE,
)


def split_into_sentences(text: str) -> list[str]:
    """
    Tách đoạn văn bản thành các câu con để mô hình dịch chuẩn ngữ nghĩa:
    - Bảo vệ dấu chấm trong các từ viết tắt học thuật (Fig. 1, Dr. Smith, e.g., et al., U.S.).
    - Bảo vệ số thứ tự đầu mục (1. Introduction, 2.3. Methodology).
    - Hỗ trợ câu kết thúc trong ngoặc kép hoặc ngoặc đơn.
    """
    if not text:
        return []

    # Tạm thay dấu chấm ở số thứ tự mục và số thập phân (VD: 98.5%)
    protected = re.sub(r"(^\s*\d+(?:\.\d+)*)\.\s+", r"\1{{DOT}} ", text)
    protected = re.sub(r"(\d+)\.(\d+)", r"\1{{DOT}}\2", protected)
    protected = ABBR_PATTERN.sub(lambda m: m.group(0)[:-1] + "{{DOT}}", protected)

    # Tách câu tại dấu chấm/chấm than/chấm hỏi (kèm theo dấu ngoặc kép hoặc ngoặc đơn nếu có)
    parts = re.split(
        r'(?<!\b[A-Z])(?:(?<=[.!?]["\'”’\)])|(?<=[.!?]))\s+(?=[A-Z0-9"“\'(\[])',
        protected,
    )
    res = [p.replace("{{DOT}}", ".").strip() for p in parts if p.strip()]
    return res if res else [text]


