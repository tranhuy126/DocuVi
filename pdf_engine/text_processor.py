import re
import unicodedata


def clean_and_normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\xad", "").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ")

    text = re.sub(r"(\d+)\s*[–—]\s*(\d+)", r"\1-\2", text)
    text = text.replace("–", " - ")
    text = text.replace("\xa0", " ").replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def join_spans_into_line(spans: list[dict]) -> str:
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
    result = ""
    for line in lines:
        text = line["text"].strip()
        if not text:
            continue
        if not result:
            result = text
        elif result.endswith(("-", "\u2010")) and len(result) > 1 and result[-2].isalpha():
            prefix = result[:-1].split()[-1].lower() if result[:-1].split() else ""
            first_word = text.split()[0].lower() if text.split() else ""


            should_keep_hyphen = (
                prefix in HYPHEN_PREFIXES
                or first_word in HYPHEN_SUFFIXES
                or (text[0].isupper() and len(prefix) >= 2)
            )

            if should_keep_hyphen:
                result = result + text
            else:
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


MATH_EXPLICIT_OP = r"(?:<=|>=|!=|==|->|<-|[=<>≤≥±×÷∈∉∑∏∫≈~])"
MATH_TOKEN_CHARS = r"[a-zA-Z0-9_\(\)\[\]\{\}\\\^\.\',]"
MATH_EXPLICIT_PATTERN = re.compile(
    r"^\s*" + MATH_TOKEN_CHARS + r"+\s*" + MATH_EXPLICIT_OP + r"\s*[a-zA-Z0-9_\(\)\[\]\{\}\\\^\.\',/+\-* ]+\s*$"
)


def pre_filter_text(text: str) -> tuple[bool, str]:
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


    if re.fullmatch(r"\[\s*\d+\s*(?:[-–,]\s*\d+\s*)*\]", text):
        return False, text


    if re.fullmatch(r"^[a-zA-Z]+(?:-[a-zA-Z]+)+$", text.strip()):
        return True, text


    if len(text) < 60 and MATH_EXPLICIT_PATTERN.match(text):
        if not re.search(r"[0-9=<>\+×÷±≤≥≈~]", text):
            return True, text
        return False, text

    return True, text


ABBR_PATTERN = re.compile(
    r"\b(Fig|Figs|Tab|Tabs|Sec|Eq|Eqs|Ref|Refs|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|e\.g|i\.e|et al|etc|vs|cf|Inc|Corp|Ltd|Co|Vol|No|pp|Dept|Univ|Approx|Est|Ph\.D|M\.Sc|B\.Sc|U\.S|U\.K)\.",
    re.IGNORECASE,
)


def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []


    protected = re.sub(r"(^\s*\d+(?:\.\d+)*)\.\s+", r"\1{{DOT}} ", text)
    protected = re.sub(r"(\d+)\.(\d+)", r"\1{{DOT}}\2", protected)
    protected = ABBR_PATTERN.sub(lambda m: m.group(0)[:-1] + "{{DOT}}", protected)


    parts = re.split(
        r'(?<!\b[A-Z])(?:(?<=[.!?]["\'”’\)])|(?<=[.!?]))\s+(?=[A-Z0-9"“\'(\[])',
        protected,
    )
    res = [p.replace("{{DOT}}", ".").strip() for p in parts if p.strip()]
    return res if res else [text]
