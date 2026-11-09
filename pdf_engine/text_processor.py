import re
import unicodedata


def clean_and_normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized_text = unicodedata.normalize("NFKC", text)
    char_replacements = (
        ("\xad", ""),
        ("\u200b", ""),
        ("\ufeff", ""),
        ("’", "'"),
        ("‘", "'"),
        ("`", "'"),
        ("“", '"'),
        ("”", '"'),
        ("—", " - "),
    )
    for source_char, target_char in char_replacements:
        normalized_text = normalized_text.replace(source_char, target_char)

    normalized_text = re.sub(r"(\d+)\s*[–—]\s*(\d+)", r"\1-\2", normalized_text)
    normalized_text = normalized_text.replace("–", " - ").replace("\xa0", " ").replace("\r", " ")
    return re.sub(r"[ \t]+", " ", normalized_text).strip()


def join_spans_into_line(spans: list[dict]) -> str:
    joined_line = ""
    previous_span = None
    sorted_spans = sorted(spans, key=lambda span: span["bbox"][0])

    for current_span in sorted_spans:
        span_text = current_span["text"]
        if not span_text:
            continue

        if previous_span is not None and joined_line:
            horizontal_gap = current_span["bbox"][0] - previous_span["bbox"][2]
            space_threshold = max(previous_span["size"], current_span["size"]) * 0.15
            needs_space = (
                horizontal_gap > space_threshold
                and not joined_line.endswith((" ", "\t"))
                and not span_text.startswith((" ", "\t"))
                and span_text[0] not in ",.;:!?%)]}»’”\"'"
                and joined_line[-1] not in "([{«“‘-/"
            )
            if needs_space:
                joined_line += " "

        joined_line += span_text
        previous_span = current_span

    return joined_line.strip()


HYPHEN_PREFIXES = {
    "self", "cross", "state", "decision", "well", "user", "real", "multi", "non", "all", "high", "low",
    "long", "short", "large", "small", "fine", "co", "re", "pre", "post", "zero", "few", "inter", "intra",
    "semi", "ultra", "anti", "over", "under", "sub", "super", "out", "auto", "end", "first", "second",
    "third", "open", "cost", "time", "peer", "cloud", "rule", "knowledge", "transformer",
}
HYPHEN_SUFFIXES = {
    "based", "driven", "free", "oriented", "ready", "level", "scale", "specific", "aware", "rich", "wide",
    "centric", "friendly", "sensitive", "dependent", "making", "time", "quality", "saving", "proof",
    "related", "critical", "constrained", "bound", "like", "type",
}


def join_lines_into_paragraph(lines: list[dict]) -> str:
    paragraph_text = ""
    for line in lines:
        line_text = line["text"].strip()
        if not line_text:
            continue

        if not paragraph_text:
            paragraph_text = line_text
        elif paragraph_text.endswith("\xad") and len(paragraph_text) > 1 and paragraph_text[-2].isalpha():
            paragraph_text = paragraph_text[:-1] + line_text
        elif paragraph_text.endswith(("-", "\u2010")) and len(paragraph_text) > 1 and paragraph_text[-2].isalpha():
            words_before_hyphen = paragraph_text[:-1].split()
            prefix_word = words_before_hyphen[-1].lower() if words_before_hyphen else ""
            words_in_next_line = line_text.split()
            first_word_next_line = words_in_next_line[0].lower() if words_in_next_line else ""

            should_keep_hyphen = (
                prefix_word in HYPHEN_PREFIXES
                or first_word_next_line in HYPHEN_SUFFIXES
                or (line_text[0].isupper() and len(prefix_word) >= 2)
            )
            base_text = paragraph_text if should_keep_hyphen else paragraph_text[:-1]
            paragraph_text = base_text + line_text
        else:
            paragraph_text += " " + line_text

    return clean_and_normalize_text(paragraph_text)


URL_PATTERN = re.compile(
    r"^(?:https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[A-Za-z]{2,})$",
    re.IGNORECASE,
)
PAGE_NUM_PATTERN = re.compile(r"^\s*(\d+)\s*(?:of|/)\s*(\d+)\s*$", re.IGNORECASE)
NUMBER_OR_SYMBOL_PATTERN = re.compile(r"^[0-9\s.,;:\-+*/=><%#$€£¥()\[\]{}|\\•·~_–—^°]+$")
MATH_EXPLICIT_OP = r"(?:<=|>=|!=|==|->|<-|[=<>≤≥±×÷∈∉∑∏∫≈~])"
MATH_TOKEN_CHARS = r"[a-zA-Z0-9_\(\)\[\]\{\}\\\^\.\',]"
MATH_EXPLICIT_PATTERN = re.compile(
    r"^\s*" + MATH_TOKEN_CHARS + r"+\s*" + MATH_EXPLICIT_OP + r"\s*[a-zA-Z0-9_\(\)\[\]\{\}\\\^\.\',/+\-* ]+\s*$"
)


def pre_filter_text(text: str) -> tuple[bool, str]:
    cleaned_text = clean_and_normalize_text(text)
    if not cleaned_text:
        return False, ""

    has_alphabetic_char = any(character.isalpha() for character in cleaned_text)
    if (
        NUMBER_OR_SYMBOL_PATTERN.match(cleaned_text)
        or not has_alphabetic_char
        or URL_PATTERN.fullmatch(cleaned_text)
    ):
        return False, cleaned_text

    page_number_match = PAGE_NUM_PATTERN.fullmatch(cleaned_text)
    if page_number_match:
        formatted_page_num = f"{page_number_match.group(1)} / {page_number_match.group(2)}"
        return False, formatted_page_num

    if re.fullmatch(r"\[\s*\d+\s*(?:[-–,]\s*\d+\s*)*\]", cleaned_text):
        return False, cleaned_text

    if re.fullmatch(r"^[a-zA-Z]+(?:-[a-zA-Z]+)+$", cleaned_text.strip()):
        return True, cleaned_text

    if len(cleaned_text) < 60 and MATH_EXPLICIT_PATTERN.match(cleaned_text):
        has_math_symbol = bool(re.search(r"[0-9=<>\+×÷±≤≥≈~]", cleaned_text))
        return not has_math_symbol, cleaned_text

    return True, cleaned_text


ABBR_PATTERN = re.compile(
    r"\b(Fig|Figs|Tab|Tabs|Sec|Eq|Eqs|Ref|Refs|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|e\.g|i\.e|et al|etc|vs|cf|Inc|Corp|Ltd|Co|Vol|No|pp|Dept|Univ|Approx|Est|Ph\.D|M\.Sc|B\.Sc|U\.S|U\.K)\.",
    re.IGNORECASE,
)


def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []

    protected_text = re.sub(r"(^\s*\d+(?:\.\d+)*)\.\s+", r"\1{{DOT}} ", text)
    protected_text = re.sub(r"(\d+)\.(\d+)", r"\1{{DOT}}\2", protected_text)
    protected_text = ABBR_PATTERN.sub(lambda match: match.group(0)[:-1] + "{{DOT}}", protected_text)

    sentence_boundary_pattern = r'(?<!\b[A-Z])(?:(?<=[.!?]["\'”’\)])|(?<=[.!?]))\s+(?=[A-Z0-9"“\'(\[])'
    raw_parts = re.split(sentence_boundary_pattern, protected_text)

    sentences = [part.replace("{{DOT}}", ".").strip() for part in raw_parts if part.strip()]
    return sentences or [text]

