import sys

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from .config import (
    BASE_DIR,
    FONT_BUFFERS,
    FONT_DIR,
    FONT_FILES,
    FONT_NAMES,
    detect_font_style,
    get_font_buffers,
)
from .layout_extractor import (
    extract_page_elements,
    group_lines_into_elements,
    split_horizontal_lines,
)
from .renderer import (
    ScratchFontTester,
    build_redaction_rects,
    collect_page_obstacles,
    compute_safe_insert_rect,
    get_page_bounds,
    get_protected_rects,
)
from .text_processor import (
    clean_and_normalize_text,
    join_lines_into_paragraph,
    join_spans_into_line,
    pre_filter_text,
    split_into_sentences,
)
from .translator import (
    MODEL_NAME,
    get_device,
    get_translation_model,
    translate_batch,
)

__all__ = [
    "BASE_DIR",
    "FONT_DIR",
    "FONT_FILES",
    "FONT_NAMES",
    "FONT_BUFFERS",
    "get_font_buffers",
    "detect_font_style",
    "MODEL_NAME",
    "get_device",
    "get_translation_model",
    "translate_batch",
    "clean_and_normalize_text",
    "join_spans_into_line",
    "join_lines_into_paragraph",
    "pre_filter_text",
    "split_into_sentences",
    "split_horizontal_lines",
    "group_lines_into_elements",
    "extract_page_elements",
    "ScratchFontTester",
    "compute_safe_insert_rect",
    "build_redaction_rects",
    "collect_page_obstacles",
    "get_page_bounds",
    "get_protected_rects",
]

