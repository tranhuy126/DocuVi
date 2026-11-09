from pathlib import Path
import threading

import pymupdf as fitz

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "Font"

FONT_FILES = {
    "regular": "times.ttf",
    "bold": "timesbd.ttf",
    "italic": "timesi.ttf",
    "bold_italic": "timesbi.ttf",
}

FONT_NAMES = {
    "regular": "TNR",
    "bold": "TNR_Bold",
    "italic": "TNR_Italic",
    "bold_italic": "TNR_BoldItalic",
}

_FONT_BUFFERS = None
_font_lock = threading.Lock()


def get_font_buffers() -> dict[str, bytes]:
    global _FONT_BUFFERS
    if _FONT_BUFFERS is not None:
        return _FONT_BUFFERS

    with _font_lock:
        if _FONT_BUFFERS is not None:
            return _FONT_BUFFERS

        regular_font_path = FONT_DIR / FONT_FILES["regular"]
        base_regular_buffer = None

        if regular_font_path.exists():
            try:
                base_regular_buffer = fitz.Font(fontfile=str(regular_font_path)).buffer
            except Exception as error:
                print(f"[CẢNH BÁO] Không đọc được font regular tại {regular_font_path}: {error}")

        if base_regular_buffer is None:
            system_fallback_paths = (
                Path("C:/Windows/Fonts/times.ttf"),
                Path("/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
            )
            for system_font_path in system_fallback_paths:
                if system_font_path.exists():
                    try:
                        base_regular_buffer = fitz.Font(fontfile=str(system_font_path)).buffer
                        print(f"[DocuVi] Đã nạp font thay thế từ hệ thống: {system_font_path}")
                        break
                    except Exception:
                        pass

        if base_regular_buffer is None:
            raise FileNotFoundError(
                f"Không tìm thấy font Times New Roman chuẩn Unicode tại {FONT_DIR} hoặc font hệ thống. "
                "Vui lòng đảm bảo thư mục 'Font/' chứa ít nhất file 'times.ttf'."
            )

        font_buffers = {"regular": base_regular_buffer}
        for style_name in ("bold", "italic", "bold_italic"):
            font_filename = FONT_FILES[style_name]
            bundled_font_path = FONT_DIR / font_filename
            system_font_path = Path(f"C:/Windows/Fonts/{font_filename}")

            if bundled_font_path.exists():
                selected_font_path = bundled_font_path
            elif system_font_path.exists():
                selected_font_path = system_font_path
            else:
                selected_font_path = None

            font_buffers[style_name] = base_regular_buffer
            if selected_font_path is not None:
                try:
                    font_buffers[style_name] = fitz.Font(fontfile=str(selected_font_path)).buffer
                except Exception as error:
                    if selected_font_path == bundled_font_path:
                        print(
                            f"[CẢNH BÁO] Lỗi đọc font {style_name} ({bundled_font_path}), "
                            f"dùng fallback regular: {error}"
                        )

        _FONT_BUFFERS = font_buffers
    return _FONT_BUFFERS


class _LazyFontBuffers(dict):
    def __getitem__(self, key):
        return get_font_buffers()[key]

    def items(self):
        return get_font_buffers().items()

    def keys(self):
        return get_font_buffers().keys()

    def values(self):
        return get_font_buffers().values()

    def get(self, key, default=None):
        return get_font_buffers().get(key, default)

    def __contains__(self, key):
        return key in get_font_buffers()

    def __len__(self):
        return len(get_font_buffers())

    def __iter__(self):
        return iter(get_font_buffers())


FONT_BUFFERS = _LazyFontBuffers()


def detect_font_style(flags: int = 0, font_name: str = "") -> str:
    font_name_lower = (font_name or "").lower()
    bold_keywords = ("bold", "black", "heavy", "semibold", "medium")
    italic_keywords = ("italic", "oblique", "slanted")

    is_bold = bool(flags & 16) or any(keyword in font_name_lower for keyword in bold_keywords)
    is_italic = bool(flags & 2) or any(keyword in font_name_lower for keyword in italic_keywords)

    if is_bold and is_italic:
        return "bold_italic"
    if is_bold:
        return "bold"
    if is_italic:
        return "italic"
    return "regular"

