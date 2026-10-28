import pymupdf as fitz
from pathlib import Path
import threading

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
    if _FONT_BUFFERS is None:
        with _font_lock:
            if _FONT_BUFFERS is None:
                buffers = {}
                # Tìm font regular trước làm điểm tựa fallback
                regular_path = FONT_DIR / FONT_FILES["regular"]
                system_font_candidates = [
                    Path("C:/Windows/Fonts/times.ttf"),
                    Path("/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
                    Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
                ]
                base_regular_buffer = None
                if regular_path.exists():
                    try:
                        base_regular_buffer = fitz.Font(fontfile=str(regular_path)).buffer
                    except Exception as e:
                        print(f"[CẢNH BÁO] Không đọc được font regular tại {regular_path}: {e}")
                
                if base_regular_buffer is None:
                    for sys_p in system_font_candidates:
                        if sys_p.exists():
                            try:
                                base_regular_buffer = fitz.Font(fontfile=str(sys_p)).buffer
                                print(f"[DocuVi] Đã nạp font thay thế từ hệ thống: {sys_p}")
                                break
                            except Exception:
                                pass

                if base_regular_buffer is None:
                    raise FileNotFoundError(
                        f"Không tìm thấy font Times New Roman chuẩn Unicode tại {FONT_DIR} hoặc font hệ thống. "
                        "Vui lòng đảm bảo thư mục 'Font/' chứa ít nhất file 'times.ttf'."
                    )

                buffers["regular"] = base_regular_buffer

                # Nạp các style còn lại, nếu thiếu thì fallback về base_regular_buffer
                for style in ("bold", "italic", "bold_italic"):
                    filename = FONT_FILES[style]
                    font_path = FONT_DIR / filename
                    if font_path.exists():
                        try:
                            buffers[style] = fitz.Font(fontfile=str(font_path)).buffer
                        except Exception as e:
                            print(f"[CẢNH BÁO] Lỗi đọc font {style} ({font_path}), dùng fallback regular: {e}")
                            buffers[style] = base_regular_buffer
                    else:
                        # Thử tìm font biến thể từ hệ thống
                        sys_style_path = Path(f"C:/Windows/Fonts/{filename}")
                        if sys_style_path.exists():
                            try:
                                buffers[style] = fitz.Font(fontfile=str(sys_style_path)).buffer
                            except Exception:
                                buffers[style] = base_regular_buffer
                        else:
                            buffers[style] = base_regular_buffer

                _FONT_BUFFERS = buffers
    return _FONT_BUFFERS


class _LazyFontBuffers(dict):
    """Ủy quyền truy cập dict để tương thích ngược nếu mã ngoài gọi trực tiếp FONT_BUFFERS."""

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
    # PyMuPDF span flags: bit 4 (16) = bold, bit 1 (2) = italic
    fn = font_name.lower() if font_name else ""
    is_bold = bool(flags & 16) or any(k in fn for k in ("bold", "black", "heavy", "semibold", "medium"))
    is_italic = bool(flags & 2) or any(k in fn for k in ("italic", "oblique", "slanted"))
    if is_bold and is_italic:
        return "bold_italic"
    elif is_bold:
        return "bold"
    elif is_italic:
        return "italic"
    else:
        return "regular"

