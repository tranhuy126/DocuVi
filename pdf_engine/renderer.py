import pymupdf as fitz

from .config import FONT_NAMES, get_font_buffers


class ScratchFontTester:
    """Trang PDF ảo trong RAM để đo kích thước font mà không ghi stream (tránh rò rỉ bộ nhớ)."""

    def __init__(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=1200, height=1200)
        font_buffers = get_font_buffers()
        for style, fontname in FONT_NAMES.items():
            self.page.insert_font(fontname=fontname, fontbuffer=font_buffers[style])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def test_fit(
        self,
        rect: fitz.Rect,
        text: str,
        fontname: str,
        size: float,
        lineheight: float,
        align: int,
    ) -> bool:
        if rect.is_empty or rect.is_infinite or rect.width <= 0 or rect.height <= 0:
            return False

        # Thêm đệm an toàn 0.5pt để đảm bảo khi vẽ thật trên trang không bị lỗi sai số làm tròn float
        test_w = max(1.0, rect.width - 0.5)
        test_h = max(1.0, rect.height - 0.5)
        test_rect = fitz.Rect(0, 0, test_w, test_h)

        # Sử dụng Shape không commit để đo kích thước mà không ghi vào page stream (tránh phình RAM)
        shape = self.page.new_shape()
        rc = shape.insert_textbox(
            test_rect,
            text,
            fontsize=size,
            fontname=fontname,
            lineheight=lineheight,
            align=align,
        )
        return rc >= 0

    def fit_size(
        self,
        rect: fitz.Rect,
        text: str,
        fontname: str,
        max_size: float,
        min_size: float = 6.0,
        lineheight: float = 1.15,
        align: int = fitz.TEXT_ALIGN_LEFT,
        tolerance: float = 0.25,
    ) -> tuple[float, bool]:
        """Tìm kiếm nhị phân cỡ font tối ưu để văn bản nằm vừa trong bounding box."""
        rect = fitz.Rect(rect)
        if not text or not text.strip():
            return min_size, False
        if rect.is_empty or rect.is_infinite or rect.width <= 0 or rect.height <= 0:
            return min_size, False

        max_size = max(max_size, min_size)

        # Trả về ngay nếu vừa vặn ở cỡ font gốc
        if self.test_fit(rect, text, fontname, max_size, lineheight, align):
            return max_size, True

        if not self.test_fit(rect, text, fontname, min_size, lineheight, align):
            return min_size, False

        low = min_size
        high = max_size
        best_size = min_size

        while high - low > tolerance:
            mid = (low + high) / 2
            if self.test_fit(rect, text, fontname, mid, lineheight, align):
                best_size = mid
                low = mid
            else:
                high = mid

        final_size = round(best_size, 2)
        # Kiểm chứng chắc chắn sau khi làm tròn, nếu không vừa thì lùi nhẹ 0.1pt
        while final_size >= min_size and not self.test_fit(rect, text, fontname, final_size, lineheight, align):
            final_size = round(final_size - 0.1, 2)

        if final_size >= min_size and self.test_fit(rect, text, fontname, final_size, lineheight, align):
            return final_size, True
        return min_size, False

    def close(self):
        doc = getattr(self, "doc", None)
        if doc is not None:
            try:
                if not doc.is_closed:
                    doc.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def compute_safe_insert_rect(
    page: fitz.Page, item: dict, all_elements: list[dict], gap: float = 2.0
) -> fitz.Rect:
    """
    Tính khung chèn chữ an toàn:
    - Cố định độ rộng cột cho văn bản nhiều dòng để giữ bố cục cột song song.
    - Chặn mở rộng xâm lấn sang các đoạn văn khác, bảng biểu, hình ảnh (page.get_image_info) và hình vẽ vector.
    - Cho phép nới dài chiều dọc vào vùng khoảng trắng tự nhiên.
    """
    rect = fitz.Rect(item["bbox"])
    if item.get("is_table_cell"):
        return item["insert_rect"]

    is_multiline = len(item.get("original_lines", [])) > 1

    max_bottom = page.rect.y1 - gap
    max_right = page.rect.x1 - gap

    # Thu thập toàn bộ chướng ngại vật trên trang: khối văn bản khác, hình ảnh, hình vẽ vector
    obstacles: list[fitz.Rect] = []

    # 1. Các phần tử văn bản khác
    for other in all_elements:
        if other is not item:
            ob = fitz.Rect(other["bbox"])
            if not ob.is_empty and not ob.is_infinite:
                obstacles.append(ob)

    # 2. Hình ảnh trên trang (chống đè chữ lên hình ảnh)
    try:
        for img_info in page.get_image_info():
            ib = fitz.Rect(img_info.get("bbox", ()))
            if not ib.is_empty and not ib.is_infinite and ib.get_area() > 4:
                obstacles.append(ib)
    except Exception:
        pass

    # 3. Hình vẽ đồ họa vector (đường kẻ, biểu đồ, khung viền lớn)
    try:
        for dwg in page.get_drawings():
            db = fitz.Rect(dwg.get("rect", ()))
            # Chỉ coi là chướng ngại nếu hình vẽ có diện tích đáng kể và không phải là vùng của chính item
            if not db.is_empty and not db.is_infinite and db.height > 6 and (rect & db).get_area() < 0.5 * db.get_area():
                obstacles.append(db)
    except Exception:
        pass

    # Xét giới hạn theo phương thẳng đứng (chặn bên dưới) và phương ngang (chặn bên phải)
    for obs in obstacles:
        # Chặn phương thẳng đứng phía dưới: chướng ngại vật nằm bên dưới và có giao thoa theo phương ngang
        ho = min(rect.x1, obs.x1) - max(rect.x0, obs.x0)
        if ho > 0 and obs.y0 >= rect.y1 - 0.5:
            max_bottom = min(max_bottom, obs.y0 - gap)

        # Chặn phương ngang bên phải: chướng ngại vật nằm bên phải và có giao thoa theo phương thẳng đứng
        vo = min(rect.y1, obs.y1) - max(rect.y0, obs.y0)
        if vo > 0 and obs.x0 >= rect.x1 - 0.5:
            max_right = min(max_right, obs.x0 - gap)

    max_bottom = max(rect.y1, max_bottom)
    max_right = max(rect.x1, max_right)

    if is_multiline:
        # Cố định độ rộng cột ngang, chỉ nới dài theo chiều dọc vào khoảng trắng an toàn
        safe_h = min(max_bottom, rect.y1 + max(30.0, rect.height * 0.45))
        return fitz.Rect(rect.x0, rect.y0, rect.x1, safe_h)
    else:
        # Tiêu đề hoặc dòng đơn: cho phép nới thêm chiều dọc (đủ để ngắt 2 dòng nếu cần)
        downward_cushion = (
            min(rect.height * 1.0, 30.0)
            if item["size"] >= 14
            else min(max(rect.height * 0.9, 14.0), 25.0)
        )
        safe_bottom = min(rect.y1 + downward_cushion, max_bottom)
        safe_right = min(max_right, rect.x1 + 35.0)
        return fitz.Rect(rect.x0, rect.y0, safe_right, safe_bottom)

