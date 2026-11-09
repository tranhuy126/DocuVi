import pymupdf as fitz

from .config import FONT_NAMES, get_font_buffers


class ScratchFontTester:
    def __init__(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=1200, height=1200)
        for style, fontname in FONT_NAMES.items():
            self.page.insert_font(fontname=fontname, fontbuffer=get_font_buffers()[style])

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def test_fit(self, rect: fitz.Rect, text: str, fontname: str, size: float, lineheight: float, align: int) -> bool:
        if rect.is_empty or rect.is_infinite or rect.width <= 0 or rect.height <= 0:
            return False
        test_rect = fitz.Rect(0, 0, max(1.0, rect.width - 0.5), max(1.0, rect.height - 0.5))
        return self.page.new_shape().insert_textbox(test_rect, text, fontsize=size, fontname=fontname, lineheight=lineheight, align=align) >= 0

    def fit_size(
        self, rect: fitz.Rect, text: str, fontname: str, max_size: float, min_size: float = 6.0,
        lineheight: float = 1.15, align: int = fitz.TEXT_ALIGN_LEFT, tolerance: float = 0.25,
    ) -> tuple[float, bool]:
        rect, max_size = fitz.Rect(rect), max(max_size, min_size)
        if not text or not text.strip() or rect.is_empty or rect.is_infinite or rect.width <= 0 or rect.height <= 0:
            return min_size, False
        if self.test_fit(rect, text, fontname, max_size, lineheight, align):
            return max_size, True
        if not self.test_fit(rect, text, fontname, min_size, lineheight, align):
            return min_size, False
        low, high, best_size = min_size, max_size, min_size
        while high - low > tolerance:
            mid = (low + high) / 2
            if self.test_fit(rect, text, fontname, mid, lineheight, align):
                best_size = low = mid
            else:
                high = mid
        final_size = round(best_size, 2)
        while final_size >= min_size and not self.test_fit(rect, text, fontname, final_size, lineheight, align):
            final_size = round(final_size - 0.1, 2)
        return (final_size, True) if (final_size >= min_size and self.test_fit(rect, text, fontname, final_size, lineheight, align)) else (min_size, False)

    def close(self):
        if (doc := getattr(self, "doc", None)) is not None and not getattr(doc, "is_closed", True):
            try: doc.close()
            except Exception: pass

    def __del__(self): self.close()


def get_page_bounds(page: fitz.Page) -> fitz.Rect:
    return fitz.Rect(page.rect * page.derotation_matrix) if getattr(page, "rotation", 0) % 360 != 0 else fitz.Rect(page.rect)


def collect_page_obstacles(page: fitz.Page) -> list[fitz.Rect]:
    obstacles: list[fitz.Rect] = []
    try:
        obstacles.extend(ib for img in page.get_image_info() if not (ib := fitz.Rect(img.get("bbox", ()))).is_empty and not ib.is_infinite and ib.get_area() > 4)
    except Exception:
        pass
    try:
        for dwg in page.get_drawings():
            if not dwg.get("rect") or (db := fitz.Rect(dwg["rect"])).is_infinite:
                continue
            sw = max(float(dwg.get("width") or 0.5), 0.5)
            if db.height < 1.0 and db.width >= 10.0:
                db = fitz.Rect(db.x0, db.y0 - sw, db.x1, db.y1 + sw)
            elif db.width < 1.0 and db.height >= 10.0:
                db = fitz.Rect(db.x0 - sw, db.y0, db.x1 + sw, db.y1)
            if not db.is_empty and (db.height > 6 or db.width >= 10.0):
                obstacles.append(db)
    except Exception:
        pass
    return obstacles


def get_protected_rects(elements: list[dict]) -> list[fitz.Rect]:
    protected: list[fitz.Rect] = []
    for elem in elements:
        if not elem.get("can_render"):
            ol = elem.get("original_lines", [])
            protected.extend(b for b in ([fitz.Rect(l.get("raw_bbox", l["bbox"])) for l in ol] if ol else [fitz.Rect(elem["bbox"])]) if not b.is_empty and not b.is_infinite)
    return protected


def build_redaction_rects(elem: dict, protected_rects: list[fitz.Rect], bounds: fitz.Rect, pad: float = 0.35) -> list[fitz.Rect]:
    ol = elem.get("original_lines", [])
    source_boxes = [fitz.Rect(l.get("raw_bbox", l["bbox"])) for l in ol] if ol else [fitz.Rect(elem["bbox"])]
    if source_boxes:
        ub = fitz.Rect(elem["bbox"])
        for sb in source_boxes:
            if not sb.is_empty and not sb.is_infinite:
                ub |= sb
        pu = fitz.Rect(max(bounds.x0, ub.x0 - pad), max(bounds.y0, ub.y0 - pad), min(bounds.x1, ub.x1 + pad), min(bounds.y1, ub.y1 + pad))
        if not any(pu.intersects(p) for p in protected_rects):
            return [pu] if not pu.is_empty else []

    rects: list[fitz.Rect] = []
    for bb in source_boxes:
        if bb.is_empty or bb.is_infinite:
            continue
        r = fitz.Rect(max(bounds.x0, bb.x0 - pad), max(bounds.y0, bb.y0 - pad), min(bounds.x1, bb.x1 + pad), min(bounds.y1, bb.y1 + pad))
        mid_y, mid_x = (bb.y0 + bb.y1) / 2.0, (bb.x0 + bb.x1) / 2.0
        for p in protected_rects:
            if not r.intersects(p):
                continue
            vo, ho = min(r.y1, p.y1) - max(r.y0, p.y0), min(r.x1, p.x1) - max(r.x0, p.x0)
            if vo <= 0 or ho <= 0:
                continue
            if p.width < bb.width * 0.4 and vo > 0.3 * min(bb.height, p.height):
                if p.x0 >= mid_x and p.x0 >= bb.x1 - p.width - 1.5:
                    r.x1 = min(r.x1, p.x0 - 0.1)
                    continue
                if p.x1 <= mid_x and p.x1 <= bb.x0 + p.width + 1.5:
                    r.x0 = max(r.x0, p.x1 + 0.1)
                    continue
            if (p.y0 + p.y1) / 2.0 >= mid_y:
                r.y1 = min(r.y1, p.y0 - 0.05)
            else:
                r.y0 = max(r.y0, p.y1 + 0.05)
        if not r.is_empty and r.width > 0.5 and r.height >= 0.25 * bb.height:
            rects.append(r)
    return rects


def compute_safe_insert_rect(
    page: fitz.Page, item: dict, all_elements: list[dict], gap: float = 2.0, page_obstacles: list[fitz.Rect] | None = None
) -> fitz.Rect:
    if item.get("is_table_cell"):
        return item["insert_rect"]
    rect, bounds = fitz.Rect(item["bbox"]), get_page_bounds(page)
    is_multiline = len(item.get("original_lines", [])) > 1
    max_bottom, max_right = bounds.y1 - gap, bounds.x1 - gap
    obstacles = [ob for other in all_elements if other is not item and not (ob := fitz.Rect(other["bbox"])).is_empty and not ob.is_infinite]
    obstacles.extend(
        db for db in (page_obstacles if page_obstacles is not None else collect_page_obstacles(page))
        if not db.contains(rect) and (rect & db).get_area() < 0.5 * min(rect.get_area(), max(db.get_area(), 1.0))
    )
    for obs in obstacles:
        if min(rect.y1, obs.y1) - max(rect.y0, obs.y0) > 0 and obs.x0 >= rect.x1 - 0.5:
            max_right = min(max_right, obs.x0 - gap)
    max_right = max(rect.x1, max_right)
    if is_multiline:
        safe_right, text_gap = rect.x1, max(gap, 6.0)
        for obs in obstacles:
            if min(rect.y1, obs.y1) - max(rect.y0, obs.y0) > 0 and obs.x0 >= rect.x1 - 0.5:
                safe_right = min(safe_right, obs.x0 - text_gap)
        safe_right = max(rect.x0 + 1.0, safe_right)
    else:
        safe_right = min(max_right, rect.x1 + 35.0)
    mid_y = (rect.y0 + rect.y1) / 2.0
    for obs in obstacles:
        if min(safe_right, obs.x1) - max(rect.x0, obs.x0) > 0 and obs.y0 >= mid_y and obs.y1 > rect.y1:
            max_bottom = min(max_bottom, obs.y0 - gap)
    max_bottom = max(rect.y1, max_bottom)
    if is_multiline:
        return fitz.Rect(rect.x0, rect.y0, safe_right, min(max_bottom, rect.y1 + max(30.0, rect.height * 0.45)))
    cushion = min(rect.height, 30.0) if item["size"] >= 14 else min(max(rect.height * 0.9, 14.0), 25.0)
    return fitz.Rect(rect.x0, rect.y0, safe_right, min(rect.y1 + cushion, max_bottom))
