import pymupdf as fitz

from .config import FONT_NAMES, get_font_buffers


class ScratchFontTester:
    def __init__(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=1200, height=1200)
        font_buffers = get_font_buffers()
        for style_name, fontname in FONT_NAMES.items():
            self.page.insert_font(fontname=fontname, fontbuffer=font_buffers[style_name])

    def __enter__(self):
        return self

    def __exit__(self, *args):
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

        test_rect = fitz.Rect(0, 0, max(1.0, rect.width - 0.5), max(1.0, rect.height - 0.5))
        remaining_height = self.page.new_shape().insert_textbox(
            test_rect,
            text,
            fontsize=size,
            fontname=fontname,
            lineheight=lineheight,
            align=align,
        )
        return remaining_height >= 0

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
        target_rect = fitz.Rect(rect)
        upper_bound_size = max(max_size, min_size)

        if (
            not text
            or not text.strip()
            or target_rect.is_empty
            or target_rect.is_infinite
            or target_rect.width <= 0
            or target_rect.height <= 0
        ):
            return min_size, False

        if self.test_fit(target_rect, text, fontname, upper_bound_size, lineheight, align):
            return upper_bound_size, True

        if not self.test_fit(target_rect, text, fontname, min_size, lineheight, align):
            return min_size, False

        low_size = min_size
        high_size = upper_bound_size
        best_size = min_size

        while high_size - low_size > tolerance:
            mid_size = (low_size + high_size) / 2
            if self.test_fit(target_rect, text, fontname, mid_size, lineheight, align):
                best_size = mid_size
                low_size = mid_size
            else:
                high_size = mid_size

        final_size = round(best_size, 2)
        while final_size >= min_size and not self.test_fit(
            target_rect, text, fontname, final_size, lineheight, align
        ):
            final_size = round(final_size - 0.1, 2)

        is_fitted = final_size >= min_size and self.test_fit(
            target_rect, text, fontname, final_size, lineheight, align
        )
        if is_fitted:
            return final_size, True
        return min_size, False

    def close(self):
        scratch_doc = getattr(self, "doc", None)
        if scratch_doc is not None and not getattr(scratch_doc, "is_closed", True):
            try:
                scratch_doc.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


def get_page_bounds(page: fitz.Page) -> fitz.Rect:
    page_rotation = getattr(page, "rotation", 0)
    if page_rotation % 360 != 0:
        return fitz.Rect(page.rect * page.derotation_matrix)
    return fitz.Rect(page.rect)


def collect_page_obstacles(page: fitz.Page) -> list[fitz.Rect]:
    obstacles: list[fitz.Rect] = []
    try:
        for image_info in page.get_image_info():
            image_bbox = fitz.Rect(image_info.get("bbox", ()))
            if not image_bbox.is_empty and not image_bbox.is_infinite and image_bbox.get_area() > 4:
                obstacles.append(image_bbox)
    except Exception:
        pass

    try:
        for drawing in page.get_drawings():
            raw_drawing_rect = drawing.get("rect")
            if not raw_drawing_rect:
                continue

            drawing_bbox = fitz.Rect(raw_drawing_rect)
            if drawing_bbox.is_infinite:
                continue

            stroke_width = max(float(drawing.get("width") or 0.5), 0.5)
            if drawing_bbox.height < 1.0 and drawing_bbox.width >= 10.0:
                drawing_bbox = fitz.Rect(
                    drawing_bbox.x0,
                    drawing_bbox.y0 - stroke_width,
                    drawing_bbox.x1,
                    drawing_bbox.y1 + stroke_width,
                )
            elif drawing_bbox.width < 1.0 and drawing_bbox.height >= 10.0:
                drawing_bbox = fitz.Rect(
                    drawing_bbox.x0 - stroke_width,
                    drawing_bbox.y0,
                    drawing_bbox.x1 + stroke_width,
                    drawing_bbox.y1,
                )

            if not drawing_bbox.is_empty and (drawing_bbox.height > 6 or drawing_bbox.width >= 10.0):
                obstacles.append(drawing_bbox)
    except Exception:
        pass

    return obstacles


def get_protected_rects(elements: list[dict]) -> list[fitz.Rect]:
    protected_rects: list[fitz.Rect] = []
    for element in elements:
        if not element.get("can_render"):
            original_lines = element.get("original_lines", [])
            if original_lines:
                candidate_rects = [
                    fitz.Rect(line.get("raw_bbox", line["bbox"]))
                    for line in original_lines
                ]
            else:
                candidate_rects = [fitz.Rect(element["bbox"])]

            for rect in candidate_rects:
                if not rect.is_empty and not rect.is_infinite:
                    protected_rects.append(rect)

    return protected_rects


def build_redaction_rects(
    elem: dict,
    protected_rects: list[fitz.Rect],
    bounds: fitz.Rect,
    pad: float = 0.35,
) -> list[fitz.Rect]:
    original_lines = elem.get("original_lines", [])
    if original_lines:
        source_boxes = [fitz.Rect(line.get("raw_bbox", line["bbox"])) for line in original_lines]
    else:
        source_boxes = [fitz.Rect(elem["bbox"])]

    if source_boxes:
        union_bbox = fitz.Rect(elem["bbox"])
        for source_box in source_boxes:
            if not source_box.is_empty and not source_box.is_infinite:
                union_bbox |= source_box

        padded_union_rect = fitz.Rect(
            max(bounds.x0, union_bbox.x0 - pad),
            max(bounds.y0, union_bbox.y0 - pad),
            min(bounds.x1, union_bbox.x1 + pad),
            min(bounds.y1, union_bbox.y1 + pad),
        )
        intersects_protected = any(
            padded_union_rect.intersects(protected_rect)
            for protected_rect in protected_rects
        )
        if not intersects_protected:
            return [padded_union_rect] if not padded_union_rect.is_empty else []

    redaction_rects: list[fitz.Rect] = []
    for source_box in source_boxes:
        if source_box.is_empty or source_box.is_infinite:
            continue

        redaction_rect = fitz.Rect(
            max(bounds.x0, source_box.x0 - pad),
            max(bounds.y0, source_box.y0 - pad),
            min(bounds.x1, source_box.x1 + pad),
            min(bounds.y1, source_box.y1 + pad),
        )
        box_center_y = (source_box.y0 + source_box.y1) / 2.0
        box_center_x = (source_box.x0 + source_box.x1) / 2.0

        for protected_rect in protected_rects:
            if not redaction_rect.intersects(protected_rect):
                continue

            vertical_overlap = min(redaction_rect.y1, protected_rect.y1) - max(redaction_rect.y0, protected_rect.y0)
            horizontal_overlap = min(redaction_rect.x1, protected_rect.x1) - max(redaction_rect.x0, protected_rect.x0)
            if vertical_overlap <= 0 or horizontal_overlap <= 0:
                continue

            is_narrow_side_obstacle = (
                protected_rect.width < source_box.width * 0.4
                and vertical_overlap > 0.3 * min(source_box.height, protected_rect.height)
            )
            if is_narrow_side_obstacle:
                if protected_rect.x0 >= box_center_x and protected_rect.x0 >= source_box.x1 - protected_rect.width - 1.5:
                    redaction_rect.x1 = min(redaction_rect.x1, protected_rect.x0 - 0.1)
                    continue
                if protected_rect.x1 <= box_center_x and protected_rect.x1 <= source_box.x0 + protected_rect.width + 1.5:
                    redaction_rect.x0 = max(redaction_rect.x0, protected_rect.x1 + 0.1)
                    continue

            protected_center_y = (protected_rect.y0 + protected_rect.y1) / 2.0
            if protected_center_y >= box_center_y:
                redaction_rect.y1 = min(redaction_rect.y1, protected_rect.y0 - 0.05)
            else:
                redaction_rect.y0 = max(redaction_rect.y0, protected_rect.y1 + 0.05)

        if (
            not redaction_rect.is_empty
            and redaction_rect.width > 0.5
            and redaction_rect.height >= 0.25 * source_box.height
        ):
            redaction_rects.append(redaction_rect)

    return redaction_rects


def compute_safe_insert_rect(
    page: fitz.Page,
    item: dict,
    all_elements: list[dict],
    gap: float = 2.0,
    page_obstacles: list[fitz.Rect] | None = None,
) -> fitz.Rect:
    if item.get("is_table_cell"):
        return item["insert_rect"]

    element_rect = fitz.Rect(item["bbox"])
    page_bounds = get_page_bounds(page)
    is_multiline = len(item.get("original_lines", [])) > 1
    max_bottom = page_bounds.y1 - gap
    max_right = page_bounds.x1 - gap

    obstacles: list[fitz.Rect] = []
    for other_element in all_elements:
        if other_element is not item:
            other_bbox = fitz.Rect(other_element["bbox"])
            if not other_bbox.is_empty and not other_bbox.is_infinite:
                obstacles.append(other_bbox)

    graphic_obstacles = (
        page_obstacles
        if page_obstacles is not None
        else collect_page_obstacles(page)
    )
    for graphic_bbox in graphic_obstacles:
        intersection_area = (element_rect & graphic_bbox).get_area()
        min_reference_area = min(element_rect.get_area(), max(graphic_bbox.get_area(), 1.0))
        if not graphic_bbox.contains(element_rect) and intersection_area < 0.5 * min_reference_area:
            obstacles.append(graphic_bbox)

    for obstacle in obstacles:
        vertical_overlap = min(element_rect.y1, obstacle.y1) - max(element_rect.y0, obstacle.y0)
        if vertical_overlap > 0 and obstacle.x0 >= element_rect.x1 - 0.5:
            max_right = min(max_right, obstacle.x0 - gap)
    max_right = max(element_rect.x1, max_right)

    if is_multiline:
        safe_right = element_rect.x1
        text_gap = max(gap, 6.0)
        for obstacle in obstacles:
            vertical_overlap = min(element_rect.y1, obstacle.y1) - max(element_rect.y0, obstacle.y0)
            if vertical_overlap > 0 and obstacle.x0 >= element_rect.x1 - 0.5:
                safe_right = min(safe_right, obstacle.x0 - text_gap)
        safe_right = max(element_rect.x0 + 1.0, safe_right)
    else:
        safe_right = min(max_right, element_rect.x1 + 35.0)

    element_center_y = (element_rect.y0 + element_rect.y1) / 2.0
    for obstacle in obstacles:
        horizontal_overlap = min(safe_right, obstacle.x1) - max(element_rect.x0, obstacle.x0)
        if horizontal_overlap > 0 and obstacle.y0 >= element_center_y and obstacle.y1 > element_rect.y1:
            max_bottom = min(max_bottom, obstacle.y0 - gap)
    max_bottom = max(element_rect.y1, max_bottom)

    if is_multiline:
        expanded_bottom = min(max_bottom, element_rect.y1 + max(30.0, element_rect.height * 0.45))
        return fitz.Rect(element_rect.x0, element_rect.y0, safe_right, expanded_bottom)

    if item["size"] >= 14:
        vertical_cushion = min(element_rect.height, 30.0)
    else:
        vertical_cushion = min(max(element_rect.height * 0.9, 14.0), 25.0)

    return fitz.Rect(
        element_rect.x0,
        element_rect.y0,
        safe_right,
        min(element_rect.y1 + vertical_cushion, max_bottom),
    )

