import pymupdf as fitz
import statistics
from collections import Counter

from .config import FONT_NAMES, detect_font_style
from .text_processor import (
    join_spans_into_line,
    join_lines_into_paragraph,
    clean_and_normalize_text,
)


def split_horizontal_lines(raw_lines: list[dict]) -> list[list[dict]]:
    """Phát hiện và phân tách các dòng văn bản nằm song song theo chiều ngang (cột, bảng, biểu đồ)."""
    if len(raw_lines) <= 1:
        return [raw_lines]

    has_side_by_side = False
    for i in range(len(raw_lines)):
        for j in range(i + 1, len(raw_lines)):
            l1 = raw_lines[i]["bbox"]
            l2 = raw_lines[j]["bbox"]
            h1 = l1.y1 - l1.y0
            h2 = l2.y1 - l2.y0
            vo = min(l1.y1, l2.y1) - max(l1.y0, l2.y0)
            if vo > 0.4 * min(h1, h2) and abs(l1.x0 - l2.x0) > 8:
                has_side_by_side = True
                break
        if has_side_by_side:
            break

    if not has_side_by_side:
        return [raw_lines]

    clusters: list[list[dict]] = []
    for line in raw_lines:
        matched = False
        l_x0, l_x1 = line["bbox"].x0, line["bbox"].x1
        for cluster in clusters:
            c_x0 = min(l["bbox"].x0 for l in cluster)
            c_x1 = max(l["bbox"].x1 for l in cluster)
            ho = min(c_x1, l_x1) - max(c_x0, l_x0)
            min_w = min(c_x1 - c_x0, l_x1 - l_x0)
            if (min_w > 0 and ho > 0.3 * min_w) or (
                abs(c_x0 - l_x0) < 6 and abs(c_x1 - l_x1) < 6
            ):
                cluster.append(line)
                matched = True
                break
        if not matched:
            clusters.append([line])

    clusters.sort(key=lambda c: min(l["bbox"].x0 for l in c))
    for c in clusters:
        c.sort(key=lambda l: l["bbox"].y0)

    return clusters


def group_lines_into_elements(raw_lines: list[dict]) -> list[list[dict]]:
    """
    Gom các dòng cùng block thành các nhóm phần tử:
    - Kiểm tra độ tương đồng cỡ chữ.
    - Phân biệt kiểu chữ Đậm (Bold) của tiêu đề và chữ Thường (Regular) của đoạn văn.
    - Phát hiện khoảng cách dòng bất thường (paragraph gap) để không gom các đoạn độc lập thành một khối.
    """
    if len(raw_lines) <= 1:
        return [raw_lines]

    horizontal_clusters = split_horizontal_lines(raw_lines)

    # Kiểm tra đặc trưng bảng biểu không viền / lưới số liệu tài chính:
    # 1. Có từ 3 cụm cột song song trở lên trong cùng một khối.
    # 2. Hoặc có từ 2 cột song song và ít nhất 1 cột chứa nhiều số liệu/ký hiệu tài chính (>30% dòng không có chữ cái).
    is_borderless_table = len(horizontal_clusters) >= 3
    if not is_borderless_table and len(horizontal_clusters) >= 2:
        for c in horizontal_clusters:
            non_alpha_count = sum(
                1 for l in c if not any(ch.isalpha() for ch in l.get("text", ""))
            )
            if non_alpha_count >= 2 and (non_alpha_count / len(c)) >= 0.30:
                is_borderless_table = True
                break

    if is_borderless_table:
        print(
            f"[CẢNH BÁO BỐ CỤC] Phát hiện cấu trúc bảng biểu không viền / lưới số liệu "
            f"({len(horizontal_clusters)} cột song song). Giữ nguyên từng ô/dòng độc lập "
            f"để bảo toàn tuyệt đối liên kết hàng - cột của số liệu."
        )
        return [[line] for cluster in horizontal_clusters for line in cluster]

    final_groups = []
    for cluster in horizontal_clusters:
        if len(cluster) <= 1:
            final_groups.append(cluster)
            continue

        current_subgroup = [cluster[0]]
        for next_line in cluster[1:]:
            prev_line = current_subgroup[-1]
            p_bbox = prev_line["bbox"]
            n_bbox = next_line["bbox"]

            prev_sizes = [
                s["size"] for s in prev_line.get("spans", []) if s.get("size")
            ]
            next_sizes = [
                s["size"] for s in next_line.get("spans", []) if s.get("size")
            ]

            prev_size = statistics.median(prev_sizes) if prev_sizes else 10.0
            next_size = statistics.median(next_sizes) if next_sizes else 10.0

            size_diff = abs(prev_size - next_size)
            size_ratio = max(prev_size, next_size) / max(
                min(prev_size, next_size), 0.1
            )

            # Kiểm tra phân định kiểu font: Tiêu đề in đậm vs Thân bài in thường
            prev_is_bold = any(
                s.get("style", "") in ("bold", "bold_italic")
                for s in prev_line.get("spans", [])
            )
            next_is_bold = any(
                s.get("style", "") in ("bold", "bold_italic")
                for s in next_line.get("spans", [])
            )
            style_diff = prev_is_bold != next_is_bold

            # Khoảng cách thẳng đứng giữa 2 dòng liên tiếp
            line_h = max(p_bbox.height, n_bbox.height, max(prev_size, next_size))
            vert_gap = n_bbox.y0 - p_bbox.y1

            # Tách thành phần tử mới khi:
            # 1. Cỡ chữ chênh lệch từ 1.5pt trở lên hoặc tỷ lệ >= 1.2
            # 2. Thay đổi style in đậm (tiêu đề / thân bài)
            # 3. Có khoảng trống giữa các đoạn văn (vert_gap > 0.65 * line_h và vert_gap > 5.0pt)
            # 4. Dòng tiếp theo bị thụt lùi hoặc nằm ngược tọa độ
            is_different_element = (
                size_diff >= 1.5
                or size_ratio >= 1.2
                or style_diff
                or (vert_gap > 0.65 * line_h and vert_gap > 5.0)
                or vert_gap < -2.0
            )

            if is_different_element:
                final_groups.append(current_subgroup)
                current_subgroup = [next_line]
            else:
                current_subgroup.append(next_line)

        if current_subgroup:
            final_groups.append(current_subgroup)

    return final_groups



def extract_page_elements(page: fitz.Page) -> list[dict]:
    """Trích xuất toàn bộ phần tử trên trang, bảo toàn cấu trúc ô bảng và tách khối văn bản ngoài bảng."""
    elements = []
    table_bboxes: list[fitz.Rect] = []

    try:
        table_finder = page.find_tables()
        for tab in getattr(table_finder, "tables", []):
            if not getattr(tab, "bbox", None):
                continue
            t_rect = fitz.Rect(tab.bbox)
            if (
                t_rect.is_empty
                or t_rect.is_infinite
                or t_rect.width <= 0
                or t_rect.height <= 0
            ):
                continue
            table_bboxes.append(t_rect)

            for cell_bbox in getattr(tab, "cells", []):
                if not cell_bbox:
                    continue
                c_rect = fitz.Rect(cell_bbox)
                if (
                    c_rect.is_empty
                    or c_rect.is_infinite
                    or c_rect.width <= 0
                    or c_rect.height <= 0
                ):
                    continue

                cell_dict = page.get_text("dict", clip=c_rect)
                cell_spans = []
                cell_lines = []
                for b in cell_dict.get("blocks", []):
                    if b.get("type") != 0:
                        continue
                    for l in b.get("lines", []):
                        valid_spans = [
                            s for s in l.get("spans", []) if s.get("text", "").strip()
                        ]
                        if valid_spans:
                            cell_spans.extend(valid_spans)
                            l_text = join_spans_into_line(
                                [
                                    {
                                        "text": s["text"],
                                        "bbox": fitz.Rect(s["bbox"]),
                                        "size": s["size"],
                                    }
                                    for s in valid_spans
                                ]
                            )
                            if l_text:
                                cell_lines.append(
                                    {"text": l_text, "bbox": fitz.Rect(l["bbox"])}
                                )

                if not cell_lines:
                    continue

                cell_text = join_lines_into_paragraph(cell_lines)
                if not cell_text:
                    continue

                sizes = [s["size"] for s in cell_spans]
                size = Counter(sizes).most_common(1)[0][0] if sizes else 9.0
                flags = cell_spans[0]["flags"] if cell_spans else 0
                font_name = cell_spans[0].get("font", "") if cell_spans else ""
                style = detect_font_style(flags, font_name)
                fontname = FONT_NAMES.get(style, "TNR")

                c_val = cell_spans[0].get("color", 0) if cell_spans else 0
                color = (
                    ((c_val >> 16) & 255) / 255,
                    ((c_val >> 8) & 255) / 255,
                    (c_val & 255) / 255,
                )

                # Căn lề ô bảng (ưu tiên căn phải cho số liệu/tiền tệ)
                align = fitz.TEXT_ALIGN_LEFT
                if cell_spans:
                    cell_mid = (c_rect.x0 + c_rect.x1) / 2
                    span_mid = (
                        cell_spans[0]["bbox"][0] + cell_spans[0]["bbox"][2]
                    ) / 2
                    span_x1 = max(s["bbox"][2] for s in cell_spans)
                    if abs(c_rect.x1 - span_x1) < 8 and c_rect.width > 30:
                        align = fitz.TEXT_ALIGN_RIGHT
                    elif abs(cell_mid - span_mid) < 12 and c_rect.width > 20:
                        align = fitz.TEXT_ALIGN_CENTER

                # Đệm lề an toàn theo tỷ lệ kích thước ô để chữ không tràn ra ngoài đường kẻ
                pad_x = min(1.5, c_rect.width * 0.08)
                pad_y = min(1.0, c_rect.height * 0.08)
                elements.append(
                    {
                        "is_table_cell": True,
                        "bbox": c_rect,
                        "insert_rect": fitz.Rect(
                            c_rect.x0 + pad_x,
                            c_rect.y0 + pad_y,
                            c_rect.x1 - pad_x,
                            c_rect.y1 - pad_y,
                        ),
                        "text": cell_text,
                        "size": round(size, 1),
                        "style": style,
                        "fontname": fontname,
                        "color": color,
                        "align": align,
                        "lineheight_ratio": 1.15,
                        "original_lines": cell_lines,
                    }
                )
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi nhận diện bảng: {e}")

    # Trích xuất các khối chữ ngoài bảng
    dict_data = page.get_text("dict", sort=True)
    for block in dict_data.get("blocks", []):
        if block.get("type") != 0:
            continue

        b_rect = fitz.Rect(block["bbox"])
        if (
            b_rect.is_empty
            or b_rect.is_infinite
            or b_rect.width <= 0
            or b_rect.height <= 0
        ):
            continue

        # Bỏ qua nếu block nằm trọn trong bảng (>95% diện tích)
        if any(
            (b_rect & t_rect).get_area() > 0.95 * b_rect.get_area()
            for t_rect in table_bboxes
        ):
            continue

        block_lines = []
        for line in block.get("lines", []):
            line_bbox = fitz.Rect(line["bbox"])
            if (
                line_bbox.is_empty
                or line_bbox.is_infinite
                or line_bbox.width <= 0
                or line_bbox.height <= 0
            ):
                continue

            # Phân định quyền sở hữu theo từng dòng (Line-level ownership):
            # Nếu >50% diện tích dòng nằm trong bảng -> dòng đó thuộc về ô bảng (đã xử lý ở Bước 1)
            line_area = line_bbox.get_area()
            is_line_in_table = False
            for t_rect in table_bboxes:
                inter = line_bbox & t_rect
                if inter.get_area() > 0.5 * line_area:
                    is_line_in_table = True
                    break

            if is_line_in_table:
                continue

            valid_spans = []
            for span in line.get("spans", []):
                stext = span.get("text", "")
                if not stext.strip():
                    continue
                flags = span["flags"]
                c_val = span.get("color", 0)
                color = (
                    ((c_val >> 16) & 255) / 255,
                    ((c_val >> 8) & 255) / 255,
                    (c_val & 255) / 255,
                )
                valid_spans.append(
                    {
                        "text": stext,
                        "bbox": fitz.Rect(span["bbox"]),
                        "size": span["size"],
                        "flags": flags,
                        "style": detect_font_style(flags, span.get("font", "")),
                        "color": color,
                    }
                )

            if not valid_spans:
                continue

            # Tách dòng nếu có khoảng cách ngang bất thường (bảng không viền, 2 cột cùng dòng)
            segments = []
            current_seg = [valid_spans[0]]
            for s in valid_spans[1:]:
                prev_s = current_seg[-1]
                gap = s["bbox"].x0 - prev_s["bbox"].x1
                avg_size = max(prev_s["size"], s["size"])
                if gap > 35 and gap > 3.5 * avg_size:
                    segments.append(current_seg)
                    current_seg = [s]
                else:
                    current_seg.append(s)
            segments.append(current_seg)

            for seg in segments:
                line_text = join_spans_into_line(seg)
                if line_text:
                    seg_rect = fitz.Rect(
                        min(s["bbox"].x0 for s in seg),
                        min(s["bbox"].y0 for s in seg),
                        max(s["bbox"].x1 for s in seg),
                        max(s["bbox"].y1 for s in seg),
                    )
                    block_lines.append(
                        {
                            "text": line_text,
                            "bbox": seg_rect,
                            "spans": seg,
                        }
                    )

        if not block_lines:
            continue

        subgroups = group_lines_into_elements(block_lines)

        for sg in subgroups:
            sg_bbox = fitz.Rect(
                min(l["bbox"].x0 for l in sg),
                min(l["bbox"].y0 for l in sg),
                max(l["bbox"].x1 for l in sg),
                max(l["bbox"].y1 for l in sg),
            )
            sg_spans = []
            for l in sg:
                sg_spans.extend(l.get("spans", []))

            sizes = [s["size"] for s in sg_spans]
            size = Counter(sizes).most_common(1)[0][0] if sizes else 10.0
            style_counter = Counter(s["style"] for s in sg_spans)
            style = style_counter.most_common(1)[0][0] if style_counter else "regular"
            fontname = FONT_NAMES.get(style, "TNR")
            color = sg_spans[0]["color"] if sg_spans else (0, 0, 0)

            lineheight_ratio = 1.2
            if len(sg) > 1 and size > 0:
                y_diffs = [
                    sg[i + 1]["bbox"].y0 - sg[i]["bbox"].y0
                    for i in range(len(sg) - 1)
                ]
                pos_diffs = [d for d in y_diffs if d > 0]
                if pos_diffs:
                    lineheight_ratio = statistics.median(pos_diffs) / size
            lineheight_ratio = max(1.05, min(lineheight_ratio, 1.45))

            source_text = (
                join_lines_into_paragraph(sg)
                if len(sg) > 1
                else clean_and_normalize_text(sg[0]["text"])
            )

            align = fitz.TEXT_ALIGN_LEFT
            page_w = page.rect.width
            if len(sg) == 1:
                item_mid = (sg_bbox.x0 + sg_bbox.x1) / 2
                if abs(page_w / 2 - item_mid) < 30 and sg_bbox.width < page_w * 0.7:
                    align = fitz.TEXT_ALIGN_CENTER

            elements.append(
                {
                    "is_table_cell": False,
                    "bbox": sg_bbox,
                    "insert_rect": sg_bbox,
                    "text": source_text,
                    "size": round(size, 1),
                    "style": style,
                    "fontname": fontname,
                    "color": color,
                    "align": align,
                    "lineheight_ratio": lineheight_ratio,
                    "original_lines": sg,
                }
            )

    # Sắp xếp các phần tử theo thứ tự đọc tự nhiên của trang (Top-to-Bottom, Left-to-Right)
    # Bảo đảm tiêu đề, đoạn văn và các ô bảng nằm đúng vị trí xuất hiện trong luồng tài liệu
    if elements:
        page_w = page.rect.width
        mid_x = page_w / 2

        # Kiểm tra xem trang có cấu trúc 2 cột rõ rệt hay không
        left_elems = [e for e in elements if e["bbox"].x1 < mid_x + 15 and e["bbox"].width < page_w * 0.6]
        right_elems = [e for e in elements if e["bbox"].x0 > mid_x - 15 and e["bbox"].width < page_w * 0.6]
        has_two_columns = len(left_elems) >= 3 and len(right_elems) >= 3

        if has_two_columns:
            # Trang 2 cột: các khối full-width xếp theo Y, khối trong cột ưu tiên cột trái rồi cột phải
            def col_sort_key(e):
                b = e["bbox"]
                is_full_width = b.width > page_w * 0.65
                if is_full_width:
                    return (round(b.y0 / 20) * 20, 0, b.y0, b.x0)
                col_id = 0 if b.x0 < mid_x else 1
                return (round(b.y0 / 120) * 120, col_id, b.y0, b.x0)

            elements.sort(key=col_sort_key)
        else:
            # Trang 1 cột hoặc bố cục thông thường: sắp xếp từ trên xuống dưới, từ trái sang phải
            elements.sort(key=lambda e: (round(e["bbox"].y0 / 4) * 4, e["bbox"].x0))

    return elements

