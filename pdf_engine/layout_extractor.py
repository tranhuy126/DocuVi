from collections import Counter
import re
import statistics

import pymupdf as fitz

from .config import FONT_NAMES, detect_font_style
from .renderer import get_page_bounds
from .text_processor import (
    clean_and_normalize_text,
    join_lines_into_paragraph,
    join_spans_into_line,
)

_MARKER_BODY = (
    r"(?:[•●▪◦‣⁃\uFFFD*\uF0B7\uF0A7\uF0D8·○■□✓☑–—-]"
    r"|\d+(?:\.\d+)*[.)]"
    r"|\(\d{1,2}\)"
    r"|\([A-Za-z]\)"
    r"|[A-Za-z][.)])"
)
LIST_ITEM_PATTERN = re.compile(rf"^\s*(?P<marker>{_MARKER_BODY})\s+")
STANDALONE_LIST_MARKER_PATTERN = re.compile(rf"^\s*(?P<marker>{_MARKER_BODY})\s*$")
BARE_NUMBER_MARKER_PATTERN = re.compile(r"^\s*(?P<marker>\d{1,2})\s*$")


def _rgb_int_to_tuple(color_int: int) -> tuple[float, float, float]:
    red = ((color_int >> 16) & 255) / 255
    green = ((color_int >> 8) & 255) / 255
    blue = (color_int & 255) / 255
    return (red, green, blue)


def split_horizontal_lines(raw_lines: list[dict]) -> list[list[dict]]:
    if len(raw_lines) <= 1:
        return [raw_lines]

    has_side_by_side_lines = False
    for first_index in range(len(raw_lines)):
        first_bbox = raw_lines[first_index]["bbox"]
        for second_index in range(first_index + 1, len(raw_lines)):
            second_bbox = raw_lines[second_index]["bbox"]
            vertical_overlap = min(first_bbox.y1, second_bbox.y1) - max(first_bbox.y0, second_bbox.y0)
            min_line_height = min(first_bbox.height, second_bbox.height)
            horizontal_offset = abs(first_bbox.x0 - second_bbox.x0)

            if vertical_overlap > 0.4 * min_line_height and horizontal_offset > 8:
                has_side_by_side_lines = True
                break
        if has_side_by_side_lines:
            break

    if not has_side_by_side_lines:
        return [raw_lines]

    clusters: list[list[dict]] = []
    for line in raw_lines:
        line_x0 = line["bbox"].x0
        line_x1 = line["bbox"].x1

        for cluster in clusters:
            cluster_x0 = min(cluster_line["bbox"].x0 for cluster_line in cluster)
            cluster_x1 = max(cluster_line["bbox"].x1 for cluster_line in cluster)
            horizontal_overlap = min(cluster_x1, line_x1) - max(cluster_x0, line_x0)
            min_width = min(cluster_x1 - cluster_x0, line_x1 - line_x0)

            is_horizontally_aligned = (
                (min_width > 0 and horizontal_overlap > 0.3 * min_width)
                or (abs(cluster_x0 - line_x0) < 6 and abs(cluster_x1 - line_x1) < 6)
            )
            if is_horizontally_aligned:
                cluster.append(line)
                break
        else:
            clusters.append([line])

    clusters.sort(key=lambda cluster: min(cluster_line["bbox"].x0 for cluster_line in cluster))
    for cluster in clusters:
        cluster.sort(key=lambda cluster_line: cluster_line["bbox"].y0)
    return clusters


def _bold_ratio(line: dict) -> float:
    """Tỷ lệ ký tự (không tính khoảng trắng) thuộc span đậm trong một dòng."""
    spans = line.get("spans", [])
    total_chars = sum(len(span.get("text", "").strip()) for span in spans)
    bold_chars = sum(
        len(span.get("text", "").strip())
        for span in spans
        if span.get("style", "") in ("bold", "bold_italic")
    )
    return bold_chars / total_chars if total_chars else 0.0


def group_lines_into_elements(raw_lines: list[dict]) -> list[list[dict]]:
    merged_lines = []
    consumed_indices = set()

    for marker_index, marker_line in enumerate(raw_lines):
        if marker_index in consumed_indices:
            continue

        line_text = marker_line.get("text", "")
        marker_match = (
            STANDALONE_LIST_MARKER_PATTERN.match(line_text)
            or BARE_NUMBER_MARKER_PATTERN.match(line_text)
        )
        if not marker_match:
            merged_lines.append(marker_line)
            continue

        marker_bbox = marker_line["bbox"]
        marker_font_size = max(
            (span.get("size", 0) for span in marker_line.get("spans", [])),
            default=10.0,
        )

        candidate_content_lines = []
        for candidate_index, candidate_line in enumerate(raw_lines):
            if candidate_index == marker_index or candidate_index in consumed_indices:
                continue

            candidate_bbox = candidate_line["bbox"]
            vertical_overlap = min(marker_bbox.y1, candidate_bbox.y1) - max(marker_bbox.y0, candidate_bbox.y0)
            min_line_height = min(marker_bbox.height, candidate_bbox.height)
            horizontal_distance = candidate_bbox.x0 - marker_bbox.x1

            if (
                vertical_overlap >= 0.5 * min_line_height
                and 0 <= horizontal_distance <= max(14.0, marker_font_size * 2)
            ):
                candidate_content_lines.append((horizontal_distance, candidate_index, candidate_line))

        if not candidate_content_lines:
            merged_lines.append(marker_line)
            continue

        _, chosen_index, content_line = min(candidate_content_lines, key=lambda item: item[0])
        content_bbox = content_line["bbox"]
        marker_raw_bbox = marker_line.get("raw_bbox", marker_bbox)
        content_raw_bbox = content_line.get("raw_bbox", content_bbox)

        marker_token = marker_match.group("marker")
        normalized_marker = f"{marker_token}." if marker_token.isdigit() else marker_token

        combined_bbox = fitz.Rect(
            min(marker_bbox.x0, content_bbox.x0),
            min(marker_bbox.y0, content_bbox.y0),
            max(marker_bbox.x1, content_bbox.x1),
            max(marker_bbox.y1, content_bbox.y1),
        )
        combined_raw_bbox = fitz.Rect(
            min(marker_raw_bbox.x0, content_raw_bbox.x0),
            min(marker_raw_bbox.y0, content_raw_bbox.y0),
            max(marker_raw_bbox.x1, content_raw_bbox.x1),
            max(marker_raw_bbox.y1, content_raw_bbox.y1),
        )

        merged_lines.append({
            **marker_line,
            "text": f"{normalized_marker} {content_line['text']}",
            "bbox": combined_bbox,
            "raw_bbox": combined_raw_bbox,
            "spans": marker_line.get("spans", []) + content_line.get("spans", []),
        })
        consumed_indices.add(chosen_index)

    sorted_lines = sorted(merged_lines, key=lambda line: (round(line["bbox"].y0 / 4) * 4, line["bbox"].x0))
    if len(sorted_lines) <= 1:
        return [sorted_lines]

    horizontal_clusters = split_horizontal_lines(sorted_lines)
    is_numeric_two_column_grid = False
    if len(horizontal_clusters) >= 2:
        for cluster in horizontal_clusters:
            non_alpha_line_count = sum(
                1
                for line in cluster
                if not any(char.isalpha() for char in line.get("text", ""))
            )
            if non_alpha_line_count >= 2 and (non_alpha_line_count / len(cluster)) >= 0.30:
                is_numeric_two_column_grid = True
                break

    if len(horizontal_clusters) >= 3 or is_numeric_two_column_grid:
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

        segments = []
        current_plain_lines = []
        current_list_item_lines = None

        for line in cluster:
            if LIST_ITEM_PATTERN.match(line.get("text", "")):
                if current_plain_lines:
                    segments.append((False, current_plain_lines))
                    current_plain_lines = []
                current_list_item_lines = [line]
                segments.append((True, current_list_item_lines))
                continue

            if current_list_item_lines:
                previous_bbox = current_list_item_lines[-1]["bbox"]
                current_bbox = line["bbox"]
                first_item_line = current_list_item_lines[0]
                first_item_bbox = first_item_line["bbox"]
                first_item_font_size = max(
                    (span.get("size", 0) for span in first_item_line.get("spans", [])),
                    default=10.0,
                )
                marker_match = LIST_ITEM_PATTERN.match(first_item_line.get("text", ""))
                if marker_match:
                    marker_width = len(marker_match.group("marker")) * first_item_font_size * 0.55
                else:
                    marker_width = first_item_font_size

                vertical_gap = current_bbox.y0 - previous_bbox.y1
                min_indent_x = first_item_bbox.x0 + max(first_item_font_size * 0.7, marker_width)
                is_continuation_line = (
                    vertical_gap <= max(previous_bbox.height * 0.8, 3.0)
                    and current_bbox.x0 >= min_indent_x
                )
                if is_continuation_line:
                    current_list_item_lines.append(line)
                    continue
                current_list_item_lines = None

            current_plain_lines.append(line)

        if current_plain_lines:
            segments.append((False, current_plain_lines))

        for is_list_item, segment_lines in segments:
            if is_list_item:
                final_groups.append(segment_lines)
                continue

            current_subgroup = [segment_lines[0]]
            for next_line in segment_lines[1:]:
                previous_line = current_subgroup[-1]
                previous_bbox = previous_line["bbox"]
                next_bbox = next_line["bbox"]

                previous_sizes = [span["size"] for span in previous_line.get("spans", []) if span.get("size")]
                next_sizes = [span["size"] for span in next_line.get("spans", []) if span.get("size")]
                previous_median_size = statistics.median(previous_sizes) if previous_sizes else 10.0
                next_median_size = statistics.median(next_sizes) if next_sizes else 10.0

                size_difference = abs(previous_median_size - next_median_size)
                size_ratio = max(previous_median_size, next_median_size) / max(
                    min(previous_median_size, next_median_size), 0.1
                )
                previous_bold_ratio = _bold_ratio(previous_line)
                next_bold_ratio = _bold_ratio(next_line)

                reference_line_height = max(
                    previous_bbox.height,
                    next_bbox.height,
                    max(previous_median_size, next_median_size),
                )
                vertical_gap = next_bbox.y0 - previous_bbox.y1

                previous_origins_y = [
                    span["origin"][1]
                    for span in previous_line.get("spans", [])
                    if span.get("origin") and len(span["origin"]) >= 2
                ]
                next_origins_y = [
                    span["origin"][1]
                    for span in next_line.get("spans", [])
                    if span.get("origin") and len(span["origin"]) >= 2
                ]

                if previous_origins_y and next_origins_y:
                    baseline_pitch = statistics.median(next_origins_y) - statistics.median(previous_origins_y)
                    max_font_size = max(previous_median_size, next_median_size, 1.0)
                    has_abnormal_vertical_spacing = (
                        baseline_pitch < 0.55 * max_font_size
                        or baseline_pitch > 1.75 * max_font_size
                    )
                else:
                    has_abnormal_vertical_spacing = vertical_gap < -2.0

                has_bold_transition = (
                    (previous_bold_ratio >= 0.95 and next_bold_ratio <= 0.05)
                    or (previous_bold_ratio <= 0.05 and next_bold_ratio >= 0.95)
                )
                has_large_vertical_gap = vertical_gap > 0.65 * reference_line_height and vertical_gap > 5.0

                should_split_subgroup = (
                    size_difference >= 1.5
                    or size_ratio >= 1.2
                    or has_bold_transition
                    or has_large_vertical_gap
                    or has_abnormal_vertical_spacing
                )
                if should_split_subgroup:
                    final_groups.append(current_subgroup)
                    current_subgroup = [next_line]
                else:
                    current_subgroup.append(next_line)

            if current_subgroup:
                final_groups.append(current_subgroup)

    return final_groups


def _collect_visible_span_rects(page_dict_data: dict) -> list[fitz.Rect]:
    """Thu thập toàn bộ các khung chữ (span bbox) hiển thị hợp lệ trên trang."""
    visible_span_rects = []
    for block in page_dict_data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("alpha", 255) == 0 or not span.get("text", "").strip():
                    continue
                span_rect = fitz.Rect(span["bbox"])
                if not span_rect.is_empty and not span_rect.is_infinite:
                    visible_span_rects.append(span_rect)
    return visible_span_rects


def _merge_spanning_table_cells(
    filtered_cells: list[fitz.Rect],
    all_visible_span_rects: list[fitz.Rect],
) -> list[fitz.Rect]:
    """Hợp nhất các ô bảng nằm cùng hàng nếu có một span văn bản trải dài cắt qua nhiều ô."""
    has_merged_spanning_cells = True
    while has_merged_spanning_cells:
        has_merged_spanning_cells = False
        for span_rect in all_visible_span_rects:
            if span_rect.is_empty or span_rect.width <= 4:
                continue

            overlapping_cell_indices = []
            for cell_index, cell_rect in enumerate(filtered_cells):
                vertical_overlap = min(span_rect.y1, cell_rect.y1) - max(span_rect.y0, cell_rect.y0)
                horizontal_overlap = min(span_rect.x1, cell_rect.x1) - max(span_rect.x0, cell_rect.x0)
                if (
                    vertical_overlap >= 0.5 * span_rect.height
                    and horizontal_overlap >= max(3.0, 0.12 * span_rect.width)
                ):
                    overlapping_cell_indices.append(cell_index)

            if len(overlapping_cell_indices) >= 2:
                base_cell = filtered_cells[overlapping_cell_indices[0]]
                same_row_indices = [
                    cell_index
                    for cell_index in overlapping_cell_indices
                    if abs(filtered_cells[cell_index].y0 - base_cell.y0) < 4.0
                    and abs(filtered_cells[cell_index].y1 - base_cell.y1) < 4.0
                ]
                if len(same_row_indices) >= 2:
                    merged_cell_rect = fitz.Rect(filtered_cells[same_row_indices[0]])
                    for cell_index in same_row_indices[1:]:
                        merged_cell_rect |= filtered_cells[cell_index]
                    filtered_cells[same_row_indices[0]] = merged_cell_rect
                    for cell_index in sorted(same_row_indices[1:], reverse=True):
                        filtered_cells.pop(cell_index)
                    has_merged_spanning_cells = True
                    break

    return filtered_cells


def _extract_bordered_table_elements(
    page: fitz.Page,
    page_dict_data: dict,
    all_visible_span_rects: list[fitz.Rect],
) -> tuple[list[dict], list[fitz.Rect], set[int]]:
    """Dò tìm bảng biểu có viền trên trang và trích xuất từng ô bảng thành các phần tử độc lập."""
    table_elements = []
    table_cell_bboxes = []
    consumed_line_ids = set()

    try:
        detected_tables = getattr(page.find_tables(), "tables", [])
        for table in detected_tables:
            raw_table_bbox = getattr(table, "bbox", None)
            if not raw_table_bbox:
                continue

            table_rect = fitz.Rect(raw_table_bbox)
            if (
                table_rect.is_empty
                or table_rect.is_infinite
                or table_rect.width <= 0
                or table_rect.height <= 0
            ):
                continue

            raw_cell_rects = []
            for raw_cell_bbox in getattr(table, "cells", []):
                if not raw_cell_bbox:
                    continue
                cell_rect = fitz.Rect(raw_cell_bbox)
                is_valid_cell_rect = (
                    not cell_rect.is_empty
                    and not cell_rect.is_infinite
                    and cell_rect.width > 0
                    and cell_rect.height > 0
                )
                if is_valid_cell_rect and cell_rect not in raw_cell_rects:
                    raw_cell_rects.append(cell_rect)

            filtered_cells = []
            for cell_index, cell_rect in enumerate(raw_cell_rects):
                is_subsumed_by_larger_cell = any(
                    other_index != cell_index
                    and other_rect.get_area() > cell_rect.get_area()
                    and (cell_rect & other_rect).get_area() >= 0.98 * cell_rect.get_area()
                    for other_index, other_rect in enumerate(raw_cell_rects)
                )
                if not is_subsumed_by_larger_cell:
                    filtered_cells.append(cell_rect)

            filtered_cells = _merge_spanning_table_cells(filtered_cells, all_visible_span_rects)
            table_cell_bboxes.extend(filtered_cells)

            cell_assigned_lines: dict[int, list[dict]] = {
                cell_index: [] for cell_index in range(len(filtered_cells))
            }

            for block in page_dict_data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    if id(line) in consumed_line_ids:
                        continue
                    line_rect = fitz.Rect(line["bbox"])
                    if (
                        line_rect.is_empty
                        or line_rect.is_infinite
                        or line_rect.get_area() <= 0
                        or not line_rect.intersects(table_rect)
                    ):
                        continue

                    best_cell_index, best_overlap_area = max(
                        (
                            (cell_index, (line_rect & cell_rect).get_area())
                            for cell_index, cell_rect in enumerate(filtered_cells)
                        ),
                        key=lambda item: item[1],
                        default=(-1, 0.0),
                    )
                    if best_cell_index >= 0 and best_overlap_area >= 0.5 * line_rect.get_area():
                        cell_assigned_lines[best_cell_index].append(line)
                        consumed_line_ids.add(id(line))

            for cell_index, cell_rect in enumerate(filtered_cells):
                cell_spans = []
                cell_lines = []
                for line in cell_assigned_lines[cell_index]:
                    valid_spans = [
                        {
                            **span,
                            "style": detect_font_style(span.get("flags", 0), span.get("font", "")),
                        }
                        for span in line.get("spans", [])
                        if span.get("alpha", 255) != 0 and span.get("text", "").strip()
                    ]
                    if not valid_spans:
                        continue

                    cell_spans.extend(valid_spans)
                    span_items_for_line = [
                        {
                            "text": span["text"],
                            "bbox": fitz.Rect(span["bbox"]),
                            "size": span["size"],
                        }
                        for span in valid_spans
                    ]
                    line_text = join_spans_into_line(span_items_for_line)
                    if line_text:
                        tight_line_bbox = fitz.Rect(
                            min(span["bbox"][0] for span in valid_spans),
                            min(span["bbox"][1] for span in valid_spans),
                            max(span["bbox"][2] for span in valid_spans),
                            max(span["bbox"][3] for span in valid_spans),
                        )
                        cell_lines.append({
                            "text": line_text,
                            "bbox": tight_line_bbox,
                            "raw_bbox": fitz.Rect(line["bbox"]),
                            "spans": valid_spans,
                        })

                if not cell_lines:
                    continue

                cell_groups = group_lines_into_elements(cell_lines)
                contains_list = any(LIST_ITEM_PATTERN.match(group[0]["text"]) for group in cell_groups)
                if contains_list:
                    cell_groups = sorted(cell_groups, key=lambda group: group[0]["bbox"].y0)
                else:
                    cell_groups = [cell_lines]

                if not any(join_lines_into_paragraph(group) for group in cell_groups):
                    continue

                span_sizes = [span["size"] for span in cell_spans]
                dominant_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 9.0
                first_span = cell_spans[0] if cell_spans else {}
                dominant_style = detect_font_style(
                    first_span.get("flags", 0),
                    first_span.get("font", ""),
                )
                dominant_color = _rgb_int_to_tuple(first_span.get("color", 0))

                text_align = fitz.TEXT_ALIGN_LEFT
                if cell_spans:
                    left_gap = min(span["bbox"][0] for span in cell_spans) - cell_rect.x0
                    right_gap = cell_rect.x1 - max(span["bbox"][2] for span in cell_spans)
                    if right_gap < 8 and left_gap > max(8.0, 0.15 * cell_rect.width) and cell_rect.width > 30:
                        text_align = fitz.TEXT_ALIGN_RIGHT
                    elif left_gap > 8 and right_gap > 8 and abs(left_gap - right_gap) < 12 and cell_rect.width > 20:
                        text_align = fitz.TEXT_ALIGN_CENTER

                pad_x = min(1.5, cell_rect.width * 0.08)
                pad_y = min(1.0, cell_rect.height * 0.08)

                for group_index, group in enumerate(cell_groups):
                    cell_text = join_lines_into_paragraph(group)
                    if not cell_text:
                        continue

                    group_bbox = fitz.Rect(
                        min(line["bbox"].x0 for line in group),
                        min(line["bbox"].y0 for line in group),
                        max(line["bbox"].x1 for line in group),
                        max(line["bbox"].y1 for line in group),
                    )
                    list_match = LIST_ITEM_PATTERN.match(group[0]["text"])
                    group_size = dominant_size
                    group_style = dominant_style
                    group_color = dominant_color

                    if contains_list:
                        group_spans = [span for line in group for span in line.get("spans", [])]
                        if group_spans:
                            longest_span = max(group_spans, key=lambda span: len(span.get("text", "")))
                            group_size = longest_span["size"]
                            group_style = detect_font_style(longest_span["flags"], longest_span.get("font", ""))
                            group_color = _rgb_int_to_tuple(longest_span.get("color", 0))

                        next_group_top = (
                            cell_groups[group_index + 1][0]["bbox"].y0
                            if group_index + 1 < len(cell_groups)
                            else cell_rect.y1
                        )
                        insert_rect = fitz.Rect(
                            cell_rect.x0 + pad_x,
                            max(cell_rect.y0 + pad_y, group_bbox.y0 - pad_y),
                            cell_rect.x1 - pad_x,
                            max(group_bbox.y1 + pad_y, min(cell_rect.y1 - pad_y, next_group_top - 1.0)),
                        )
                    else:
                        insert_rect = fitz.Rect(
                            cell_rect.x0 + pad_x,
                            cell_rect.y0 + pad_y,
                            cell_rect.x1 - pad_x,
                            cell_rect.y1 - pad_y,
                        )

                    element = {
                        "is_table_cell": True,
                        "bbox": group_bbox,
                        "insert_rect": insert_rect,
                        "text": cell_text,
                        "size": round(group_size, 1),
                        "style": group_style,
                        "fontname": FONT_NAMES.get(group_style, "TNR"),
                        "color": group_color,
                        "align": text_align,
                        "lineheight_ratio": 1.15,
                        "original_lines": group,
                    }
                    if list_match:
                        element["list_marker"] = list_match.group("marker")
                        first_line_without_marker = dict(
                            group[0],
                            text=group[0]["text"][list_match.end() :],
                        )
                        element["translation_text"] = join_lines_into_paragraph(
                            [first_line_without_marker, *group[1:]]
                        )
                    table_elements.append(element)
    except Exception as table_error:
        print(f"[CẢNH BÁO] Lỗi nhận diện bảng: {table_error}")

    return table_elements, table_cell_bboxes, consumed_line_ids


def _extract_text_block_elements(
    page_dict_data: dict,
    table_cell_bboxes: list[fitz.Rect],
    consumed_line_ids: set[int],
    page_width: float,
) -> list[dict]:
    """Trích xuất các khối văn bản thông thường (ngoài bảng có viền) và phân nhóm thành đoạn/danh sách."""
    text_elements = []
    monospace_keywords = ("courier", "consolas", "menlo", "monaco", "monospace", "firacode", "jetbrains")
    proportional_keywords = ("times", "arial", "helvetica", "calibri", "cambria", "georgia", "serif", "sans")

    for block in page_dict_data.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_rect = fitz.Rect(block["bbox"])
        if block_rect.is_empty or block_rect.is_infinite or block_rect.width <= 0 or block_rect.height <= 0:
            continue

        block_lines = []
        for line in block.get("lines", []):
            if id(line) in consumed_line_ids:
                continue
            line_bbox = fitz.Rect(line["bbox"])
            if line_bbox.is_empty or line_bbox.is_infinite or line_bbox.width <= 0 or line_bbox.height <= 0:
                continue

            is_inside_table_cell = any(
                (line_bbox & cell_bbox).get_area() >= 0.5 * line_bbox.get_area()
                for cell_bbox in table_cell_bboxes
            )
            if is_inside_table_cell:
                continue

            line_direction = line.get("dir", (1.0, 0.0))
            is_rotated_line = abs(line_direction[0] - 1.0) > 0.05 or abs(line_direction[1]) > 0.05

            valid_spans = []
            for span in line.get("spans", []):
                if span.get("alpha", 255) == 0 or not span.get("text", "").strip():
                    continue

                span_flags = span["flags"]
                font_name = span.get("font", "")
                font_name_lower = (font_name or "").lower()

                has_mono_name = any(keyword in font_name_lower for keyword in monospace_keywords)
                has_mono_flag = bool(span_flags & fitz.TEXT_FONT_MONOSPACED) and not any(
                    keyword in font_name_lower for keyword in proportional_keywords
                )
                is_monospace = has_mono_name or has_mono_flag

                valid_spans.append({
                    "text": span["text"],
                    "bbox": fitz.Rect(span["bbox"]),
                    "size": span["size"],
                    "flags": span_flags,
                    "font": font_name,
                    "origin": span.get("origin"),
                    "is_monospace": is_monospace,
                    "style": detect_font_style(span_flags, font_name),
                    "color": _rgb_int_to_tuple(span.get("color", 0)),
                })

            if not valid_spans:
                continue

            span_segments = []
            current_segment = [valid_spans[0]]
            for span in valid_spans[1:]:
                previous_span = current_segment[-1]
                horizontal_gap = span["bbox"].x0 - previous_span["bbox"].x1
                if horizontal_gap > 35 and horizontal_gap > 3.5 * max(previous_span["size"], span["size"]):
                    span_segments.append(current_segment)
                    current_segment = [span]
                else:
                    current_segment.append(span)
            span_segments.append(current_segment)

            for segment in span_segments:
                line_text = join_spans_into_line(segment)
                if line_text:
                    segment_rect = fitz.Rect(
                        min(span["bbox"].x0 for span in segment),
                        min(span["bbox"].y0 for span in segment),
                        max(span["bbox"].x1 for span in segment),
                        max(span["bbox"].y1 for span in segment),
                    )
                    raw_line_bbox = fitz.Rect(line["bbox"]) if len(span_segments) == 1 else segment_rect
                    block_lines.append({
                        "text": line_text,
                        "bbox": segment_rect,
                        "raw_bbox": raw_line_bbox,
                        "spans": segment,
                        "is_rotated": is_rotated_line,
                    })

        if not block_lines:
            continue

        for subgroup in group_lines_into_elements(block_lines):
            subgroup_bbox = fitz.Rect(
                min(line["bbox"].x0 for line in subgroup),
                min(line["bbox"].y0 for line in subgroup),
                max(line["bbox"].x1 for line in subgroup),
                max(line["bbox"].y1 for line in subgroup),
            )
            subgroup_spans = [span for line in subgroup for span in line.get("spans", [])]
            span_sizes = [span["size"] for span in subgroup_spans]
            dominant_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 10.0

            style_counter = Counter(span["style"] for span in subgroup_spans)
            dominant_style = style_counter.most_common(1)[0][0] if style_counter else "regular"

            total_chars = sum(len(span["text"].strip()) for span in subgroup_spans)
            monospace_chars = sum(
                len(span["text"].strip())
                for span in subgroup_spans
                if span.get("is_monospace")
            )

            lineheight_ratio = 1.2
            if len(subgroup) > 1 and dominant_size > 0:
                positive_y_diffs = [
                    subgroup[index + 1]["bbox"].y0 - subgroup[index]["bbox"].y0
                    for index in range(len(subgroup) - 1)
                    if subgroup[index + 1]["bbox"].y0 > subgroup[index]["bbox"].y0
                ]
                if positive_y_diffs:
                    lineheight_ratio = statistics.median(positive_y_diffs) / dominant_size
            lineheight_ratio = max(1.05, min(lineheight_ratio, 1.45))

            if len(subgroup) > 1:
                source_text = join_lines_into_paragraph(subgroup)
            else:
                source_text = clean_and_normalize_text(subgroup[0]["text"])

            list_match = LIST_ITEM_PATTERN.match(subgroup[0]["text"])
            subgroup_center_x = (subgroup_bbox.x0 + subgroup_bbox.x1) / 2
            is_centered_single_line = (
                len(subgroup) == 1
                and not list_match
                and abs(page_width / 2 - subgroup_center_x) < 30
                and subgroup_bbox.width < page_width * 0.7
            )
            text_align = fitz.TEXT_ALIGN_CENTER if is_centered_single_line else fitz.TEXT_ALIGN_LEFT

            element = {
                "is_table_cell": False,
                "is_code": total_chars > 0 and (monospace_chars / total_chars) >= 0.8,
                "is_rotated": any(line.get("is_rotated") for line in subgroup),
                "bbox": subgroup_bbox,
                "insert_rect": subgroup_bbox,
                "text": source_text,
                "size": round(dominant_size, 1),
                "style": dominant_style,
                "fontname": FONT_NAMES.get(dominant_style, "TNR"),
                "color": subgroup_spans[0]["color"] if subgroup_spans else (0, 0, 0),
                "align": text_align,
                "lineheight_ratio": lineheight_ratio,
                "original_lines": subgroup,
            }
            if list_match:
                element["list_marker"] = list_match.group("marker")
                first_line_without_marker = dict(
                    subgroup[0],
                    text=subgroup[0]["text"][list_match.end() :],
                )
                element["translation_text"] = join_lines_into_paragraph(
                    [first_line_without_marker, *subgroup[1:]]
                )
            text_elements.append(element)

    return text_elements


def _sort_and_normalize_elements(elements: list[dict], page_width: float) -> None:
    """Sắp xếp các phần tử theo thứ tự đọc tự nhiên (1 cột hoặc 2 cột) và chuẩn hóa căn lề."""
    if not elements:
        return

    page_center_x = page_width / 2
    left_column_count = sum(
        1
        for element in elements
        if not element.get("is_table_cell")
        and element["bbox"].x1 < page_center_x + 15
        and element["bbox"].width < page_width * 0.6
    )
    right_column_count = sum(
        1
        for element in elements
        if not element.get("is_table_cell")
        and element["bbox"].x0 > page_center_x - 15
        and element["bbox"].width < page_width * 0.6
    )

    if left_column_count >= 3 and right_column_count >= 3:
        spanning_y_ranges = sorted(
            (element["bbox"].y0, element["bbox"].y1)
            for element in elements
            if element["bbox"].width > page_width * 0.65
        )
        elements.sort(
            key=lambda element: (
                sum(1 for _, span_y1 in spanning_y_ranges if span_y1 <= element["bbox"].y0 + 2.0),
                0
                if element["bbox"].width > page_width * 0.65
                else (1 if element["bbox"].x0 < page_center_x else 2),
                round(element["bbox"].y0 / 4) * 4,
                element["bbox"].x0,
            )
        )
    else:
        elements.sort(key=lambda element: (round(element["bbox"].y0 / 4) * 4, element["bbox"].x0))

    # Dòng đơn gần giữa trang bị nhận là "căn giữa" nhưng cùng mép trái với >= 2 phần tử khác
    # thì thực chất là một cột căn trái (ví dụ các gạch đầu dòng) -> đặt lại căn trái.
    for element in elements:
        if element["align"] == fitz.TEXT_ALIGN_CENTER and not element.get("is_table_cell"):
            same_left_margin_count = sum(
                1
                for other_element in elements
                if other_element is not element and abs(other_element["bbox"].x0 - element["bbox"].x0) < 2.0
            )
            if same_left_margin_count >= 2:
                element["align"] = fitz.TEXT_ALIGN_LEFT


def extract_page_elements(page: fitz.Page) -> list[dict]:
    """Điều phối trích xuất toàn bộ các phần tử bảng biểu và khối văn bản trên một trang PDF."""
    page_width = get_page_bounds(page).width
    page_dict_data = page.get_text("dict", sort=True)
    all_visible_span_rects = _collect_visible_span_rects(page_dict_data)

    table_elements, table_cell_bboxes, consumed_line_ids = _extract_bordered_table_elements(
        page,
        page_dict_data,
        all_visible_span_rects,
    )
    text_elements = _extract_text_block_elements(
        page_dict_data,
        table_cell_bboxes,
        consumed_line_ids,
        page_width,
    )

    elements = table_elements + text_elements
    _sort_and_normalize_elements(elements, page_width)
    return elements


