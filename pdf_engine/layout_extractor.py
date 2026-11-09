from collections import Counter
import re
import statistics

import pymupdf as fitz

from .config import FONT_NAMES, detect_font_style
from .renderer import get_page_bounds
from .text_processor import clean_and_normalize_text, join_lines_into_paragraph, join_spans_into_line

_MARKER_BODY = r"(?:[•●▪◦‣⁃\uFFFD*\uF0B7\uF0A7\uF0D8·○■□✓☑–—-]|\d+(?:\.\d+)*[.)]|\(\d{1,2}\)|\([A-Za-z]\)|[A-Za-z][.)])"
LIST_ITEM_PATTERN = re.compile(rf"^\s*(?P<marker>{_MARKER_BODY})\s+")
STANDALONE_LIST_MARKER_PATTERN = re.compile(rf"^\s*(?P<marker>{_MARKER_BODY})\s*$")
BARE_NUMBER_MARKER_PATTERN = re.compile(r"^\s*(?P<marker>\d{1,2})\s*$")

def split_horizontal_lines(raw_lines: list[dict]) -> list[list[dict]]:
    if len(raw_lines) <= 1 or not any(
        min(raw_lines[i]["bbox"].y1, raw_lines[j]["bbox"].y1) - max(raw_lines[i]["bbox"].y0, raw_lines[j]["bbox"].y0)
        > 0.4 * min(raw_lines[i]["bbox"].height, raw_lines[j]["bbox"].height)
        and abs(raw_lines[i]["bbox"].x0 - raw_lines[j]["bbox"].x0) > 8
        for i in range(len(raw_lines)) for j in range(i + 1, len(raw_lines))
    ):
        return [raw_lines]

    clusters: list[list[dict]] = []
    for line in raw_lines:
        l_x0, l_x1 = line["bbox"].x0, line["bbox"].x1
        for cluster in clusters:
            c_x0, c_x1 = min(l["bbox"].x0 for l in cluster), max(l["bbox"].x1 for l in cluster)
            ho, min_w = min(c_x1, l_x1) - max(c_x0, l_x0), min(c_x1 - c_x0, l_x1 - l_x0)
            if (min_w > 0 and ho > 0.3 * min_w) or (abs(c_x0 - l_x0) < 6 and abs(c_x1 - l_x1) < 6):
                cluster.append(line)
                break
        else:
            clusters.append([line])

    clusters.sort(key=lambda c: min(l["bbox"].x0 for l in c))
    for c in clusters:
        c.sort(key=lambda l: l["bbox"].y0)
    return clusters

def _bold_ratio(line: dict) -> float:
    """Tỷ lệ ký tự (không tính khoảng trắng) thuộc span đậm trong một dòng."""
    spans = line.get("spans", [])
    total = sum(len(s.get("text", "").strip()) for s in spans)
    bold = sum(len(s.get("text", "").strip()) for s in spans if s.get("style", "") in ("bold", "bold_italic"))
    return bold / total if total else 0.0

def group_lines_into_elements(raw_lines: list[dict]) -> list[list[dict]]:
    merged_lines, consumed = [], set()
    for m_idx, m_line in enumerate(raw_lines):
        if m_idx in consumed:
            continue
        m_match = STANDALONE_LIST_MARKER_PATTERN.match(m_line.get("text", "")) or BARE_NUMBER_MARKER_PATTERN.match(m_line.get("text", ""))
        if not m_match:
            merged_lines.append(m_line)
            continue
        m_bbox = m_line["bbox"]
        f_size = max((s.get("size", 0) for s in m_line.get("spans", [])), default=10.0)
        candidates = [
            (c_line["bbox"].x0 - m_bbox.x1, c_idx, c_line)
            for c_idx, c_line in enumerate(raw_lines)
            if c_idx != m_idx and c_idx not in consumed
            and (min(m_bbox.y1, c_line["bbox"].y1) - max(m_bbox.y0, c_line["bbox"].y0)) >= 0.5 * min(m_bbox.height, c_line["bbox"].height)
            and 0 <= (c_line["bbox"].x0 - m_bbox.x1) <= max(14.0, f_size * 2)
        ]
        if not candidates:
            merged_lines.append(m_line)
            continue
        _, c_idx, c_line = min(candidates, key=lambda x: x[0])
        c_bbox, m_raw, c_raw = c_line["bbox"], m_line.get("raw_bbox", m_bbox), c_line.get("raw_bbox", c_line["bbox"])
        merged_lines.append({
            **m_line,
            "text": f"{m_match.group('marker')}{'.' if m_match.group('marker').isdigit() else ''} {c_line['text']}",
            "bbox": fitz.Rect(min(m_bbox.x0, c_bbox.x0), min(m_bbox.y0, c_bbox.y0), max(m_bbox.x1, c_bbox.x1), max(m_bbox.y1, c_bbox.y1)),
            "raw_bbox": fitz.Rect(min(m_raw.x0, c_raw.x0), min(m_raw.y0, c_raw.y0), max(m_raw.x1, c_raw.x1), max(m_raw.y1, c_raw.y1)),
            "spans": m_line.get("spans", []) + c_line.get("spans", []),
        })
        consumed.add(c_idx)

    raw_lines = sorted(merged_lines, key=lambda l: (round(l["bbox"].y0 / 4) * 4, l["bbox"].x0))
    if len(raw_lines) <= 1:
        return [raw_lines]

    horizontal_clusters = split_horizontal_lines(raw_lines)
    if len(horizontal_clusters) >= 3 or (
        len(horizontal_clusters) >= 2
        and any((na := sum(1 for l in c if not any(ch.isalpha() for ch in l.get("text", "")))) >= 2 and (na / len(c)) >= 0.30 for c in horizontal_clusters)
    ):
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

        segments, current_plain, current_list_item = [], [], None
        for line in cluster:
            if LIST_ITEM_PATTERN.match(line.get("text", "")):
                if current_plain:
                    segments.append((False, current_plain))
                    current_plain = []
                current_list_item = [line]
                segments.append((True, current_list_item))
                continue
            if current_list_item:
                prev_b, line_b, start_b = current_list_item[-1]["bbox"], line["bbox"], current_list_item[0]["bbox"]
                f_sz = max((s.get("size", 0) for s in current_list_item[0].get("spans", [])), default=10.0)
                mk = LIST_ITEM_PATTERN.match(current_list_item[0].get("text", ""))
                mk_w = len(mk.group("marker")) * f_sz * 0.55 if mk else f_sz
                if (line_b.y0 - prev_b.y1) <= max(prev_b.height * 0.8, 3.0) and line_b.x0 >= start_b.x0 + max(f_sz * 0.7, mk_w):
                    current_list_item.append(line)
                    continue
                current_list_item = None
            current_plain.append(line)
        if current_plain:
            segments.append((False, current_plain))

        for is_list_item, lines in segments:
            if is_list_item:
                final_groups.append(lines)
                continue
            current_subgroup = [lines[0]]
            for next_line in lines[1:]:
                prev_line = current_subgroup[-1]
                p_bbox, n_bbox = prev_line["bbox"], next_line["bbox"]
                prev_sizes = [s["size"] for s in prev_line.get("spans", []) if s.get("size")]
                next_sizes = [s["size"] for s in next_line.get("spans", []) if s.get("size")]
                prev_size = statistics.median(prev_sizes) if prev_sizes else 10.0
                next_size = statistics.median(next_sizes) if next_sizes else 10.0
                size_diff = abs(prev_size - next_size)
                size_ratio = max(prev_size, next_size) / max(min(prev_size, next_size), 0.1)
                prev_ratio = _bold_ratio(prev_line)
                next_ratio = _bold_ratio(next_line)
                line_h, vert_gap = max(p_bbox.height, n_bbox.height, max(prev_size, next_size)), n_bbox.y0 - p_bbox.y1
                p_orig = [s["origin"][1] for s in prev_line.get("spans", []) if s.get("origin") and len(s["origin"]) >= 2]
                n_orig = [s["origin"][1] for s in next_line.get("spans", []) if s.get("origin") and len(s["origin"]) >= 2]
                if p_orig and n_orig:
                    pitch, max_sz = statistics.median(n_orig) - statistics.median(p_orig), max(prev_size, next_size, 1.0)
                    abnormal_vert = pitch < 0.55 * max_sz or pitch > 1.75 * max_sz
                else:
                    abnormal_vert = vert_gap < -2.0
                if size_diff >= 1.5 or size_ratio >= 1.2 or ((prev_ratio >= 0.95 and next_ratio <= 0.05) or (prev_ratio <= 0.05 and next_ratio >= 0.95)) or (vert_gap > 0.65 * line_h and vert_gap > 5.0) or abnormal_vert:
                    final_groups.append(current_subgroup)
                    current_subgroup = [next_line]
                else:
                    current_subgroup.append(next_line)
            if current_subgroup:
                final_groups.append(current_subgroup)
    return final_groups


def extract_page_elements(page: fitz.Page) -> list[dict]:
    elements, table_cell_bboxes, consumed_line_ids = [], [], set()
    page_w = get_page_bounds(page).width
    dict_data = page.get_text("dict", sort=True)
    all_visible_span_rects = [
        sr for b in dict_data.get("blocks", []) if b.get("type") == 0
        for l in b.get("lines", []) for s in l.get("spans", [])
        if s.get("alpha", 255) != 0 and s.get("text", "").strip() and not (sr := fitz.Rect(s["bbox"])).is_empty and not sr.is_infinite
    ]

    try:
        for tab in getattr(page.find_tables(), "tables", []):
            if not getattr(tab, "bbox", None):
                continue
            t_rect = fitz.Rect(tab.bbox)
            if t_rect.is_empty or t_rect.is_infinite or t_rect.width <= 0 or t_rect.height <= 0:
                continue
            raw_cell_rects = []
            for cb in getattr(tab, "cells", []):
                if cb and not (cr := fitz.Rect(cb)).is_empty and not cr.is_infinite and cr.width > 0 and cr.height > 0 and cr not in raw_cell_rects:
                    raw_cell_rects.append(cr)
            filtered_cells = [
                cr for idx, cr in enumerate(raw_cell_rects)
                if not any(o_idx != idx and other.get_area() > cr.get_area() and (cr & other).get_area() >= 0.98 * cr.get_area() for o_idx, other in enumerate(raw_cell_rects))
            ]
            changed = True
            while changed:
                changed = False
                for sr in all_visible_span_rects:
                    if sr.is_empty or sr.width <= 4:
                        continue
                    hits = [
                        idx for idx, cr in enumerate(filtered_cells)
                        if (min(sr.y1, cr.y1) - max(sr.y0, cr.y0)) >= 0.5 * sr.height and (min(sr.x1, cr.x1) - max(sr.x0, cr.x0)) >= max(3.0, 0.12 * sr.width)
                    ]
                    if len(hits) >= 2:
                        bc = filtered_cells[hits[0]]
                        same_row = [i for i in hits if abs(filtered_cells[i].y0 - bc.y0) < 4.0 and abs(filtered_cells[i].y1 - bc.y1) < 4.0]
                        if len(same_row) >= 2:
                            ur = fitz.Rect(filtered_cells[same_row[0]])
                            for i in same_row[1:]:
                                ur |= filtered_cells[i]
                            filtered_cells[same_row[0]] = ur
                            for i in sorted(same_row[1:], reverse=True):
                                filtered_cells.pop(i)
                            changed = True
                            break

            table_cell_bboxes.extend(filtered_cells)
            cell_assigned_lines: dict[int, list[dict]] = {i: [] for i in range(len(filtered_cells))}
            for b in dict_data.get("blocks", []):
                if b.get("type") != 0:
                    continue
                for l in b.get("lines", []):
                    if id(l) in consumed_line_ids or (lr := fitz.Rect(l["bbox"])).is_empty or lr.is_infinite or lr.get_area() <= 0 or not lr.intersects(t_rect):
                        continue
                    best_idx, best_area = max(((idx, (lr & cr).get_area()) for idx, cr in enumerate(filtered_cells)), key=lambda x: x[1], default=(-1, 0.0))
                    if best_idx >= 0 and best_area >= 0.5 * lr.get_area():
                        cell_assigned_lines[best_idx].append(l)
                        consumed_line_ids.add(id(l))

            for idx, c_rect in enumerate(filtered_cells):
                cell_spans, cell_lines = [], []
                for l in cell_assigned_lines[idx]:
                    valid_spans = [
                        {**s, "style": detect_font_style(s.get("flags", 0), s.get("font", ""))}
                        for s in l.get("spans", []) if s.get("alpha", 255) != 0 and s.get("text", "").strip()
                    ]
                    if valid_spans:
                        cell_spans.extend(valid_spans)
                        if l_text := join_spans_into_line([{"text": s["text"], "bbox": fitz.Rect(s["bbox"]), "size": s["size"]} for s in valid_spans]):
                            cell_lines.append({
                                "text": l_text,
                                "bbox": fitz.Rect(min(s["bbox"][0] for s in valid_spans), min(s["bbox"][1] for s in valid_spans), max(s["bbox"][2] for s in valid_spans), max(s["bbox"][3] for s in valid_spans)),
                                "raw_bbox": fitz.Rect(l["bbox"]), "spans": valid_spans,
                            })
                if not cell_lines:
                    continue
                cell_groups = group_lines_into_elements(cell_lines)
                contains_list = any(LIST_ITEM_PATTERN.match(g[0]["text"]) for g in cell_groups)
                cell_groups = sorted(cell_groups, key=lambda g: g[0]["bbox"].y0) if contains_list else [cell_lines]
                if not any(join_lines_into_paragraph(g) for g in cell_groups):
                    continue

                sizes = [s["size"] for s in cell_spans]
                size = Counter(sizes).most_common(1)[0][0] if sizes else 9.0
                style = detect_font_style(cell_spans[0]["flags"] if cell_spans else 0, cell_spans[0].get("font", "") if cell_spans else "")
                c_val = cell_spans[0].get("color", 0) if cell_spans else 0
                color = (((c_val >> 16) & 255) / 255, ((c_val >> 8) & 255) / 255, (c_val & 255) / 255)
                align = fitz.TEXT_ALIGN_LEFT
                if cell_spans:
                    left_gap = min(s["bbox"][0] for s in cell_spans) - c_rect.x0
                    right_gap = c_rect.x1 - max(s["bbox"][2] for s in cell_spans)
                    if right_gap < 8 and left_gap > max(8.0, 0.15 * c_rect.width) and c_rect.width > 30:
                        align = fitz.TEXT_ALIGN_RIGHT
                    elif left_gap > 8 and right_gap > 8 and abs(left_gap - right_gap) < 12 and c_rect.width > 20:
                        align = fitz.TEXT_ALIGN_CENTER

                pad_x, pad_y = min(1.5, c_rect.width * 0.08), min(1.0, c_rect.height * 0.08)
                for group_index, group in enumerate(cell_groups):
                    if not (cell_text := join_lines_into_paragraph(group)):
                        continue
                    bbox = fitz.Rect(min(l["bbox"].x0 for l in group), min(l["bbox"].y0 for l in group), max(l["bbox"].x1 for l in group), max(l["bbox"].y1 for l in group))
                    list_match = LIST_ITEM_PATTERN.match(group[0]["text"])
                    g_size, g_style, g_color = size, style, color
                    if contains_list:
                        g_spans = [sp for ln in group for sp in ln.get("spans", [])]
                        if g_spans:
                            main = max(g_spans, key=lambda sp: len(sp.get("text", "")))
                            g_size = main["size"]
                            g_style = detect_font_style(main["flags"], main.get("font", ""))
                            g_cv = main.get("color", 0)
                            g_color = (((g_cv >> 16) & 255) / 255, ((g_cv >> 8) & 255) / 255, (g_cv & 255) / 255)
                    if contains_list:
                        next_top = cell_groups[group_index + 1][0]["bbox"].y0 if group_index + 1 < len(cell_groups) else c_rect.y1
                        insert_rect = fitz.Rect(c_rect.x0 + pad_x, max(c_rect.y0 + pad_y, bbox.y0 - pad_y), c_rect.x1 - pad_x, max(bbox.y1 + pad_y, min(c_rect.y1 - pad_y, next_top - 1.0)))
                    else:
                        insert_rect = fitz.Rect(c_rect.x0 + pad_x, c_rect.y0 + pad_y, c_rect.x1 - pad_x, c_rect.y1 - pad_y)
                    element = {
                        "is_table_cell": True, "bbox": bbox, "insert_rect": insert_rect, "text": cell_text,
                        "size": round(g_size, 1), "style": g_style, "fontname": FONT_NAMES.get(g_style, "TNR"),
                        "color": g_color, "align": align, "lineheight_ratio": 1.15, "original_lines": group,
                    }
                    if list_match:
                        element["list_marker"] = list_match.group("marker")
                        element["translation_text"] = join_lines_into_paragraph([dict(group[0], text=group[0]["text"][list_match.end():]), *group[1:]])
                    elements.append(element)
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi nhận diện bảng: {e}")

    for block in dict_data.get("blocks", []):
        if block.get("type") != 0 or (b_rect := fitz.Rect(block["bbox"])).is_empty or b_rect.is_infinite or b_rect.width <= 0 or b_rect.height <= 0:
            continue
        block_lines = []
        for line in block.get("lines", []):
            if id(line) in consumed_line_ids or (lb := fitz.Rect(line["bbox"])).is_empty or lb.is_infinite or lb.width <= 0 or lb.height <= 0:
                continue
            if any((lb & cb).get_area() >= 0.5 * lb.get_area() for cb in table_cell_bboxes):
                continue
            line_dir = line.get("dir", (1.0, 0.0))
            is_rotated_line = abs(line_dir[0] - 1.0) > 0.05 or abs(line_dir[1]) > 0.05
            valid_spans = []
            for span in line.get("spans", []):
                if span.get("alpha", 255) == 0 or not span.get("text", "").strip():
                    continue
                flags, font_str, c_val = span["flags"], span.get("font", ""), span.get("color", 0)
                fn_low = (font_str or "").lower()
                is_mono = any(k in fn_low for k in ("courier", "consolas", "menlo", "monaco", "monospace", "firacode", "jetbrains")) or (
                    bool(flags & fitz.TEXT_FONT_MONOSPACED) and not any(k in fn_low for k in ("times", "arial", "helvetica", "calibri", "cambria", "georgia", "serif", "sans"))
                )
                valid_spans.append({
                    "text": span["text"], "bbox": fitz.Rect(span["bbox"]), "size": span["size"],
                    "flags": flags, "font": font_str, "origin": span.get("origin"), "is_monospace": is_mono,
                    "style": detect_font_style(flags, font_str),
                    "color": (((c_val >> 16) & 255) / 255, ((c_val >> 8) & 255) / 255, (c_val & 255) / 255),
                })
            if not valid_spans:
                continue

            segments, current_seg = [], [valid_spans[0]]
            for s in valid_spans[1:]:
                gap = s["bbox"].x0 - current_seg[-1]["bbox"].x1
                if gap > 35 and gap > 3.5 * max(current_seg[-1]["size"], s["size"]):
                    segments.append(current_seg)
                    current_seg = [s]
                else:
                    current_seg.append(s)
            segments.append(current_seg)

            for seg in segments:
                if line_text := join_spans_into_line(seg):
                    seg_rect = fitz.Rect(min(s["bbox"].x0 for s in seg), min(s["bbox"].y0 for s in seg), max(s["bbox"].x1 for s in seg), max(s["bbox"].y1 for s in seg))
                    block_lines.append({
                        "text": line_text, "bbox": seg_rect, "raw_bbox": fitz.Rect(line["bbox"]) if len(segments) == 1 else seg_rect,
                        "spans": seg, "is_rotated": is_rotated_line,
                    })

        if not block_lines:
            continue

        for sg in group_lines_into_elements(block_lines):
            sg_bbox = fitz.Rect(min(l["bbox"].x0 for l in sg), min(l["bbox"].y0 for l in sg), max(l["bbox"].x1 for l in sg), max(l["bbox"].y1 for l in sg))
            sg_spans = [s for l in sg for s in l.get("spans", [])]
            sizes = [s["size"] for s in sg_spans]
            size = Counter(sizes).most_common(1)[0][0] if sizes else 10.0
            style_counter = Counter(s["style"] for s in sg_spans)
            style = style_counter.most_common(1)[0][0] if style_counter else "regular"
            total_chars = sum(len(s["text"].strip()) for s in sg_spans)
            mono_chars = sum(len(s["text"].strip()) for s in sg_spans if s.get("is_monospace"))
            lineheight_ratio = 1.2
            if len(sg) > 1 and size > 0:
                pos_diffs = [sg[i + 1]["bbox"].y0 - sg[i]["bbox"].y0 for i in range(len(sg) - 1) if sg[i + 1]["bbox"].y0 > sg[i]["bbox"].y0]
                if pos_diffs:
                    lineheight_ratio = statistics.median(pos_diffs) / size
            lineheight_ratio = max(1.05, min(lineheight_ratio, 1.45))

            source_text = join_lines_into_paragraph(sg) if len(sg) > 1 else clean_and_normalize_text(sg[0]["text"])
            list_match = LIST_ITEM_PATTERN.match(sg[0]["text"])
            align = fitz.TEXT_ALIGN_CENTER if len(sg) == 1 and not list_match and abs(page_w / 2 - (sg_bbox.x0 + sg_bbox.x1) / 2) < 30 and sg_bbox.width < page_w * 0.7 else fitz.TEXT_ALIGN_LEFT
            element = {
                "is_table_cell": False, "is_code": total_chars > 0 and (mono_chars / total_chars) >= 0.8,
                "is_rotated": any(l.get("is_rotated") for l in sg), "bbox": sg_bbox, "insert_rect": sg_bbox,
                "text": source_text, "size": round(size, 1), "style": style, "fontname": FONT_NAMES.get(style, "TNR"),
                "color": sg_spans[0]["color"] if sg_spans else (0, 0, 0), "align": align, "lineheight_ratio": lineheight_ratio, "original_lines": sg,
            }
            if list_match:
                element["list_marker"] = list_match.group("marker")
                element["translation_text"] = join_lines_into_paragraph([dict(sg[0], text=sg[0]["text"][list_match.end():]), *sg[1:]])
            elements.append(element)

    if elements:
        mid_x = page_w / 2
        left_cnt = sum(1 for e in elements if not e.get("is_table_cell") and e["bbox"].x1 < mid_x + 15 and e["bbox"].width < page_w * 0.6)
        right_cnt = sum(1 for e in elements if not e.get("is_table_cell") and e["bbox"].x0 > mid_x - 15 and e["bbox"].width < page_w * 0.6)
        if left_cnt >= 3 and right_cnt >= 3:
            spanning_ys = sorted((e["bbox"].y0, e["bbox"].y1) for e in elements if e["bbox"].width > page_w * 0.65)
            elements.sort(key=lambda e: (
                sum(1 for _, sy1 in spanning_ys if sy1 <= e["bbox"].y0 + 2.0),
                0 if e["bbox"].width > page_w * 0.65 else (1 if e["bbox"].x0 < mid_x else 2),
                round(e["bbox"].y0 / 4) * 4, e["bbox"].x0,
            ))
        else:
            elements.sort(key=lambda e: (round(e["bbox"].y0 / 4) * 4, e["bbox"].x0))

    # Dòng đơn gần giữa trang bị nhận là "căn giữa" nhưng cùng mép trái với >= 2 phần tử khác
    # thì thực chất là một cột căn trái (ví dụ các gạch đầu dòng) -> đặt lại căn trái.
    for e in elements:
        if e["align"] == fitz.TEXT_ALIGN_CENTER and not e.get("is_table_cell"):
            same_left = sum(1 for o in elements if o is not e and abs(o["bbox"].x0 - e["bbox"].x0) < 2.0)
            if same_left >= 2:
                e["align"] = fitz.TEXT_ALIGN_LEFT

    return elements
