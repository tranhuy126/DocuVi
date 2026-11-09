import gc
import os
import sys
import time

import pymupdf as fitz

from pdf_engine import (
    FONT_NAMES,
    ScratchFontTester,
    build_redaction_rects,
    clean_and_normalize_text,
    collect_page_obstacles,
    compute_safe_insert_rect,
    extract_page_elements,
    get_font_buffers,
    get_page_bounds,
    get_protected_rects,
    pre_filter_text,
    split_into_sentences,
    translate_batch,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def translate_pdf_document(doc: fitz.Document, batch_size: int = 32, num_beams: int = 1, progress_callback=None):
    if doc.is_encrypted:
        raise ValueError("Tài liệu PDF được bảo vệ bằng mật khẩu hoặc bị mã hóa. Vui lòng mở khóa file trước khi dịch.")

    all_page_elements: list[list[dict]] = []
    sentences_to_translate: list[str] = []
    sentence_map: list[tuple[int, int, int]] = []
    total_pages = len(doc)

    if progress_callback and total_pages > 0:
        progress_callback(5, f"Bắt đầu phân tích cấu trúc {total_pages} trang...")

    with ScratchFontTester() as scratch_tester:
        for page_idx, page in enumerate(doc):
            if progress_callback and total_pages > 0:
                progress_callback(5 + int(15 * page_idx / total_pages), f"Đang phân tích cấu trúc trang {page_idx + 1}/{total_pages}...")

            elements = extract_page_elements(page)
            all_page_elements.append(elements)

            for elem_idx, elem in enumerate(elements):
                if elem.get("is_code") or elem.get("is_rotated"):
                    elem["need_trans"], elem["translated_text"] = False, elem["text"]
                    continue
                need_trans, text = pre_filter_text(elem.get("translation_text", elem["text"]))
                elem["need_trans"] = need_trans
                if not need_trans:
                    elem["translated_text"] = text
                    continue
                sents = split_into_sentences(text)
                elem["raw_sentences"] = sents
                queued = False
                for s_idx, sent in enumerate(sents):
                    s_need, s_clean = pre_filter_text(sent)
                    if s_need:
                        sentences_to_translate.append(s_clean)
                        sentence_map.append((page_idx, elem_idx, s_idx))
                        queued = True
                    else:
                        sents[s_idx] = s_clean
                if not queued:
                    elem["need_trans"] = False
                    elem["translated_text"] = text

        if not any(all_page_elements) and total_pages > 0:
            raise ValueError(
                "Tài liệu PDF không chứa lớp văn bản (text layer) có thể trích xuất "
                "(tài liệu trống hoặc là dạng ảnh quét/scanned PDF). "
                "Hệ thống hiện tại chỉ hỗ trợ các tài liệu PDF có lớp văn bản."
            )

        if sentences_to_translate:
            if progress_callback:
                progress_callback(20, f"Đang dịch {len(sentences_to_translate)} câu bằng mô hình AI...")

            translated_sents = translate_batch(
                sentences_to_translate,
                batch_size=batch_size,
                num_beams=num_beams,
                clear_cache=True,
                progress_callback=lambda done, total: progress_callback(
                    20 + int(55 * done / total), f"Đang dịch AI: {done}/{total} câu ({int(done / total * 100)}%)..."
                ) if progress_callback and total > 0 else None,
            )

            if len(translated_sents) != len(sentence_map):
                print(
                    f"[DocuVi] Số câu dịch trả về không khớp với đầu vào "
                    f"({len(translated_sents)} kết quả cho {len(sentence_map)} câu); các câu thiếu sẽ giữ nguyên bản gốc."
                )

            for (p_idx, e_idx, s_idx), trans_sent in zip(sentence_map, translated_sents):
                elem = all_page_elements[p_idx][e_idx]
                trans_list = elem.setdefault("trans_sents", list(elem["raw_sentences"]))
                trans_list[s_idx] = (trans_sent or "").strip() or elem["raw_sentences"][s_idx]

            for elem in (e for page_elems in all_page_elements for e in page_elems if e.get("need_trans")):
                joined = " ".join(elem.get("trans_sents", elem.get("raw_sentences", []))).strip()
                if elem.get("list_marker"):
                    joined = f"{elem['list_marker']} {joined}"
                elem["translated_text"] = clean_and_normalize_text(joined)

        font_buffers = get_font_buffers()
        for page_idx, page in enumerate(doc):
            if progress_callback and total_pages > 0:
                progress_callback(75 + int(23 * page_idx / total_pages), f"Đang căn chỉnh font và render trang {page_idx + 1}/{total_pages}...")

            elements = all_page_elements[page_idx]
            page_obstacles = collect_page_obstacles(page)

            for elem in elements:
                trans_text = elem.get("translated_text", "").strip()
                if not elem.get("need_trans") or not trans_text:
                    elem["can_render"] = False
                    continue

                insert_rect = compute_safe_insert_rect(page, elem, elements, page_obstacles=page_obstacles)
                is_cell = elem.get("is_table_cell", False)
                base_min, abs_min = (4.5, 4.0) if is_cell else (5.0, 4.5)
                lh = elem["lineheight_ratio"]
                attempts = [(base_min, lh)] + ([(base_min, 1.05)] if lh > 1.05 else []) + [(abs_min, 0.98)]

                for min_sz, cand_lh in attempts:
                    font_size, fitted = scratch_tester.fit_size(
                        rect=insert_rect, text=trans_text, fontname=elem["fontname"],
                        max_size=elem["size"], min_size=min_sz, lineheight=cand_lh, align=elem["align"],
                    )
                    lh = cand_lh
                    if fitted:
                        break

                elem["can_render"] = fitted and font_size >= abs_min
                if elem["can_render"]:
                    elem["final_insert_rect"], elem["final_size"], elem["final_lineheight"] = insert_rect, font_size, lh
                else:
                    print(
                        f"[DocuVi] Phần tử tại trang {page_idx + 1} ({elem['bbox']}) "
                        "không thể vừa khung an toàn mà không làm tràn chữ. GIỮ NGUYÊN VĂN BẢN GỐC để đảm bảo an toàn tài liệu."
                    )

            renderable = [e for e in elements if e.get("can_render")]
            if not renderable:
                continue

            page_backup = fitz.open()
            page_backup.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            bounds, protected = get_page_bounds(page), get_protected_rects(elements)

            for elem in renderable:
                for clean_bbox in build_redaction_rects(elem, protected, bounds):
                    page.add_redact_annot(clean_bbox, fill=None)

            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
            for style, fontname in FONT_NAMES.items():
                page.insert_font(fontname=fontname, fontbuffer=font_buffers[style])

            insertion_ok = False
            try:
                insertion_ok = all(
                    any(
                        page.insert_textbox(
                            e["final_insert_rect"], e["translated_text"], fontsize=fs,
                            fontname=e["fontname"], color=e["color"], lineheight=lh, align=e["align"],
                        ) >= 0
                        for fs, lh in ((e["final_size"], e["final_lineheight"]), (max(3.8, e["final_size"] - 0.2), min(e["final_lineheight"], 1.0)))
                    )
                    for e in renderable
                )
            finally:
                if not insertion_ok:
                    doc.delete_page(page_idx)
                    doc.insert_pdf(page_backup, from_page=0, to_page=0, start_at=page_idx)
                    print(f"[DocuVi] Không thể chèn bản dịch an toàn ở trang {page_idx + 1}; đã khôi phục nguyên trang gốc.")
                page_backup.close()

        if progress_callback:
            progress_callback(98, "Đang tối ưu hóa và ghi file PDF kết quả...")

    gc.collect()


def translate_pdf(input_path: str, output_path: str, batch_size: int = 32, num_beams: int = 1, progress_callback=None):
    with fitz.open(input_path) as doc:
        translate_pdf_document(doc, batch_size=batch_size, num_beams=num_beams, progress_callback=progress_callback)
        doc.save(output_path, garbage=4, deflate=True, clean=True)
    gc.collect()


def translate_pdf_bytes(pdf_bytes: bytes, batch_size: int = 32, num_beams: int = 1, progress_callback=None) -> bytes:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        translate_pdf_document(doc, batch_size=batch_size, num_beams=num_beams, progress_callback=progress_callback)
        return doc.tobytes(garbage=4, deflate=True, clean=True)


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    test_in, test_out = os.path.join(BASE, "uploads", "test1.pdf"), os.path.join(BASE, "outputs", "translated.pdf")
    t0 = time.time()
    translate_pdf(test_in, test_out)
    print(f"Hoàn thành dịch PDF trong {round(time.time() - t0, 2)}s: {test_out}")
