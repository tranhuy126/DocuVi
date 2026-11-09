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


def translate_pdf_document(
    doc: fitz.Document,
    batch_size: int = 32,
    num_beams: int = 1,
    progress_callback=None,
):
    if doc.is_encrypted:
        raise ValueError(
            "Tài liệu PDF được bảo vệ bằng mật khẩu hoặc bị mã hóa. "
            "Vui lòng mở khóa file trước khi dịch."
        )

    all_page_elements: list[list[dict]] = []
    sentences_to_translate: list[str] = []
    sentence_locations: list[tuple[int, int, int]] = []
    total_pages = len(doc)

    if progress_callback and total_pages > 0:
        progress_callback(5, f"Bắt đầu phân tích cấu trúc {total_pages} trang...")

    with ScratchFontTester() as scratch_tester:
        for page_index, page in enumerate(doc):
            if progress_callback and total_pages > 0:
                progress_percent = 5 + int(15 * page_index / total_pages)
                progress_callback(
                    progress_percent,
                    f"Đang phân tích cấu trúc trang {page_index + 1}/{total_pages}...",
                )

            page_elements = extract_page_elements(page)
            all_page_elements.append(page_elements)

            for element_index, element in enumerate(page_elements):
                if element.get("is_code") or element.get("is_rotated"):
                    element["need_trans"] = False
                    element["translated_text"] = element["text"]
                    continue

                raw_element_text = element.get("translation_text", element["text"])
                needs_translation, filtered_text = pre_filter_text(raw_element_text)
                element["need_trans"] = needs_translation
                if not needs_translation:
                    element["translated_text"] = filtered_text
                    continue

                sentences = split_into_sentences(filtered_text)
                element["raw_sentences"] = sentences
                has_queued_sentence = False

                for sentence_index, sentence in enumerate(sentences):
                    sentence_needs_trans, cleaned_sentence = pre_filter_text(sentence)
                    if sentence_needs_trans:
                        sentences_to_translate.append(cleaned_sentence)
                        sentence_locations.append((page_index, element_index, sentence_index))
                        has_queued_sentence = True
                    else:
                        sentences[sentence_index] = cleaned_sentence

                if not has_queued_sentence:
                    element["need_trans"] = False
                    element["translated_text"] = filtered_text

        if not any(all_page_elements) and total_pages > 0:
            raise ValueError(
                "Tài liệu PDF không chứa lớp văn bản (text layer) có thể trích xuất "
                "(tài liệu trống hoặc là dạng ảnh quét/scanned PDF). "
                "Hệ thống hiện tại chỉ hỗ trợ các tài liệu PDF có lớp văn bản."
            )

        if sentences_to_translate:
            if progress_callback:
                progress_callback(20, f"Đang dịch {len(sentences_to_translate)} câu bằng mô hình AI...")

            def _on_batch_progress(done_count: int, total_count: int):
                if progress_callback and total_count > 0:
                    progress_percent = 20 + int(55 * done_count / total_count)
                    completion_ratio = int(done_count / total_count * 100)
                    progress_callback(
                        progress_percent,
                        f"Đang dịch AI: {done_count}/{total_count} câu ({completion_ratio}%)...",
                    )

            translated_sentences = translate_batch(
                sentences_to_translate,
                batch_size=batch_size,
                num_beams=num_beams,
                clear_cache=True,
                progress_callback=_on_batch_progress,
            )

            if len(translated_sentences) != len(sentence_locations):
                print(
                    f"[DocuVi] Số câu dịch trả về không khớp với đầu vào "
                    f"({len(translated_sentences)} kết quả cho {len(sentence_locations)} câu); "
                    "các câu thiếu sẽ giữ nguyên bản gốc."
                )

            for (page_index, element_index, sentence_index), translated_sentence in zip(
                sentence_locations,
                translated_sentences,
            ):
                element = all_page_elements[page_index][element_index]
                translated_list = element.setdefault("trans_sents", list(element["raw_sentences"]))
                cleaned_translation = (translated_sentence or "").strip()
                translated_list[sentence_index] = (
                    cleaned_translation or element["raw_sentences"][sentence_index]
                )

            for page_elements in all_page_elements:
                for element in page_elements:
                    if not element.get("need_trans"):
                        continue
                    sentence_parts = element.get("trans_sents", element.get("raw_sentences", []))
                    joined_paragraph = " ".join(sentence_parts).strip()
                    if element.get("list_marker"):
                        joined_paragraph = f"{element['list_marker']} {joined_paragraph}"
                    element["translated_text"] = clean_and_normalize_text(joined_paragraph)

        font_buffers = get_font_buffers()
        for page_index, page in enumerate(doc):
            if progress_callback and total_pages > 0:
                progress_percent = 75 + int(23 * page_index / total_pages)
                progress_callback(
                    progress_percent,
                    f"Đang căn chỉnh font và render trang {page_index + 1}/{total_pages}...",
                )

            page_elements = all_page_elements[page_index]
            page_obstacles = collect_page_obstacles(page)

            for element in page_elements:
                translated_text = element.get("translated_text", "").strip()
                if not element.get("need_trans") or not translated_text:
                    element["can_render"] = False
                    continue

                insert_rect = compute_safe_insert_rect(
                    page,
                    element,
                    page_elements,
                    page_obstacles=page_obstacles,
                )
                is_table_cell = element.get("is_table_cell", False)
                base_min_size, absolute_min_size = (4.5, 4.0) if is_table_cell else (5.0, 4.5)
                current_lineheight = element["lineheight_ratio"]

                fitting_attempts = [(base_min_size, current_lineheight)]
                if current_lineheight > 1.05:
                    fitting_attempts.append((base_min_size, 1.05))
                fitting_attempts.append((absolute_min_size, 0.98))

                is_fitted = False
                fitted_font_size = absolute_min_size
                for min_allowed_size, candidate_lineheight in fitting_attempts:
                    fitted_font_size, is_fitted = scratch_tester.fit_size(
                        rect=insert_rect,
                        text=translated_text,
                        fontname=element["fontname"],
                        max_size=element["size"],
                        min_size=min_allowed_size,
                        lineheight=candidate_lineheight,
                        align=element["align"],
                    )
                    current_lineheight = candidate_lineheight
                    if is_fitted:
                        break

                element["can_render"] = is_fitted and fitted_font_size >= absolute_min_size
                if element["can_render"]:
                    element["final_insert_rect"] = insert_rect
                    element["final_size"] = fitted_font_size
                    element["final_lineheight"] = current_lineheight
                else:
                    print(
                        f"[DocuVi] Phần tử tại trang {page_index + 1} ({element['bbox']}) "
                        "không thể vừa khung an toàn mà không làm tràn chữ. "
                        "GIỮ NGUYÊN VĂN BẢN GỐC để đảm bảo an toàn tài liệu."
                    )

            renderable_elements = [element for element in page_elements if element.get("can_render")]
            if not renderable_elements:
                continue

            page_backup = fitz.open()
            page_backup.insert_pdf(doc, from_page=page_index, to_page=page_index)
            page_bounds = get_page_bounds(page)
            protected_rects = get_protected_rects(page_elements)

            for element in renderable_elements:
                for clean_bbox in build_redaction_rects(element, protected_rects, page_bounds):
                    page.add_redact_annot(clean_bbox, fill=None)

            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
            for style_name, fontname in FONT_NAMES.items():
                page.insert_font(fontname=fontname, fontbuffer=font_buffers[style_name])

            insertion_succeeded = False
            try:
                all_elements_inserted = True
                for element in renderable_elements:
                    primary_attempt = (element["final_size"], element["final_lineheight"])
                    fallback_attempt = (
                        max(3.8, element["final_size"] - 0.2),
                        min(element["final_lineheight"], 1.0),
                    )
                    element_inserted = False
                    for candidate_font_size, candidate_lineheight in (primary_attempt, fallback_attempt):
                        remaining_height = page.insert_textbox(
                            element["final_insert_rect"],
                            element["translated_text"],
                            fontsize=candidate_font_size,
                            fontname=element["fontname"],
                            color=element["color"],
                            lineheight=candidate_lineheight,
                            align=element["align"],
                        )
                        if remaining_height >= 0:
                            element_inserted = True
                            break

                    if not element_inserted:
                        all_elements_inserted = False
                        break

                insertion_succeeded = all_elements_inserted
            finally:
                if not insertion_succeeded:
                    doc.delete_page(page_index)
                    doc.insert_pdf(page_backup, from_page=0, to_page=0, start_at=page_index)
                    print(
                        f"[DocuVi] Không thể chèn bản dịch an toàn ở trang {page_index + 1}; "
                        "đã khôi phục nguyên trang gốc."
                    )
                page_backup.close()

        if progress_callback:
            progress_callback(98, "Đang tối ưu hóa và ghi file PDF kết quả...")

    gc.collect()


def translate_pdf(
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    num_beams: int = 1,
    progress_callback=None,
):
    with fitz.open(input_path) as doc:
        translate_pdf_document(
            doc,
            batch_size=batch_size,
            num_beams=num_beams,
            progress_callback=progress_callback,
        )
        doc.save(output_path, garbage=4, deflate=True, clean=True)
    gc.collect()


def translate_pdf_bytes(
    pdf_bytes: bytes,
    batch_size: int = 32,
    num_beams: int = 1,
    progress_callback=None,
) -> bytes:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        translate_pdf_document(
            doc,
            batch_size=batch_size,
            num_beams=num_beams,
            progress_callback=progress_callback,
        )
        return doc.tobytes(garbage=4, deflate=True, clean=True)


if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    sample_input_path = os.path.join(base_directory, "uploads", "test1.pdf")
    sample_output_path = os.path.join(base_directory, "outputs", "translated.pdf")
    start_time = time.time()
    translate_pdf(sample_input_path, sample_output_path)
    elapsed_seconds = round(time.time() - start_time, 2)
    print(f"Hoàn thành dịch PDF trong {elapsed_seconds}s: {sample_output_path}")

