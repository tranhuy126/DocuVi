import os
import pymupdf as fitz
import sys
import time

from pdf_engine import (
    BASE_DIR,
    FONT_DIR,
    FONT_FILES,
    FONT_NAMES,
    FONT_BUFFERS,
    get_font_buffers,
    detect_font_style,
    MODEL_NAME,
    get_device,
    get_translation_model,
    translate_batch,
    clean_and_normalize_text,
    join_spans_into_line,
    join_lines_into_paragraph,
    pre_filter_text,
    split_into_sentences,
    split_horizontal_lines,
    group_lines_into_elements,
    extract_page_elements,
    ScratchFontTester,
    compute_safe_insert_rect,
)

# Ép mã hóa UTF-8 trên Windows console để in log tiếng Việt không bị lỗi font
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def translate_pdf_document(
    doc: fitz.Document, batch_size: int = 32, num_beams: int = 1
):
    """Pipeline dịch toàn bộ PDF: trích xuất layout, dịch batch GPU và tái tạo trang giữ nguyên định dạng."""
    if doc.is_encrypted:
        raise ValueError(
            "Tài liệu PDF được bảo vệ bằng mật khẩu hoặc bị mã hóa. Vui lòng mở khóa file trước khi dịch."
        )

    scratch_tester = ScratchFontTester()
    try:
        all_page_elements: list[list[dict]] = []
        sentences_to_translate: list[str] = []
        sentence_map: list[tuple[int, int, int]] = []

        # Bước 1: Trích xuất cấu trúc từng trang
        for page_idx, page in enumerate(doc):
            elements = extract_page_elements(page)
            all_page_elements.append(elements)

            for elem_idx, elem in enumerate(elements):
                need_trans, text = pre_filter_text(elem["text"])
                if not need_trans:
                    elem["need_trans"] = False
                    elem["translated_text"] = text
                else:
                    elem["need_trans"] = True
                    sents = split_into_sentences(text)
                    elem["raw_sentences"] = sents
                    for s_idx, sent in enumerate(sents):
                        sentences_to_translate.append(sent)
                        sentence_map.append((page_idx, elem_idx, s_idx))

        total_extracted = sum(len(elems) for elems in all_page_elements)
        if total_extracted == 0 and len(doc) > 0:
            raise ValueError(
                "Tài liệu PDF không chứa lớp văn bản (text layer) có thể trích xuất "
                "(tài liệu trống hoặc là dạng ảnh quét/scanned PDF). "
                "Hệ thống hiện tại chỉ hỗ trợ các tài liệu PDF có lớp văn bản."
            )

        # Bước 2: Dịch toàn bộ cùng lúc theo batch tối ưu trên GPU/CPU
        if sentences_to_translate:
            translated_sents = translate_batch(
                sentences_to_translate,
                batch_size=batch_size,
                num_beams=num_beams,
                clear_cache=True,
            )

            for (p_idx, e_idx, s_idx), trans_sent in zip(sentence_map, translated_sents):
                elem = all_page_elements[p_idx][e_idx]
                if "trans_sents" not in elem:
                    elem["trans_sents"] = [""] * len(elem["raw_sentences"])
                # Fallback về câu gốc nếu câu dịch rỗng hoặc lỗi
                raw_sent = elem["raw_sentences"][s_idx]
                elem["trans_sents"][s_idx] = trans_sent.strip() if (trans_sent and trans_sent.strip()) else raw_sent

            for page_elements in all_page_elements:
                for elem in page_elements:
                    if elem.get("need_trans"):
                        raw_joined = " ".join(elem.get("trans_sents", [])).strip()
                        elem["translated_text"] = clean_and_normalize_text(raw_joined)

        # Bước 3: Điều chỉnh cỡ font và chèn chữ an toàn vào trang
        for page_idx, page in enumerate(doc):
            elements = all_page_elements[page_idx]

            # 3.1: Kiểm tra trước khả năng vừa vặn cho từng phần tử bằng ScratchFontTester
            for elem in elements:
                if not elem.get("need_trans"):
                    elem["can_render"] = False
                    continue

                trans_text = elem.get("translated_text", "").strip()
                if not trans_text:
                    elem["can_render"] = False
                    continue

                insert_rect = compute_safe_insert_rect(page, elem, elements)
                is_cell = elem.get("is_table_cell", False)
                base_min_size = 4.5 if is_cell else 5.0
                max_size = elem["size"]
                lineheight = elem["lineheight_ratio"]

                # Lần thử 1: Với cỡ chữ gốc và lineheight hiện tại
                font_size, fitted = scratch_tester.fit_size(
                    rect=insert_rect,
                    text=trans_text,
                    fontname=elem["fontname"],
                    max_size=max_size,
                    min_size=base_min_size,
                    lineheight=lineheight,
                    align=elem["align"],
                )

                # Lần thử 2: Nếu chưa vừa, giảm nhẹ lineheight xuống 1.05
                if not fitted and lineheight > 1.05:
                    lineheight = 1.05
                    font_size, fitted = scratch_tester.fit_size(
                        rect=insert_rect,
                        text=trans_text,
                        fontname=elem["fontname"],
                        max_size=max_size,
                        min_size=base_min_size,
                        lineheight=lineheight,
                        align=elem["align"],
                    )

                # Lần thử 3: Nếu chưa vừa, giảm lineheight xuống 0.98 và min_size xuống 4.0 (cell) / 4.5 (text)
                if not fitted:
                    lineheight = 0.98
                    min_limit = 4.0 if is_cell else 4.5
                    font_size, fitted = scratch_tester.fit_size(
                        rect=insert_rect,
                        text=trans_text,
                        fontname=elem["fontname"],
                        max_size=max_size,
                        min_size=min_limit,
                        lineheight=lineheight,
                        align=elem["align"],
                    )

                if fitted and font_size >= (4.0 if is_cell else 4.5):
                    elem["can_render"] = True
                    elem["final_insert_rect"] = insert_rect
                    elem["final_size"] = font_size
                    elem["final_lineheight"] = lineheight
                else:
                    elem["can_render"] = False
                    print(
                        f"[DocuVi] Phần tử tại trang {page_idx + 1} ({elem['bbox']}) "
                        f"không thể vừa khung an toàn mà không làm tràn chữ. "
                        "GIỮ NGUYÊN VĂN BẢN GỐC để đảm bảo an toàn tài liệu."
                    )

            # 3.2: CHỈ XÓA VÙNG CHỮ CỦA NHỮNG PHẦN TỬ ĐÃ XÁC NHẬN CHÈN AN TOÀN
            # Tuyệt đối không xóa văn bản gốc nếu bản dịch không thể hiển thị vừa vặn!
            renderable_elements = [e for e in elements if e.get("can_render")]

            for elem in renderable_elements:
                b = elem["bbox"]
                clean_bbox = fitz.Rect(
                    max(0, b.x0 - 0.5),
                    max(0, b.y0 - 0.5),
                    min(page.rect.x1, b.x1 + 0.5),
                    min(page.rect.y1, b.y1 + 0.5),
                )
                page.add_redact_annot(clean_bbox, fill=None)

            if renderable_elements:
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                )

                font_buffers = get_font_buffers()
                for style, fontname in FONT_NAMES.items():
                    page.insert_font(fontname=fontname, fontbuffer=font_buffers[style])

                # 3.3: Chèn văn bản dịch đã được xác nhận vào đúng tọa độ an toàn
                for elem in renderable_elements:
                    rc = page.insert_textbox(
                        elem["final_insert_rect"],
                        elem["translated_text"],
                        fontsize=elem["final_size"],
                        fontname=elem["fontname"],
                        color=elem["color"],
                        lineheight=elem["final_lineheight"],
                        align=elem["align"],
                    )
                    # Nếu có sai số float vi lượng làm rc < 0, hạ nhẹ 0.2pt để vừa khít tuyệt đối trong cùng bounding box
                    if rc < 0:
                        reduced_size = max(3.8, elem["final_size"] - 0.2)
                        page.insert_textbox(
                            elem["final_insert_rect"],
                            elem["translated_text"],
                            fontsize=reduced_size,
                            fontname=elem["fontname"],
                            color=elem["color"],
                            lineheight=min(elem["final_lineheight"], 1.0),
                            align=elem["align"],
                        )
                        # Tuyệt đối KHÔNG gọi insert_text tọa độ điểm làm tràn chữ ra ngoài lề trang!

    finally:
        scratch_tester.close()


def translate_pdf(
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    num_beams: int = 1,
):
    """Dịch file PDF từ đường dẫn và lưu file kết quả."""
    with fitz.open(input_path) as doc:
        translate_pdf_document(doc, batch_size=batch_size, num_beams=num_beams)
        doc.save(output_path, garbage=4, deflate=True, clean=True)


def translate_pdf_bytes(
    pdf_bytes: bytes,
    batch_size: int = 32,
    num_beams: int = 1,
) -> bytes:
    """Dịch file PDF từ bộ nhớ đệm bytes và trả về bytes kết quả."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        translate_pdf_document(doc, batch_size=batch_size, num_beams=num_beams)
        return doc.tobytes(garbage=4, deflate=True, clean=True)


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    test_in = os.path.join(BASE, "uploads", "test1.pdf")
    test_out = os.path.join(BASE, "outputs", "translated.pdf")
    t0 = time.time()
    translate_pdf(test_in, test_out)
    print(f"Hoàn thành dịch PDF trong {round(time.time() - t0, 2)}s: {test_out}")
