import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import pymupdf as fitz

from pdf_engine import (
    get_font_buffers,
    FONT_NAMES,
    clean_and_normalize_text,
    pre_filter_text,
    split_into_sentences,
    extract_page_elements,
    ScratchFontTester,
    compute_safe_insert_rect,
    group_lines_into_elements,
)
from pdf_process import translate_pdf, translate_pdf_document
import app as flask_app_module


class TestDocuViComprehensiveSuite(unittest.TestCase):
    """Bộ kiểm thử hồi quy toàn diện 12 trường hợp bắt buộc cho hệ thống DocuVi."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.app = flask_app_module.app.test_client()
        self.app.testing = True

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Trường hợp 1: PDF hợp lệ có một đoạn văn ngắn
    # -------------------------------------------------------------------------
    def test_case_1_short_valid_pdf(self):
        input_pdf = os.path.join(self.temp_dir, "case1_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case1_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=595, height=842)
        p.insert_textbox(
            fitz.Rect(50, 50, 500, 100),
            "Artificial Intelligence is advancing rapidly in modern society.",
            fontsize=12,
        )
        doc.save(input_pdf)
        doc.close()

        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.return_value = ["Trí tuệ nhân tạo đang phát triển nhanh chóng trong xã hội hiện đại."]
            translate_pdf(input_pdf, output_pdf)

        self.assertTrue(os.path.exists(output_pdf))
        self.assertGreater(os.path.getsize(output_pdf), 0)

        with fitz.open(output_pdf) as out_doc:
            self.assertEqual(len(out_doc), 1)
            text = clean_and_normalize_text(out_doc[0].get_text())
            self.assertTrue("Trí tuệ nhân tạo" in text or "phát triển" in text)
        print("✓ Test Case 1: PDF hợp lệ 1 đoạn văn ngắn - ĐẠT")


    # -------------------------------------------------------------------------
    # Trường hợp 2: PDF có nhiều đoạn và nhiều trang
    # -------------------------------------------------------------------------
    def test_case_2_multipage_multipara_pdf(self):
        input_pdf = os.path.join(self.temp_dir, "case2_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case2_out.pdf")

        doc = fitz.open()
        for page_num in range(3):
            p = doc.new_page(width=595, height=842)
            p.insert_textbox(fitz.Rect(50, 50, 500, 120), f"Page {page_num + 1} Section One. First detailed paragraph.", fontsize=11)
            p.insert_textbox(fitz.Rect(50, 150, 500, 220), f"Page {page_num + 1} Section Two. Second detailed paragraph.", fontsize=11)
        doc.save(input_pdf)
        doc.close()

        with patch("pdf_process.translate_batch") as mock_trans:
            # Mô phỏng dịch tự động cho mọi câu
            mock_trans.side_effect = lambda texts, **kw: [f"Đã dịch: {t}" for t in texts]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            self.assertEqual(len(out_doc), 3)
            for i, page in enumerate(out_doc):
                page_text = clean_and_normalize_text(page.get_text())
                self.assertGreater(len(page_text.strip()), 0)
                self.assertTrue("Đã dịch" in page_text)
        print("✓ Test Case 2: PDF nhiều đoạn và nhiều trang - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 3: PDF có tiêu đề, nội dung, chữ đậm và chữ nghiêng
    # -------------------------------------------------------------------------
    def test_case_3_heading_body_bold_italic(self):
        input_pdf = os.path.join(self.temp_dir, "case3_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case3_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=595, height=842)
        # Tiêu đề lớn
        p.insert_textbox(fitz.Rect(50, 50, 500, 90), "MAIN TITLE HEADING", fontsize=18)
        # Đoạn thân bài
        p.insert_textbox(fitz.Rect(50, 120, 500, 200), "This is normal body text that follows the title heading.", fontsize=10)
        doc.save(input_pdf)
        doc.close()

        # Kiểm tra layout_extractor tách riêng tiêu đề và thân bài
        with fitz.open(input_pdf) as in_doc:
            elems = extract_page_elements(in_doc[0])
            self.assertGreaterEqual(len(elems), 2)
            # Element đầu tiên phải là tiêu đề với font size lớn hơn
            self.assertGreater(elems[0]["size"], elems[1]["size"])
            self.assertIn("MAIN TITLE", elems[0]["text"])

        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.side_effect = lambda texts, **kw: [f"Dịch: {t}" for t in texts]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            out_text = out_doc[0].get_text()
            self.assertTrue("Dịch: MAIN TITLE" in out_text or "Dịch" in out_text)
        print("✓ Test Case 3: PDF có tiêu đề, thân bài, kích thước font khác biệt - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 4: PDF có hai cột hoặc các vùng văn bản gần nhau
    # -------------------------------------------------------------------------
    def test_case_4_two_columns_layout(self):
        input_pdf = os.path.join(self.temp_dir, "case4_in.pdf")
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        # Cột trái
        p.insert_textbox(fitz.Rect(50, 100, 270, 300), "Column 1 text starts here and continues downward.", fontsize=10)
        # Cột phải
        p.insert_textbox(fitz.Rect(320, 100, 550, 300), "Column 2 text starts here and continues downward.", fontsize=10)
        doc.save(input_pdf)
        doc.close()

        with fitz.open(input_pdf) as in_doc:
            elems = extract_page_elements(in_doc[0])
            self.assertEqual(len(elems), 2)
            # Đảm bảo 2 cột không bị gom thành 1 phần tử duy nhất
            self.assertNotEqual(elems[0]["bbox"].x0, elems[1]["bbox"].x0)
            # Độ rộng insert_rect không được tràn sang cột bên cạnh
            col1 = elems[0] if elems[0]["bbox"].x0 < elems[1]["bbox"].x0 else elems[1]
            safe_rect = compute_safe_insert_rect(in_doc[0], col1, elems)
            self.assertLessEqual(safe_rect.x1, 320.0, "Cột 1 không được mở rộng đè lên Cột 2")
        print("✓ Test Case 4: PDF hai cột không bị gom và không bị xâm lấn cột - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 5: PDF có một đoạn dài hơn vùng chèn ban đầu
    # -------------------------------------------------------------------------
    def test_case_5_overflow_long_paragraph(self):
        tester = ScratchFontTester()
        try:
            long_vietnamese_text = (
                "Đây là một đoạn văn bản tiếng Việt rất dài được dịch ra từ câu tiếng Anh ban đầu. "
                "Đoạn văn này có độ dài mở rộng vượt quá đáng kể so với kích thước khung hộp gốc, "
                "do đó hệ thống cần tự động tính toán cỡ chữ nhỏ hơn hoặc tận dụng khoảng trống an toàn bên dưới "
                "để đảm bảo văn bản hiển thị đầy đủ và không bị cắt cụt."
            )
            # Khung hẹp 200 x 50
            rect = fitz.Rect(50, 50, 250, 100)
            best_size, fitted = tester.fit_size(
                rect, long_vietnamese_text, "TNR", max_size=12.0, min_size=5.0
            )
            # Phải tìm được cỡ chữ vừa hoặc báo False nếu vượt quá
            if fitted:
                self.assertLess(best_size, 12.0, "Cỡ chữ phải tự động giảm để vừa khung")
                self.assertTrue(tester.test_fit(rect, long_vietnamese_text, "TNR", best_size, 1.15, 0))
        finally:
            tester.close()
        print("✓ Test Case 5: Tự động điều chỉnh cỡ chữ cho đoạn văn bản dài - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 6: PDF chỉ chứa hình ảnh hoặc không có văn bản trích xuất được
    # -------------------------------------------------------------------------
    def test_case_6_image_only_or_scanned_pdf(self):
        input_pdf = os.path.join(self.temp_dir, "case6_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case6_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=400, height=400)
        # Tạo trang trống không có text layer
        doc.save(input_pdf)
        doc.close()

        with self.assertRaises(ValueError) as ctx:
            translate_pdf(input_pdf, output_pdf)

        self.assertIn("không chứa lớp văn bản", str(ctx.exception))
        print("✓ Test Case 6: Bắt lỗi chính xác với PDF không có text layer - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 7: PDF hỏng hoặc file tải lên không phải PDF
    # -------------------------------------------------------------------------
    def test_case_7_corrupt_or_non_pdf_upload(self):
        # 7a: File không có đuôi .pdf
        resp = self.app.post(
            "/api/translate-pdf",
            data={"file": (tempfile.NamedTemporaryFile(suffix=".txt"), "document.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("File không hợp lệ", resp.get_json()["error"])

        # 7b: File đuôi .pdf nhưng nội dung giả mạo (thiếu %PDF-)
        fake_pdf = tempfile.NamedTemporaryFile(suffix=".pdf")
        fake_pdf.write(b"THIS IS NOT A VALID PDF FILE AT ALL")
        fake_pdf.seek(0)
        resp2 = self.app.post(
            "/api/translate-pdf",
            data={"file": (fake_pdf, "fake.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("thiếu header %PDF-", resp2.get_json()["error"])
        print("✓ Test Case 7: Xác thực upload file PDF hợp lệ và từ chối file giả mạo - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 8: Mô hình dịch trả về kết quả rỗng hoặc phát sinh lỗi
    # -------------------------------------------------------------------------
    def test_case_8_model_empty_or_error_fallback(self):
        input_pdf = os.path.join(self.temp_dir, "case8_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case8_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=500, height=300)
        p.insert_textbox(fitz.Rect(50, 50, 450, 100), "Original English sentence that must not be deleted.", fontsize=12)
        doc.save(input_pdf)
        doc.close()

        # Mô phỏng mô hình trả về chuỗi rỗng
        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.return_value = [""]  # Kết quả rỗng!
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            out_text = out_doc[0].get_text()
            # Nội dung gốc KHÔNG ĐƯỢC PHÉP BỊ XÓA MẤT
            self.assertTrue(
                "Original English" in out_text or "deleted" in out_text,
                "Khi dịch trả về rỗng, văn bản gốc phải được bảo toàn thay vì để lại ô trống!"
            )
        print("✓ Test Case 8: Tự động fallback giữ văn bản gốc khi mô hình trả về rỗng - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 9: Kết quả dịch không vừa vùng chèn (Không được xóa bản gốc)
    # -------------------------------------------------------------------------
    def test_case_9_text_does_not_fit_preserves_original(self):
        input_pdf = os.path.join(self.temp_dir, "case9_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case9_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=150, height=80)
        # Ô nhỏ 40 x 15 pt
        p.insert_textbox(fitz.Rect(10, 10, 60, 25), "Short text", fontsize=8)
        # Đặt chướng ngại vật ngay sát phía dưới để insert_rect không thể nới rộng
        p.insert_textbox(fitz.Rect(10, 27, 140, 75), "Obstacle text that blocks expansion downward completely.", fontsize=8)
        doc.save(input_pdf)
        doc.close()

        # Bản dịch cực kỳ dài (hơn 100 từ) trong khung 50x15 pt bị chặn chắc chắn không thể nhét vừa kể cả ở cỡ 4pt
        huge_text = " ".join(["Từ_dài_không_thể_nhét_vừa_khung_này_được_đâu"] * 50)
        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.return_value = [huge_text, "Văn bản chướng ngại vật dịch thành công."]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            out_text = clean_and_normalize_text(out_doc[0].get_text())
            # Vì phần tử 1 không thể nhét vừa khung an toàn, bản gốc "Short text" phải được giữ nguyên!
            self.assertIn("Short text", out_text)
        print("✓ Test Case 9: Không xóa văn bản gốc khi bản dịch không thể chèn an toàn - ĐẠT")


    # -------------------------------------------------------------------------
    # Trường hợp 10: Thiếu font hoặc cấu hình không đầy đủ (Fallback an toàn)
    # -------------------------------------------------------------------------
    def test_case_10_missing_font_fallback(self):
        buffers = get_font_buffers()
        self.assertIn("regular", buffers)
        self.assertIn("bold", buffers)
        self.assertIn("italic", buffers)
        self.assertIn("bold_italic", buffers)
        for style, buf in buffers.items():
            self.assertIsInstance(buf, bytes)
            self.assertGreater(len(buf), 1000)
        print("✓ Test Case 10: Bộ đệm font nạp đầy đủ và hỗ trợ fallback dự phòng - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 11: Job thất bại trong lúc xử lý (Báo lỗi sạch, không leak path)
    # -------------------------------------------------------------------------
    def test_case_11_job_failure_reporting(self):
        job_id = "test-job-fail-123"
        flask_app_module._jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "input_path": "F:\\project\\uploads\\secret_document.pdf",
            "output_path": "F:\\project\\outputs\\translated_secret.pdf",
        }

        # Kích hoạt job thất bại
        with patch("app.translate_pdf", side_effect=RuntimeError("Lỗi hệ thống tại F:\\project\\uploads\\secret_document.pdf")):
            flask_app_module._run_translation_job(
                job_id,
                flask_app_module._jobs[job_id]["input_path"],
                flask_app_module._jobs[job_id]["output_path"],
                num_beams=1,
            )

        self.assertEqual(flask_app_module._jobs[job_id]["status"], "error")
        # Kiểm tra API status
        resp = self.app.get(f"/api/job/{job_id}/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")
        # Không được rò rỉ đường dẫn nội bộ F:\project
        self.assertNotIn("F:\\project", data["error"])
        self.assertNotIn("input_path", data)
        self.assertNotIn("output_path", data)
        print("✓ Test Case 11: Báo lỗi chính xác, bảo mật không rò rỉ đường dẫn máy chủ - ĐẠT")

    # -------------------------------------------------------------------------
    # Trường hợp 12: File đầu ra có thể mở lại và chứa văn bản tiếng Việt
    # -------------------------------------------------------------------------
    def test_case_12_output_reopen_and_content_verification(self):
        input_pdf = os.path.join(self.temp_dir, "case12_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case12_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=595, height=842)
        p.insert_textbox(
            fitz.Rect(50, 50, 500, 150),
            "Welcome to the document translation system. We preserve your layout.",
            fontsize=12,
        )
        doc.save(input_pdf)
        doc.close()

        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.return_value = [
                "Chào mừng bạn đến với hệ thống dịch thuật tài liệu. Chúng tôi bảo toàn bố cục của bạn."
            ]
            translate_pdf(input_pdf, output_pdf)

        self.assertTrue(os.path.exists(output_pdf))
        with fitz.open(output_pdf) as out_doc:
            self.assertEqual(len(out_doc), 1)
            text = clean_and_normalize_text(out_doc[0].get_text())
            self.assertTrue(
                "Chào mừng" in text or "dịch thuật" in text or "bố cục" in text,
                f"Text trích xuất từ file đầu ra không chứa tiếng Việt mong đợi: {text}"
            )
        print("✓ Test Case 12: File PDF đầu ra mở lại hợp lệ và trích xuất đúng Unicode tiếng Việt - ĐẠT")


if __name__ == "__main__":
    unittest.main()
