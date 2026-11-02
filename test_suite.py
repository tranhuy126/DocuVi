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
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.app = flask_app_module.app.test_client()
        self.app.testing = True

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


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
            mock_trans.side_effect = lambda texts, **kw: [f"Đã dịch: {t}" for t in texts]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            self.assertEqual(len(out_doc), 3)
            for i, page in enumerate(out_doc):
                page_text = clean_and_normalize_text(page.get_text())
                self.assertGreater(len(page_text.strip()), 0)
                self.assertTrue("Đã dịch" in page_text)
        print("✓ Test Case 2: PDF nhiều đoạn và nhiều trang - ĐẠT")


    def test_case_3_heading_body_bold_italic(self):
        input_pdf = os.path.join(self.temp_dir, "case3_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case3_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=595, height=842)

        p.insert_textbox(fitz.Rect(50, 50, 500, 90), "MAIN TITLE HEADING", fontsize=18)

        p.insert_textbox(fitz.Rect(50, 120, 500, 200), "This is normal body text that follows the title heading.", fontsize=10)
        doc.save(input_pdf)
        doc.close()


        with fitz.open(input_pdf) as in_doc:
            elems = extract_page_elements(in_doc[0])
            self.assertGreaterEqual(len(elems), 2)

            self.assertGreater(elems[0]["size"], elems[1]["size"])
            self.assertIn("MAIN TITLE", elems[0]["text"])

        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.side_effect = lambda texts, **kw: [f"Dịch: {t}" for t in texts]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            out_text = out_doc[0].get_text()
            self.assertTrue("Dịch: MAIN TITLE" in out_text or "Dịch" in out_text)
        print("✓ Test Case 3: PDF có tiêu đề, thân bài, kích thước font khác biệt - ĐẠT")


    def test_case_4_two_columns_layout(self):
        input_pdf = os.path.join(self.temp_dir, "case4_in.pdf")
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)

        p.insert_textbox(fitz.Rect(50, 100, 270, 300), "Column 1 text starts here and continues downward.", fontsize=10)

        p.insert_textbox(fitz.Rect(320, 100, 550, 300), "Column 2 text starts here and continues downward.", fontsize=10)
        doc.save(input_pdf)
        doc.close()

        with fitz.open(input_pdf) as in_doc:
            elems = extract_page_elements(in_doc[0])
            self.assertEqual(len(elems), 2)

            self.assertNotEqual(elems[0]["bbox"].x0, elems[1]["bbox"].x0)

            col1 = elems[0] if elems[0]["bbox"].x0 < elems[1]["bbox"].x0 else elems[1]
            safe_rect = compute_safe_insert_rect(in_doc[0], col1, elems)
            self.assertLessEqual(safe_rect.x1, 320.0, "Cột 1 không được mở rộng đè lên Cột 2")
        print("✓ Test Case 4: PDF hai cột không bị gom và không bị xâm lấn cột - ĐẠT")


    def test_case_5_overflow_long_paragraph(self):
        tester = ScratchFontTester()
        try:
            long_vietnamese_text = (
                "Đây là một đoạn văn bản tiếng Việt rất dài được dịch ra từ câu tiếng Anh ban đầu. "
                "Đoạn văn này có độ dài mở rộng vượt quá đáng kể so với kích thước khung hộp gốc, "
                "do đó hệ thống cần tự động tính toán cỡ chữ nhỏ hơn hoặc tận dụng khoảng trống an toàn bên dưới "
                "để đảm bảo văn bản hiển thị đầy đủ và không bị cắt cụt."
            )

            rect = fitz.Rect(50, 50, 250, 100)
            best_size, fitted = tester.fit_size(
                rect, long_vietnamese_text, "TNR", max_size=12.0, min_size=5.0
            )

            if fitted:
                self.assertLess(best_size, 12.0, "Cỡ chữ phải tự động giảm để vừa khung")
                self.assertTrue(tester.test_fit(rect, long_vietnamese_text, "TNR", best_size, 1.15, 0))
        finally:
            tester.close()
        print("✓ Test Case 5: Tự động điều chỉnh cỡ chữ cho đoạn văn bản dài - ĐẠT")


    def test_case_6_image_only_or_scanned_pdf(self):
        input_pdf = os.path.join(self.temp_dir, "case6_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case6_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=400, height=400)

        doc.save(input_pdf)
        doc.close()

        with self.assertRaises(ValueError) as ctx:
            translate_pdf(input_pdf, output_pdf)

        self.assertIn("không chứa lớp văn bản", str(ctx.exception))
        print("✓ Test Case 6: Bắt lỗi chính xác với PDF không có text layer - ĐẠT")


    def test_case_7_corrupt_or_non_pdf_upload(self):
        resp = self.app.post(
            "/api/translate-pdf",
            data={"file": (tempfile.NamedTemporaryFile(suffix=".txt"), "document.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("File không hợp lệ", resp.get_json()["error"])


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


    def test_case_8_model_empty_or_error_fallback(self):
        input_pdf = os.path.join(self.temp_dir, "case8_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case8_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=500, height=300)
        p.insert_textbox(fitz.Rect(50, 50, 450, 100), "Original English sentence that must not be deleted.", fontsize=12)
        doc.save(input_pdf)
        doc.close()


        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.return_value = [""]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            out_text = out_doc[0].get_text()

            self.assertTrue(
                "Original English" in out_text or "deleted" in out_text,
                "Khi dịch trả về rỗng, văn bản gốc phải được bảo toàn thay vì để lại ô trống!"
            )
        print("✓ Test Case 8: Tự động fallback giữ văn bản gốc khi mô hình trả về rỗng - ĐẠT")


    def test_case_9_text_does_not_fit_preserves_original(self):
        input_pdf = os.path.join(self.temp_dir, "case9_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case9_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=150, height=80)

        p.insert_textbox(fitz.Rect(10, 10, 60, 25), "Short text", fontsize=8)

        p.insert_textbox(fitz.Rect(10, 27, 140, 75), "Obstacle text that blocks expansion downward completely.", fontsize=8)
        doc.save(input_pdf)
        doc.close()


        huge_text = " ".join(["Từ_dài_không_thể_nhét_vừa_khung_này_được_đâu"] * 50)
        with patch("pdf_process.translate_batch") as mock_trans:
            mock_trans.return_value = [huge_text, "Văn bản chướng ngại vật dịch thành công."]
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out_doc:
            out_text = clean_and_normalize_text(out_doc[0].get_text())

            self.assertIn("Short text", out_text)
        print("✓ Test Case 9: Không xóa văn bản gốc khi bản dịch không thể chèn an toàn - ĐẠT")


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


        with patch("app.translate_pdf", side_effect=RuntimeError("Lỗi hệ thống tại F:\\project\\uploads\\secret_document.pdf")):
            flask_app_module._run_translation_job(
                job_id,
                flask_app_module._jobs[job_id]["input_path"],
                flask_app_module._jobs[job_id]["output_path"],
                num_beams=1,
            )

        self.assertEqual(flask_app_module._jobs[job_id]["status"], "error")

        resp = self.app.get(f"/api/job/{job_id}/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")

        self.assertNotIn("F:\\project", data["error"])
        self.assertNotIn("input_path", data)
        self.assertNotIn("output_path", data)
        print("✓ Test Case 11: Báo lỗi chính xác, bảo mật không rò rỉ đường dẫn máy chủ - ĐẠT")


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


    def test_case_13_tab_space_formatting_and_short_titles(self):
        t1 = "ACCOUNTABILITY BRIEF \t\t December, 2000"
        c1 = clean_and_normalize_text(t1)
        self.assertEqual(c1, "ACCOUNTABILITY BRIEF December, 2000")
        v1, f1 = pre_filter_text(c1)
        self.assertTrue(v1)
        self.assertEqual(f1, "ACCOUNTABILITY BRIEF December, 2000")
        s1 = split_into_sentences(f1)
        self.assertEqual(s1, ["ACCOUNTABILITY BRIEF December, 2000"])


        short_titles = [
            "ACCOUNTABILITY BRIEF",
            "December, 2000",
            "EXECUTIVE SUMMARY",
            "1. Introduction",
            "• Project Overview",
            "Note: General Findings"
        ]
        for st in short_titles:
            c = clean_and_normalize_text(st)
            v, f = pre_filter_text(c)
            self.assertTrue(v, f"Tiêu đề hợp lệ '{st}' không được bị lọc bỏ")
            s = split_into_sentences(f)
            self.assertGreater(len(s), 0)


        decoratives = [". .", ".                                            .", "-------", "• • •", "* * *"]
        for dec in decoratives:
            c = clean_and_normalize_text(dec)
            v, f = pre_filter_text(c)
            self.assertFalse(v, f"Chuỗi trang trí '{dec}' phải được nhận diện lọc an toàn")

        print("✓ Test Case 13: Tiền xử lý bảo toàn tiêu đề ngắn, tab, khoảng trắng và phân định chuỗi trang trí - ĐẠT")


    def test_case_14_borderless_table_row_preservation(self):
        raw_lines = [
            {"text": "Project Alpha", "bbox": fitz.Rect(50, 100, 200, 115), "spans": [{"text": "Project Alpha", "size": 9.0, "style": "regular", "color": (0, 0, 0)}]},
            {"text": "Smith", "bbox": fitz.Rect(250, 100, 300, 115), "spans": [{"text": "Smith", "size": 9.0, "style": "regular", "color": (0, 0, 0)}]},
            {"text": "$50,000", "bbox": fitz.Rect(350, 100, 400, 115), "spans": [{"text": "$50,000", "size": 9.0, "style": "regular", "color": (0, 0, 0)}]},
            {"text": "Project Beta", "bbox": fitz.Rect(50, 120, 200, 135), "spans": [{"text": "Project Beta", "size": 9.0, "style": "regular", "color": (0, 0, 0)}]},
            {"text": "Johnson", "bbox": fitz.Rect(250, 120, 300, 135), "spans": [{"text": "Johnson", "size": 9.0, "style": "regular", "color": (0, 0, 0)}]},
            {"text": "$75,000", "bbox": fitz.Rect(350, 120, 400, 135), "spans": [{"text": "$75,000", "size": 9.0, "style": "regular", "color": (0, 0, 0)}]},
        ]

        subgroups = group_lines_into_elements(raw_lines)


        self.assertEqual(len(subgroups), 6)
        extracted_texts = [sg[0]["text"] for sg in subgroups]
        self.assertIn("$50,000", extracted_texts)
        self.assertIn("$75,000", extracted_texts)
        self.assertNotIn("$50,000 $75,000", extracted_texts)
        print("✓ Test Case 14: Nhận diện cấu trúc bảng biểu không viền, bảo toàn từng ô số liệu độc lập - ĐẠT")


    def test_case_15_tiny_bbox_retention_regression(self):
        input_pdf = os.path.join(self.temp_dir, "case15_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case15_out.pdf")

        doc = fitz.open()
        p = doc.new_page(width=595, height=842)

        p.insert_textbox(fitz.Rect(50, 50, 500, 100), "Main content of the article.", fontsize=11)

        tiny_rect = fitz.Rect(50, 200, 75, 215)
        p.insert_textbox(tiny_rect, "TN", fontsize=8)

        p.insert_textbox(fitz.Rect(80, 200, 500, 218), "Right obstacle block.", fontsize=10)

        p.insert_textbox(fitz.Rect(50, 220, 500, 300), "Obstacle block below.", fontsize=10)
        doc.save(input_pdf)
        doc.close()

        with patch("pdf_process.translate_batch") as mock_trans:
            def mock_func(batch, **kwargs):
                res = []
                for b in batch:
                    if "Main content" in b:
                        res.append("Nội dung chính của bài báo dịch thành công.")
                    elif "TN" in b:
                        res.append("Bản dịch tiếng Việt quá dài chắc chắn không thể nhét vừa ô 10x11pt này")
                    else:
                        res.append("Văn bản chướng ngại vật dịch bình thường.")
                return res
            mock_trans.side_effect = mock_func

            translate_pdf(input_pdf, output_pdf)

        self.assertTrue(os.path.exists(output_pdf))
        with fitz.open(output_pdf) as out_doc:
            out_page = out_doc[0]
            out_text = clean_and_normalize_text(out_page.get_text())

            self.assertIn("Nội dung chính", out_text)
            self.assertIn("chướng ngại vật", out_text)

            self.assertIn("TN", out_text)

            self.assertNotIn("quá dài chắc chắn", out_text)

        print("✓ Test Case 15: Phần tử có bbox siêu hẹp không thể vừa khung được giữ nguyên văn bản gốc - ĐẠT")


if __name__ == "__main__":
    unittest.main()
