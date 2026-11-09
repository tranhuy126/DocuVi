import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import pymupdf as fitz

import app as flask_app_module
from pdf_engine import (
    ScratchFontTester, clean_and_normalize_text, compute_safe_insert_rect, extract_page_elements,
    get_font_buffers, group_lines_into_elements, join_lines_into_paragraph, pre_filter_text, split_into_sentences,
)
from pdf_process import translate_pdf, translate_pdf_document


class TestDocuViComprehensiveSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.app = flask_app_module.app.test_client()
        self.app.testing = True

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_case_1_short_valid_pdf(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case1_in.pdf"), os.path.join(self.temp_dir, "case1_out.pdf")
        doc = fitz.open()
        doc.new_page(width=595, height=842).insert_textbox(fitz.Rect(50, 50, 500, 100), "Artificial Intelligence is advancing rapidly in modern society.", fontsize=12)
        doc.save(input_pdf)
        doc.close()
        with patch("pdf_process.translate_batch", return_value=["Trí tuệ nhân tạo đang phát triển nhanh chóng trong xã hội hiện đại."]):
            translate_pdf(input_pdf, output_pdf)
        self.assertTrue(os.path.exists(output_pdf) and os.path.getsize(output_pdf) > 0)
        with fitz.open(output_pdf) as out_doc:
            self.assertEqual(len(out_doc), 1)
            text = clean_and_normalize_text(out_doc[0].get_text())
            self.assertTrue("Trí tuệ nhân tạo" in text or "phát triển" in text)

    def test_case_2_multipage_multipara_pdf(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case2_in.pdf"), os.path.join(self.temp_dir, "case2_out.pdf")
        doc = fitz.open()
        for page_num in range(3):
            p = doc.new_page(width=595, height=842)
            p.insert_textbox(fitz.Rect(50, 50, 500, 120), f"Page {page_num + 1} Section One. First detailed paragraph.", fontsize=11)
            p.insert_textbox(fitz.Rect(50, 150, 500, 220), f"Page {page_num + 1} Section Two. Second detailed paragraph.", fontsize=11)
        doc.save(input_pdf)
        doc.close()
        with patch("pdf_process.translate_batch", side_effect=lambda texts, **kw: [f"Đã dịch: {t}" for t in texts]):
            translate_pdf(input_pdf, output_pdf)
        with fitz.open(output_pdf) as out_doc:
            self.assertEqual(len(out_doc), 3)
            for page in out_doc:
                self.assertIn("Đã dịch", clean_and_normalize_text(page.get_text()))

    def test_case_3_heading_body_bold_italic(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case3_in.pdf"), os.path.join(self.temp_dir, "case3_out.pdf")
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
        with patch("pdf_process.translate_batch", side_effect=lambda texts, **kw: [f"Dịch: {t}" for t in texts]):
            translate_pdf(input_pdf, output_pdf)
        with fitz.open(output_pdf) as out_doc:
            self.assertIn("Dịch", out_doc[0].get_text())

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
            self.assertLessEqual(compute_safe_insert_rect(in_doc[0], col1, elems).x1, 320.0)

    def test_case_5_overflow_long_paragraph(self):
        with ScratchFontTester() as tester:
            long_vn = (
                "Đây là một đoạn văn bản tiếng Việt rất dài được dịch ra từ câu tiếng Anh ban đầu. "
                "Đoạn văn này có độ dài mở rộng vượt quá đáng kể so với kích thước khung hộp gốc, "
                "do đó hệ thống cần tự động tính toán cỡ chữ nhỏ hơn hoặc tận dụng khoảng trống an toàn bên dưới "
                "để đảm bảo văn bản hiển thị đầy đủ và không bị cắt cụt."
            )
            rect = fitz.Rect(50, 50, 250, 100)
            best_size, fitted = tester.fit_size(rect, long_vn, "TNR", max_size=12.0, min_size=5.0)
            if fitted:
                self.assertLess(best_size, 12.0)
                self.assertTrue(tester.test_fit(rect, long_vn, "TNR", best_size, 1.15, 0))

    def test_case_6_image_only_or_scanned_pdf(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case6_in.pdf"), os.path.join(self.temp_dir, "case6_out.pdf")
        doc = fitz.open()
        doc.new_page(width=400, height=400)
        doc.save(input_pdf)
        doc.close()
        with self.assertRaises(ValueError) as ctx:
            translate_pdf(input_pdf, output_pdf)
        self.assertIn("không chứa lớp văn bản", str(ctx.exception))

    def test_case_7_corrupt_or_non_pdf_upload(self):
        resp = self.app.post("/api/translate-pdf", data={"file": (tempfile.NamedTemporaryFile(suffix=".txt"), "document.txt")}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("File không hợp lệ", resp.get_json()["error"])
        fake_pdf = tempfile.NamedTemporaryFile(suffix=".pdf")
        fake_pdf.write(b"THIS IS NOT A VALID PDF FILE AT ALL")
        fake_pdf.seek(0)
        resp2 = self.app.post("/api/translate-pdf", data={"file": (fake_pdf, "fake.pdf")}, content_type="multipart/form-data")
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("thiếu header %PDF-", resp2.get_json()["error"])

    def test_case_8_model_empty_or_error_fallback(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case8_in.pdf"), os.path.join(self.temp_dir, "case8_out.pdf")
        doc = fitz.open()
        doc.new_page(width=500, height=300).insert_textbox(fitz.Rect(50, 50, 450, 100), "Original English sentence that must not be deleted.", fontsize=12)
        doc.save(input_pdf)
        doc.close()
        with patch("pdf_process.translate_batch", return_value=[""]):
            translate_pdf(input_pdf, output_pdf)
        with fitz.open(output_pdf) as out_doc:
            out_text = out_doc[0].get_text()
            self.assertTrue("Original English" in out_text or "deleted" in out_text)

    def test_case_9_text_does_not_fit_preserves_original(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case9_in.pdf"), os.path.join(self.temp_dir, "case9_out.pdf")
        doc = fitz.open()
        p = doc.new_page(width=150, height=80)
        p.insert_textbox(fitz.Rect(10, 10, 60, 25), "Short text", fontsize=8)
        p.insert_textbox(fitz.Rect(10, 27, 140, 75), "Obstacle text that blocks expansion downward completely.", fontsize=8)
        doc.save(input_pdf)
        doc.close()
        huge_text = " ".join(["Từ_dài_không_thể_nhét_vừa_khung_này_được_đâu"] * 50)
        with patch("pdf_process.translate_batch", return_value=[huge_text, "Văn bản chướng ngại vật dịch thành công."]):
            translate_pdf(input_pdf, output_pdf)
        with fitz.open(output_pdf) as out_doc:
            self.assertIn("Short text", clean_and_normalize_text(out_doc[0].get_text()))

    def test_case_10_missing_font_fallback(self):
        buffers = get_font_buffers()
        for style in ("regular", "bold", "italic", "bold_italic"):
            self.assertIn(style, buffers)
            self.assertIsInstance(buffers[style], bytes)
            self.assertGreater(len(buffers[style]), 1000)

    def test_case_11_job_failure_reporting(self):
        job_id = "test-job-fail-123"
        flask_app_module._jobs[job_id] = {
            "status": "pending", "result": None, "error": None, "created_at": time.time(),
            "input_path": "F:\\project\\uploads\\secret_document.pdf", "output_path": "F:\\project\\outputs\\translated_secret.pdf",
        }
        with patch("app.translate_pdf", side_effect=RuntimeError("Lỗi hệ thống tại F:\\project\\uploads\\secret_document.pdf")):
            flask_app_module._run_translation_job(job_id, flask_app_module._jobs[job_id]["input_path"], flask_app_module._jobs[job_id]["output_path"], num_beams=1)
        self.assertEqual(flask_app_module._jobs[job_id]["status"], "error")
        resp = self.app.get(f"/api/job/{job_id}/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")
        self.assertNotIn("F:\\project", data["error"])
        self.assertNotIn("input_path", data)
        self.assertNotIn("output_path", data)

    def test_case_12_output_reopen_and_content_verification(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case12_in.pdf"), os.path.join(self.temp_dir, "case12_out.pdf")
        doc = fitz.open()
        doc.new_page(width=595, height=842).insert_textbox(fitz.Rect(50, 50, 500, 150), "Welcome to the document translation system. We preserve your layout.", fontsize=12)
        doc.save(input_pdf)
        doc.close()
        with patch("pdf_process.translate_batch", return_value=["Chào mừng bạn đến với hệ thống dịch thuật tài liệu. Chúng tôi bảo toàn bố cục của bạn."]):
            translate_pdf(input_pdf, output_pdf)
        self.assertTrue(os.path.exists(output_pdf))
        with fitz.open(output_pdf) as out_doc:
            text = clean_and_normalize_text(out_doc[0].get_text())
            self.assertTrue("Chào mừng" in text or "dịch thuật" in text or "bố cục" in text)

    def test_case_13_tab_space_formatting_and_short_titles(self):
        c1 = clean_and_normalize_text("ACCOUNTABILITY BRIEF \t\t December, 2000")
        self.assertEqual(c1, "ACCOUNTABILITY BRIEF December, 2000")
        v1, f1 = pre_filter_text(c1)
        self.assertTrue(v1)
        self.assertEqual(split_into_sentences(f1), ["ACCOUNTABILITY BRIEF December, 2000"])
        for st in ("ACCOUNTABILITY BRIEF", "December, 2000", "EXECUTIVE SUMMARY", "1. Introduction", "• Project Overview", "Note: General Findings"):
            v, f = pre_filter_text(clean_and_normalize_text(st))
            self.assertTrue(v and len(split_into_sentences(f)) > 0)
        for dec in (". .", ".                                            .", "-------", "• • •", "* * *"):
            self.assertFalse(pre_filter_text(clean_and_normalize_text(dec))[0])

    def test_case_14_borderless_table_row_preservation(self):
        raw_lines = [
            {"text": t, "bbox": fitz.Rect(x0, y0, x1, y0 + 15), "spans": [{"text": t, "size": 9.0, "style": "regular", "color": (0, 0, 0)}]}
            for y0, row in ((100, (("Project Alpha", 50, 200), ("Smith", 250, 300), ("$50,000", 350, 400))), (120, (("Project Beta", 50, 200), ("Johnson", 250, 300), ("$75,000", 350, 400))))
            for t, x0, x1 in row
        ]
        subgroups = group_lines_into_elements(raw_lines)
        self.assertEqual(len(subgroups), 6)
        extracted_texts = [sg[0]["text"] for sg in subgroups]
        self.assertIn("$50,000", extracted_texts)
        self.assertIn("$75,000", extracted_texts)
        self.assertNotIn("$50,000 $75,000", extracted_texts)

    def test_case_15_tiny_bbox_retention_regression(self):
        input_pdf, output_pdf = os.path.join(self.temp_dir, "case15_in.pdf"), os.path.join(self.temp_dir, "case15_out.pdf")
        doc = fitz.open()
        p = doc.new_page(width=595, height=842)
        p.insert_textbox(fitz.Rect(50, 50, 500, 100), "Main content of the article.", fontsize=11)
        p.insert_textbox(fitz.Rect(50, 200, 75, 215), "TN", fontsize=8)
        p.insert_textbox(fitz.Rect(80, 200, 500, 218), "Right obstacle block.", fontsize=10)
        p.insert_textbox(fitz.Rect(50, 220, 500, 300), "Obstacle block below.", fontsize=10)
        doc.save(input_pdf)
        doc.close()
        mock_fn = lambda batch, **kw: [
            "Nội dung chính của bài báo dịch thành công." if "Main content" in b
            else ("Bản dịch tiếng Việt quá dài chắc chắn không thể nhét vừa ô 10x11pt này" if "TN" in b else "Văn bản chướng ngại vật dịch bình thường.")
            for b in batch
        ]
        with patch("pdf_process.translate_batch", side_effect=mock_fn):
            translate_pdf(input_pdf, output_pdf)
        with fitz.open(output_pdf) as out_doc:
            out_text = clean_and_normalize_text(out_doc[0].get_text())
            self.assertTrue("Nội dung chính" in out_text and "chướng ngại vật" in out_text and "TN" in out_text)
            self.assertNotIn("quá dài chắc chắn", out_text)

    def test_case_16_insertion_failure_restores_original_page(self):
        input_pdf = os.path.join(self.temp_dir, "case16_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case16_out.pdf")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(60, 60, 500, 140),
                                "Original English paragraph for the restore test.", fontsize=12)
        doc.save(input_pdf)
        doc.close()

        with patch("pdf_process.translate_batch",
            side_effect=lambda texts, **kw: ["Đoạn văn tiếng Việt." for _ in texts]), \
            patch.object(fitz.Page, "insert_textbox", return_value=-1):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as out:
            self.assertEqual(len(out), 1)
            text = " ".join(out[0].get_text().split())
        self.assertIn("Original English paragraph", text)
        self.assertNotIn("Đoạn văn tiếng Việt", text)
        print("✓ Test Case 16: Chèn thất bại thì khôi phục nguyên trang gốc - ĐẠT")

    def test_case_17_list_items_split_and_marker_kept(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="TNR", fontbuffer=get_font_buffers()["regular"])
        page.insert_textbox(
            fitz.Rect(60, 60, 500, 200),
            "• First list item about security\n• Second list item about funding\n• Third list item about planning",
            fontsize=11, fontname="TNR", lineheight=1.1,
        )
        elems = extract_page_elements(page)
        self.assertEqual(len(elems), 3)
        self.assertTrue(all(e.get("list_marker") == "•" for e in elems))
        print("✓ Test Case 17: Danh sách gạch đầu dòng được tách và giữ ký hiệu - ĐẠT")
  
    def test_case_18_bare_number_list_split(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="TNR", fontbuffer=get_font_buffers()["regular"])
        items = ["First item about replacing chemicals",
                 "Second item about regional collaboration",
                 "Third item about vulnerability assessments"]
        y = 100
        for n, text in enumerate(items, start=1):
            page.insert_text((60, y), str(n), fontname="TNR", fontsize=11)
            page.insert_text((78, y), text, fontname="TNR", fontsize=11)
            y += 18
        elems = extract_page_elements(page)
        self.assertEqual(len(elems), 3)
        self.assertEqual([e.get("list_marker") for e in elems], ["1.", "2.", "3."])
        print("✓ Test Case 18: Danh sách đánh số không dấu chấm được tách đúng - ĐẠT")

    def test_case_19_bold_lead_in_stays_in_paragraph(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_htmlbox(
            fitz.Rect(60, 60, 330, 200),
            "<p style='font-size:11pt'><b>Creativity.</b> Children who have ADHD can be marvelously "
            "creative and imaginative. The child who daydreams and has ten different thoughts at once "
            "can become a master problem-solver.</p>",
        )
        elems = extract_page_elements(page)
        self.assertEqual(len(elems), 1)
        self.assertGreaterEqual(len(elems[0]["original_lines"]), 3)
        print("✓ Test Case 19: Đoạn có từ đậm mở đầu không bị cắt - ĐẠT") 

if __name__ == "__main__":
    unittest.main()
