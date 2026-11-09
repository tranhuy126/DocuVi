import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import pymupdf as fitz

import app as flask_app_module
from pdf_engine import (
    ScratchFontTester,
    clean_and_normalize_text,
    compute_safe_insert_rect,
    extract_page_elements,
    get_font_buffers,
    group_lines_into_elements,
    join_lines_into_paragraph,
    pre_filter_text,
    split_into_sentences,
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
        input_pdf = os.path.join(self.temp_dir, "case1_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case1_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(50, 50, 500, 100),
            "Artificial Intelligence is advancing rapidly in modern society.",
            fontsize=12,
        )
        doc.save(input_pdf)
        doc.close()

        mocked_translations = ["Trí tuệ nhân tạo đang phát triển nhanh chóng trong xã hội hiện đại."]
        with patch("pdf_process.translate_batch", return_value=mocked_translations):
            translate_pdf(input_pdf, output_pdf)

        self.assertTrue(os.path.exists(output_pdf) and os.path.getsize(output_pdf) > 0)
        with fitz.open(output_pdf) as output_doc:
            self.assertEqual(len(output_doc), 1)
            extracted_text = clean_and_normalize_text(output_doc[0].get_text())
            self.assertTrue("Trí tuệ nhân tạo" in extracted_text or "phát triển" in extracted_text)

    def test_case_2_multipage_multipara_pdf(self):
        input_pdf = os.path.join(self.temp_dir, "case2_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case2_out.pdf")

        doc = fitz.open()
        for page_number in range(3):
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(
                fitz.Rect(50, 50, 500, 120),
                f"Page {page_number + 1} Section One. First detailed paragraph.",
                fontsize=11,
            )
            page.insert_textbox(
                fitz.Rect(50, 150, 500, 220),
                f"Page {page_number + 1} Section Two. Second detailed paragraph.",
                fontsize=11,
            )
        doc.save(input_pdf)
        doc.close()

        with patch(
            "pdf_process.translate_batch",
            side_effect=lambda texts, **kwargs: [f"Đã dịch: {text}" for text in texts],
        ):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as output_doc:
            self.assertEqual(len(output_doc), 3)
            for page in output_doc:
                self.assertIn("Đã dịch", clean_and_normalize_text(page.get_text()))

    def test_case_3_heading_body_bold_italic(self):
        input_pdf = os.path.join(self.temp_dir, "case3_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case3_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(50, 50, 500, 90), "MAIN TITLE HEADING", fontsize=18)
        page.insert_textbox(
            fitz.Rect(50, 120, 500, 200),
            "This is normal body text that follows the title heading.",
            fontsize=10,
        )
        doc.save(input_pdf)
        doc.close()

        with fitz.open(input_pdf) as input_doc:
            extracted_elements = extract_page_elements(input_doc[0])
            self.assertGreaterEqual(len(extracted_elements), 2)
            self.assertGreater(extracted_elements[0]["size"], extracted_elements[1]["size"])
            self.assertIn("MAIN TITLE", extracted_elements[0]["text"])

        with patch(
            "pdf_process.translate_batch",
            side_effect=lambda texts, **kwargs: [f"Dịch: {text}" for text in texts],
        ):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as output_doc:
            self.assertIn("Dịch", output_doc[0].get_text())

    def test_case_4_two_columns_layout(self):
        input_pdf = os.path.join(self.temp_dir, "case4_in.pdf")
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        page.insert_textbox(
            fitz.Rect(50, 100, 270, 300),
            "Column 1 text starts here and continues downward.",
            fontsize=10,
        )
        page.insert_textbox(
            fitz.Rect(320, 100, 550, 300),
            "Column 2 text starts here and continues downward.",
            fontsize=10,
        )
        doc.save(input_pdf)
        doc.close()

        with fitz.open(input_pdf) as input_doc:
            extracted_elements = extract_page_elements(input_doc[0])
            self.assertEqual(len(extracted_elements), 2)
            self.assertNotEqual(extracted_elements[0]["bbox"].x0, extracted_elements[1]["bbox"].x0)
            left_column_element = (
                extracted_elements[0]
                if extracted_elements[0]["bbox"].x0 < extracted_elements[1]["bbox"].x0
                else extracted_elements[1]
            )
            safe_rect = compute_safe_insert_rect(input_doc[0], left_column_element, extracted_elements)
            self.assertLessEqual(safe_rect.x1, 320.0)

    def test_case_5_overflow_long_paragraph(self):
        with ScratchFontTester() as font_tester:
            long_vietnamese_paragraph = (
                "Đây là một đoạn văn bản tiếng Việt rất dài được dịch ra từ câu tiếng Anh ban đầu. "
                "Đoạn văn này có độ dài mở rộng vượt quá đáng kể so với kích thước khung hộp gốc, "
                "do đó hệ thống cần tự động tính toán cỡ chữ nhỏ hơn hoặc tận dụng khoảng trống an toàn bên dưới "
                "để đảm bảo văn bản hiển thị đầy đủ và không bị cắt cụt."
            )
            target_rect = fitz.Rect(50, 50, 250, 100)
            best_font_size, is_fitted = font_tester.fit_size(
                target_rect,
                long_vietnamese_paragraph,
                "TNR",
                max_size=12.0,
                min_size=5.0,
            )
            if is_fitted:
                self.assertLess(best_font_size, 12.0)
                self.assertTrue(
                    font_tester.test_fit(
                        target_rect,
                        long_vietnamese_paragraph,
                        "TNR",
                        best_font_size,
                        1.15,
                        0,
                    )
                )

    def test_case_6_image_only_or_scanned_pdf(self):
        input_pdf = os.path.join(self.temp_dir, "case6_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case6_out.pdf")

        doc = fitz.open()
        doc.new_page(width=400, height=400)
        doc.save(input_pdf)
        doc.close()

        with self.assertRaises(ValueError) as error_context:
            translate_pdf(input_pdf, output_pdf)
        self.assertIn("không chứa lớp văn bản", str(error_context.exception))

    def test_case_7_corrupt_or_non_pdf_upload(self):
        non_pdf_response = self.app.post(
            "/api/translate-pdf",
            data={"file": (tempfile.NamedTemporaryFile(suffix=".txt"), "document.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(non_pdf_response.status_code, 400)
        self.assertIn("File không hợp lệ", non_pdf_response.get_json()["error"])

        fake_pdf_file = tempfile.NamedTemporaryFile(suffix=".pdf")
        fake_pdf_file.write(b"THIS IS NOT A VALID PDF FILE AT ALL")
        fake_pdf_file.seek(0)
        fake_pdf_response = self.app.post(
            "/api/translate-pdf",
            data={"file": (fake_pdf_file, "fake.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(fake_pdf_response.status_code, 400)
        self.assertIn("thiếu header %PDF-", fake_pdf_response.get_json()["error"])

    def test_case_8_model_empty_or_error_fallback(self):
        input_pdf = os.path.join(self.temp_dir, "case8_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case8_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=500, height=300)
        page.insert_textbox(
            fitz.Rect(50, 50, 450, 100),
            "Original English sentence that must not be deleted.",
            fontsize=12,
        )
        doc.save(input_pdf)
        doc.close()

        with patch("pdf_process.translate_batch", return_value=[""]):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as output_doc:
            output_text = output_doc[0].get_text()
            self.assertTrue("Original English" in output_text or "deleted" in output_text)

    def test_case_9_text_does_not_fit_preserves_original(self):
        input_pdf = os.path.join(self.temp_dir, "case9_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case9_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=150, height=80)
        page.insert_textbox(fitz.Rect(10, 10, 60, 25), "Short text", fontsize=8)
        page.insert_textbox(
            fitz.Rect(10, 27, 140, 75),
            "Obstacle text that blocks expansion downward completely.",
            fontsize=8,
        )
        doc.save(input_pdf)
        doc.close()

        oversized_translation = " ".join(["Từ_dài_không_thể_nhét_vừa_khung_này_được_đâu"] * 50)
        with patch(
            "pdf_process.translate_batch",
            return_value=[oversized_translation, "Văn bản chướng ngại vật dịch thành công."],
        ):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as output_doc:
            self.assertIn("Short text", clean_and_normalize_text(output_doc[0].get_text()))

    def test_case_10_missing_font_fallback(self):
        font_buffers = get_font_buffers()
        for style_name in ("regular", "bold", "italic", "bold_italic"):
            self.assertIn(style_name, font_buffers)
            self.assertIsInstance(font_buffers[style_name], bytes)
            self.assertGreater(len(font_buffers[style_name]), 1000)

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
        with patch(
            "app.translate_pdf",
            side_effect=RuntimeError("Lỗi hệ thống tại F:\\project\\uploads\\secret_document.pdf"),
        ):
            flask_app_module._run_translation_job(
                job_id,
                flask_app_module._jobs[job_id]["input_path"],
                flask_app_module._jobs[job_id]["output_path"],
                num_beams=1,
            )

        self.assertEqual(flask_app_module._jobs[job_id]["status"], "error")
        status_response = self.app.get(f"/api/job/{job_id}/status")
        self.assertEqual(status_response.status_code, 200)
        response_data = status_response.get_json()
        self.assertEqual(response_data["status"], "error")
        self.assertNotIn("F:\\project", response_data["error"])
        self.assertNotIn("input_path", response_data)
        self.assertNotIn("output_path", response_data)

    def test_case_12_output_reopen_and_content_verification(self):
        input_pdf = os.path.join(self.temp_dir, "case12_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case12_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(50, 50, 500, 150),
            "Welcome to the document translation system. We preserve your layout.",
            fontsize=12,
        )
        doc.save(input_pdf)
        doc.close()

        mocked_translations = [
            "Chào mừng bạn đến với hệ thống dịch thuật tài liệu. Chúng tôi bảo toàn bố cục của bạn."
        ]
        with patch("pdf_process.translate_batch", return_value=mocked_translations):
            translate_pdf(input_pdf, output_pdf)

        self.assertTrue(os.path.exists(output_pdf))
        with fitz.open(output_pdf) as output_doc:
            extracted_text = clean_and_normalize_text(output_doc[0].get_text())
            self.assertTrue(
                "Chào mừng" in extracted_text
                or "dịch thuật" in extracted_text
                or "bố cục" in extracted_text
            )

    def test_case_13_tab_space_formatting_and_short_titles(self):
        cleaned_header = clean_and_normalize_text("ACCOUNTABILITY BRIEF \t\t December, 2000")
        self.assertEqual(cleaned_header, "ACCOUNTABILITY BRIEF December, 2000")
        needs_translation, filtered_header = pre_filter_text(cleaned_header)
        self.assertTrue(needs_translation)
        self.assertEqual(split_into_sentences(filtered_header), ["ACCOUNTABILITY BRIEF December, 2000"])

        short_titles = (
            "ACCOUNTABILITY BRIEF",
            "December, 2000",
            "EXECUTIVE SUMMARY",
            "1. Introduction",
            "• Project Overview",
            "Note: General Findings",
        )
        for short_title in short_titles:
            title_needs_trans, filtered_title = pre_filter_text(clean_and_normalize_text(short_title))
            self.assertTrue(title_needs_trans and len(split_into_sentences(filtered_title)) > 0)

        decorative_strings = (
            ". .",
            ".                                            .",
            "-------",
            "• • •",
            "* * *",
        )
        for decorative_text in decorative_strings:
            should_translate, _ = pre_filter_text(clean_and_normalize_text(decorative_text))
            self.assertFalse(should_translate)

    def test_case_14_borderless_table_row_preservation(self):
        table_rows = (
            (100, (("Project Alpha", 50, 200), ("Smith", 250, 300), ("$50,000", 350, 400))),
            (120, (("Project Beta", 50, 200), ("Johnson", 250, 300), ("$75,000", 350, 400))),
        )
        raw_lines = [
            {
                "text": cell_text,
                "bbox": fitz.Rect(x0, y0, x1, y0 + 15),
                "spans": [{"text": cell_text, "size": 9.0, "style": "regular", "color": (0, 0, 0)}],
            }
            for y0, row_cells in table_rows
            for cell_text, x0, x1 in row_cells
        ]
        subgroups = group_lines_into_elements(raw_lines)
        self.assertEqual(len(subgroups), 6)
        extracted_texts = [subgroup[0]["text"] for subgroup in subgroups]
        self.assertIn("$50,000", extracted_texts)
        self.assertIn("$75,000", extracted_texts)
        self.assertNotIn("$50,000 $75,000", extracted_texts)

    def test_case_15_tiny_bbox_retention_regression(self):
        input_pdf = os.path.join(self.temp_dir, "case15_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case15_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(50, 50, 500, 100), "Main content of the article.", fontsize=11)
        page.insert_textbox(fitz.Rect(50, 200, 75, 215), "TN", fontsize=8)
        page.insert_textbox(fitz.Rect(80, 200, 500, 218), "Right obstacle block.", fontsize=10)
        page.insert_textbox(fitz.Rect(50, 220, 500, 300), "Obstacle block below.", fontsize=10)
        doc.save(input_pdf)
        doc.close()

        def _mock_translate_batch(batch, **kwargs):
            translations = []
            for source_text in batch:
                if "Main content" in source_text:
                    translations.append("Nội dung chính của bài báo dịch thành công.")
                elif "TN" in source_text:
                    translations.append(
                        "Bản dịch tiếng Việt quá dài chắc chắn không thể nhét vừa ô 10x11pt này"
                    )
                else:
                    translations.append("Văn bản chướng ngại vật dịch bình thường.")
            return translations

        with patch("pdf_process.translate_batch", side_effect=_mock_translate_batch):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as output_doc:
            output_text = clean_and_normalize_text(output_doc[0].get_text())
            self.assertTrue(
                "Nội dung chính" in output_text
                and "chướng ngại vật" in output_text
                and "TN" in output_text
            )
            self.assertNotIn("quá dài chắc chắn", output_text)

    def test_case_16_insertion_failure_restores_original_page(self):
        input_pdf = os.path.join(self.temp_dir, "case16_in.pdf")
        output_pdf = os.path.join(self.temp_dir, "case16_out.pdf")

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(60, 60, 500, 140),
            "Original English paragraph for the restore test.",
            fontsize=12,
        )
        doc.save(input_pdf)
        doc.close()

        with (
            patch(
                "pdf_process.translate_batch",
                side_effect=lambda texts, **kwargs: ["Đoạn văn tiếng Việt." for _ in texts],
            ),
            patch.object(fitz.Page, "insert_textbox", return_value=-1),
        ):
            translate_pdf(input_pdf, output_pdf)

        with fitz.open(output_pdf) as output_doc:
            self.assertEqual(len(output_doc), 1)
            extracted_text = " ".join(output_doc[0].get_text().split())

        self.assertIn("Original English paragraph", extracted_text)
        self.assertNotIn("Đoạn văn tiếng Việt", extracted_text)
        print("✓ Test Case 16: Chèn thất bại thì khôi phục nguyên trang gốc - ĐẠT")

    def test_case_17_list_items_split_and_marker_kept(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="TNR", fontbuffer=get_font_buffers()["regular"])
        page.insert_textbox(
            fitz.Rect(60, 60, 500, 200),
            "• First list item about security\n"
            "• Second list item about funding\n"
            "• Third list item about planning",
            fontsize=11,
            fontname="TNR",
            lineheight=1.1,
        )
        extracted_elements = extract_page_elements(page)
        self.assertEqual(len(extracted_elements), 3)
        self.assertTrue(all(element.get("list_marker") == "•" for element in extracted_elements))
        print("✓ Test Case 17: Danh sách gạch đầu dòng được tách và giữ ký hiệu - ĐẠT")

    def test_case_18_bare_number_list_split(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="TNR", fontbuffer=get_font_buffers()["regular"])
        list_items = [
            "First item about replacing chemicals",
            "Second item about regional collaboration",
            "Third item about vulnerability assessments",
        ]
        current_y = 100
        for item_number, item_text in enumerate(list_items, start=1):
            page.insert_text((60, current_y), str(item_number), fontname="TNR", fontsize=11)
            page.insert_text((78, current_y), item_text, fontname="TNR", fontsize=11)
            current_y += 18

        extracted_elements = extract_page_elements(page)
        self.assertEqual(len(extracted_elements), 3)
        self.assertEqual(
            [element.get("list_marker") for element in extracted_elements],
            ["1.", "2.", "3."],
        )
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
        extracted_elements = extract_page_elements(page)
        self.assertEqual(len(extracted_elements), 1)
        self.assertGreaterEqual(len(extracted_elements[0]["original_lines"]), 3)
        print("✓ Test Case 19: Đoạn có từ đậm mở đầu không bị cắt - ĐẠT")


if __name__ == "__main__":
    unittest.main()

