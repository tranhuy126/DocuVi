import os
import re
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import pymupdf as fitz
from pdf_process import translate_pdf, translate_batch
from pdf_engine import clean_and_normalize_text, pre_filter_text, split_into_sentences

# Ép mã hóa UTF-8 trên Windows console để in log tiếng Việt không bị lỗi font
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__)

# Giới hạn kích thước tải lên tối đa 50MB để đảm bảo an toàn cho máy chủ
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'Dung lượng file vượt quá giới hạn cho phép (tối đa 50MB)! Vui lòng chọn file có kích thước nhỏ hơn.'
    }), 413


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Giới hạn 1 worker để tránh nghẽn/tràn VRAM do mô hình chạy trên GPU
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)


def _run_translation_job(job_id, input_path, output_path, num_beams, original_filename=None):
    """Thực thi dịch PDF trong thread pool worker và cập nhật trạng thái job."""
    if job_id in _jobs:
        _jobs[job_id]["status"] = "running"
    try:
        translate_pdf(input_path, output_path, num_beams=num_beams)

        # Kiểm chứng file PDF kết quả thực sự tồn tại, có dung lượng > 0 và mở được
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Quá trình dịch không tạo được file PDF kết quả hợp lệ.")

        with fitz.open(output_path) as out_doc:
            if len(out_doc) == 0:
                raise RuntimeError("File PDF kết quả không chứa trang nào.")

        clean_orig = original_filename or os.path.basename(input_path)
        stored_out = os.path.basename(output_path)
        stored_in = os.path.basename(input_path)
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {
                "success": True,
                "job_id": job_id,
                "download_url": f"/download/{stored_out}",
                "output_name": f"translated_{clean_orig}",
                "original_file": clean_orig,
                "translated_file": stored_out,
                "view_url": f"/view-pdf/outputs/{stored_out}",
                "original_view_url": f"/view-pdf/uploads/{stored_in}",
                "num_beams": num_beams,
            }
    except Exception as e:
        err_msg = str(e)
        # Loại bỏ các đường dẫn ổ đĩa nội bộ của máy chủ để không làm lộ cấu trúc server
        err_msg = re.sub(r"[A-Za-z]:\\[^ \n\r\t]+", "[đường dẫn hệ thống]", err_msg)
        err_msg = re.sub(r"/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+", "[đường dẫn hệ thống]", err_msg)
        print(f"Lỗi xử lý file PDF trong job {job_id}: {e}")
        if job_id in _jobs:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = err_msg


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/translate-pdf', methods=['POST'])
def translate_api():
    """Khởi tạo tác vụ dịch PDF bất đồng bộ với UUID định danh duy nhất cho từng job."""
    req_data = request.get_json(silent=True) or {}
    file = request.files.get('file') or request.files.get('pdfFile')

    job_id = str(uuid.uuid4())
    original_filename = None
    input_path = None
    stored_input_filename = None

    if file and file.filename != '':
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File không hợp lệ! Vui lòng chọn file có định dạng .pdf'}), 400

        # Kiểm tra magic bytes %PDF- ngay từ stream để tránh file giả mạo đuôi mở rộng
        header = file.read(1024)
        if not header.startswith(b'%PDF-') and b'%PDF-' not in header[:1024]:
            return jsonify({'error': 'File không phải là định dạng PDF hợp lệ (thiếu header %PDF-)!'}), 400
        file.seek(0)

        safe_name = secure_filename(file.filename)
        if not safe_name or safe_name == '.pdf':
            safe_name = f"document_{int(time.time())}.pdf"

        original_filename = safe_name
        stored_input_filename = f"{job_id}_{safe_name}"
        input_path = os.path.join(UPLOAD_FOLDER, stored_input_filename)
        file.save(input_path)

        # Kiểm tra mở thử bằng PyMuPDF để bắt ngay file hỏng / rỗng / mã hóa mật khẩu
        try:
            with fitz.open(input_path) as check_doc:
                if len(check_doc) == 0:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    return jsonify({'error': 'File PDF rỗng, không chứa trang nào!'}), 400
                if check_doc.is_encrypted:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    return jsonify({'error': 'File PDF được bảo vệ bằng mật khẩu. Vui lòng mở khóa tài liệu trước khi dịch.'}), 400
        except Exception as e:
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'error': f'Không thể đọc file PDF (file bị hỏng hoặc sai cấu trúc): {e}'}), 400
    else:
        # Hỗ trợ kiểm thử nhanh với file mẫu có sẵn trên server
        chosen_name = req_data.get('filename') or request.form.get('filename')
        if chosen_name:
            safe_chosen = secure_filename(chosen_name)
            candidate_path = os.path.join(UPLOAD_FOLDER, safe_chosen)
            if os.path.isfile(candidate_path):
                original_filename = safe_chosen
                stored_input_filename = f"{job_id}_{safe_chosen}"
                input_path = os.path.join(UPLOAD_FOLDER, stored_input_filename)
                shutil.copy2(candidate_path, input_path)

    if not input_path or not original_filename:
        return jsonify({'error': 'Không tìm thấy file PDF! Vui lòng tải lên file cần dịch.'}), 400

    stored_output_filename = f"translated_{job_id}_{original_filename}"
    output_path = os.path.join(OUTPUT_FOLDER, stored_output_filename)

    try:
        num_beams = int(request.form.get('num_beams') or req_data.get('num_beams') or 1)
        if num_beams not in [1, 2, 4]:
            num_beams = 1
    except Exception:
        num_beams = 1

    _jobs[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "input_path": input_path,
        "output_path": output_path,
        "original_filename": original_filename,
        "stored_input_filename": stored_input_filename,
        "stored_output_filename": stored_output_filename,
    }
    _executor.submit(_run_translation_job, job_id, input_path, output_path, num_beams, original_filename)

    return jsonify({"job_id": job_id}), 202


@app.route('/api/job/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    now = time.time()
    # Dọn dẹp job đã xong hoặc lỗi sau 1 giờ để tránh tràn RAM
    expired_ids = [
        jid for jid, info in _jobs.items()
        if info.get("status") in ("done", "error") and now - info.get("created_at", 0) > 3600
    ]
    for jid in expired_ids:
        _jobs.pop(jid, None)

    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Không tìm thấy job"}), 404

    # Chỉ trả về thông tin trạng thái cần thiết cho frontend, không để lộ đường dẫn nội bộ máy chủ
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
    })



@app.route('/api/translate-text', methods=['POST'])
def translate_text_api():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    try:
        num_beams = int(data.get('num_beams') or 1)
        if num_beams not in [1, 2, 4]:
            num_beams = 1
    except Exception:
        num_beams = 1

    if not text:
        return jsonify({'translation': ''})

    # Giới hạn kích thước văn bản dịch thuần tối đa 10,000 ký tự để bảo vệ server
    if len(text) > 10000:
        return jsonify({'error': 'Độ dài văn bản vượt quá giới hạn (tối đa 10,000 ký tự mỗi lượt).'}), 400

    try:
        # Tiền xử lý văn bản: chuẩn hóa Unicode và phân tách theo đoạn / câu
        paragraphs = text.splitlines()
        all_sentences_to_translate: list[str] = []
        structure = []

        for para in paragraphs:
            cleaned_para = clean_and_normalize_text(para)
            if not cleaned_para:
                structure.append(('empty', ''))
                continue

            need_trans_para, filtered_para = pre_filter_text(cleaned_para)
            if not need_trans_para:
                structure.append(('raw', filtered_para))
                continue

            # Phân tách thành từng câu con bảo vệ số thứ tự mục và từ viết tắt
            sents = split_into_sentences(filtered_para)
            para_sents = []
            for sent in sents:
                need_trans_sent, filtered_sent = pre_filter_text(sent)
                if not need_trans_sent:
                    para_sents.append(('raw', filtered_sent))
                else:
                    idx = len(all_sentences_to_translate)
                    all_sentences_to_translate.append(filtered_sent)
                    para_sents.append(('trans', idx))
            structure.append(('para', para_sents))

        # Dịch đồng thời theo batch qua GPU nếu có câu cần dịch
        translated_sentences = []
        if all_sentences_to_translate:
            translated_sentences = translate_batch(
                all_sentences_to_translate,
                batch_size=32,
                num_beams=num_beams,
                clear_cache=False,
            )

        # Ghép kết quả dịch theo đúng cấu trúc đoạn và câu ban đầu
        result_lines = []
        for item_type, content in structure:
            if item_type == 'empty':
                result_lines.append('')
            elif item_type == 'raw':
                result_lines.append(content)
            elif item_type == 'para':
                line_parts = [
                    translated_sentences[val] if stype == 'trans' else val
                    for stype, val in content
                ]
                result_lines.append(' '.join(line_parts))

        final_translation = '\n'.join(result_lines)
        return jsonify({'translation': final_translation})
    except Exception as e:
        print(f"Lỗi dịch văn bản: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system-status', methods=['GET'])
def system_status():
    try:
        from pdf_engine import MODEL_NAME, get_device
        import torch
        dev = get_device()
        gpu_name = torch.cuda.get_device_name(0) if dev == 'cuda' and torch.cuda.is_available() else 'CPU'
        return jsonify({
            'model': MODEL_NAME,
            'device': dev.upper(),
            'device_name': gpu_name,
            'status': 'online'
        })
    except Exception as e:
        return jsonify({'status': 'online', 'model': 'vinai/vinai-translate-en2vi', 'device': 'CPU', 'error': str(e)})


@app.route('/view-pdf/<folder>/<filename>')
def view_pdf(folder, filename):
    folder = secure_filename(folder)
    filename = secure_filename(filename)

    if not filename:
        return jsonify({'error': 'Tên file không hợp lệ'}), 400

    if folder == 'uploads':
        path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
        if os.path.commonpath([path, os.path.abspath(UPLOAD_FOLDER)]) != os.path.abspath(UPLOAD_FOLDER):
            return jsonify({'error': 'Truy cập không hợp lệ'}), 403
    elif folder == 'outputs':
        path = os.path.abspath(os.path.join(OUTPUT_FOLDER, filename))
        if os.path.commonpath([path, os.path.abspath(OUTPUT_FOLDER)]) != os.path.abspath(OUTPUT_FOLDER):
            return jsonify({'error': 'Truy cập không hợp lệ'}), 403
    elif folder in ('job', 'jobs'):
        if filename in _jobs and _jobs[filename].get("output_path"):
            path = _jobs[filename]["output_path"]
        else:
            return jsonify({'error': 'Không tìm thấy job'}), 404
    else:
        return jsonify({'error': 'Thư mục không hợp lệ'}), 400

    if not os.path.isfile(path):
        return jsonify({'error': 'File không tồn tại'}), 404

    return send_file(path, mimetype='application/pdf')


@app.route('/download/<filename>')
def download_file(filename):
    # Hỗ trợ tải bằng job_id hoặc tên file trực tiếp
    target_path = None
    download_name = None

    if filename in _jobs and _jobs[filename].get("output_path"):
        target_path = _jobs[filename]["output_path"]
        orig_name = _jobs[filename].get("original_filename", "document.pdf")
        download_name = f"translated_{orig_name}"
    else:
        clean_name = secure_filename(filename)
        if not clean_name:
            return jsonify({'error': 'Tên file không hợp lệ'}), 400

        candidate = os.path.abspath(os.path.join(OUTPUT_FOLDER, clean_name))
        if os.path.commonpath([candidate, os.path.abspath(OUTPUT_FOLDER)]) != os.path.abspath(OUTPUT_FOLDER):
            return jsonify({'error': 'Truy cập không hợp lệ'}), 403

        if os.path.isfile(candidate):
            target_path = candidate
            # Loại bỏ tiền tố UUID nội bộ nếu có để file tải về có tên hiển thị đẹp cho người dùng
            match = re.match(r"^translated_[0-9a-fA-F\-]{8,36}_(.+)$", clean_name)
            if match:
                download_name = f"translated_{match.group(1)}"
            else:
                download_name = clean_name

    if not target_path or not os.path.isfile(target_path):
        return jsonify({'error': 'File không tồn tại'}), 404

    return send_file(target_path, as_attachment=True, download_name=download_name, mimetype='application/pdf')



if __name__ == '__main__':
    # Hỗ trợ chạy trực tiếp từ CLI: python app.py [path/to/file.pdf]
    pdf_args = [arg for arg in sys.argv[1:] if arg.lower().endswith('.pdf')]
    if pdf_args:
        target = pdf_args[0]
        if not os.path.isabs(target):
            if os.path.exists(target):
                target_path = os.path.abspath(target)
            elif os.path.exists(os.path.join(UPLOAD_FOLDER, target)):
                target_path = os.path.join(UPLOAD_FOLDER, target)
            else:
                target_path = os.path.abspath(target)
        else:
            target_path = target

        if not os.path.exists(target_path):
            print(f"[LỖI] Không tìm thấy file: {target_path}")
            sys.exit(1)

        fname = os.path.basename(target_path)
        out_path = os.path.join(OUTPUT_FOLDER, f"translated_{fname}")
        print(f"Bắt đầu dịch file: {target_path}")
        print(f"File kết quả sẽ lưu tại: {out_path}")
        t_start = time.time()
        translate_pdf(target_path, out_path)
        print(f"Dịch hoàn tất sau {round(time.time() - t_start, 2)}s! Kết quả đã lưu tại: {out_path}")
    else:
        print("Khởi động server tại http://127.0.0.1:5000 ...")
        app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)