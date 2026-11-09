from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
import sys
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_file
import pymupdf as fitz
from werkzeug.utils import secure_filename

from pdf_process import translate_pdf

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)
JOB_RETENTION_SECONDS = 24 * 3600


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Dung lượng file vượt quá giới hạn cho phép (tối đa 50MB)! Vui lòng chọn file có kích thước nhỏ hơn."}), 413


def _path_in_folder(folder, filename):
    base_path = os.path.abspath(folder)
    path = os.path.abspath(os.path.join(folder, filename))
    return path if os.path.commonpath([path, base_path]) == base_path else None


def _cleanup_expired_jobs(now=None):
    now = time.time() if now is None else now
    expired = [
        jid for jid, info in list(_jobs.items())
        if info.get("status") in ("done", "error") and now - info.get("created_at", now) > JOB_RETENTION_SECONDS
    ]
    for jid in expired:
        info = _jobs.pop(jid, {})
        for path in (info.get("input_path"), info.get("output_path")):
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    print(f"Không thể dọn file job hết hạn {jid}: {exc}")


def _run_translation_job(job_id, input_path, output_path, num_beams, original_filename=None):
    if job_id in _jobs:
        _jobs[job_id].update(status="running", progress=5, message="Đang bắt đầu phân tích tài liệu...")

    try:
        translate_pdf(
            input_path,
            output_path,
            num_beams=num_beams,
            progress_callback=lambda pct, msg: _jobs[job_id].update(progress=pct, message=msg) if job_id in _jobs else None,
        )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Quá trình dịch không tạo được file PDF kết quả hợp lệ.")

        with fitz.open(output_path) as out_doc:
            if len(out_doc) == 0:
                raise RuntimeError("File PDF kết quả không chứa trang nào.")

        clean_orig = original_filename or os.path.basename(input_path)
        stored_out, stored_in = os.path.basename(output_path), os.path.basename(input_path)
        if job_id in _jobs:
            _jobs[job_id].update(
                status="done",
                progress=100,
                message="Dịch hoàn tất! Đang nạp bản xem trước...",
                result={
                    "success": True,
                    "job_id": job_id,
                    "download_url": f"/download/{stored_out}",
                    "output_name": f"translated_{clean_orig}",
                    "original_file": clean_orig,
                    "translated_file": stored_out,
                    "view_url": f"/view-pdf/outputs/{stored_out}",
                    "original_view_url": f"/view-pdf/uploads/{stored_in}",
                    "num_beams": num_beams,
                },
            )
    except Exception as e:
        err_msg = re.sub(r"[A-Za-z]:\\[^ \n\r\t]+|/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+", "[đường dẫn hệ thống]", str(e))
        print(f"Lỗi xử lý file PDF trong job {job_id}: {e}")
        if job_id in _jobs:
            _jobs[job_id].update(status="error", error=err_msg)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/translate-pdf", methods=["POST"])
def translate_api():
    _cleanup_expired_jobs()
    req_data = request.get_json(silent=True) or {}
    file = request.files.get("file") or request.files.get("pdfFile")
    job_id = str(uuid.uuid4())
    original_filename = input_path = stored_input_filename = None

    if file and file.filename != "":
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File không hợp lệ! Vui lòng chọn file có định dạng .pdf"}), 400

        header = file.read(1024)
        file.seek(0)
        if b"%PDF-" not in header[:1024]:
            return jsonify({"error": "File không phải là định dạng PDF hợp lệ (thiếu header %PDF-)!"}), 400

        safe_name = secure_filename(file.filename)
        original_filename = safe_name if safe_name and safe_name != ".pdf" else f"document_{int(time.time())}.pdf"
        stored_input_filename = f"{job_id}_{original_filename}"
        input_path = os.path.join(UPLOAD_FOLDER, stored_input_filename)
        file.save(input_path)

        try:
            with fitz.open(input_path) as check_doc:
                err = (
                    "File PDF rỗng, không chứa trang nào!" if len(check_doc) == 0
                    else "File PDF được bảo vệ bằng mật khẩu. Vui lòng mở khóa tài liệu trước khi dịch." if check_doc.is_encrypted
                    else None
                )
        except Exception as e:
            err = f"Không thể đọc file PDF (file bị hỏng hoặc sai cấu trúc): {e}"

        if err:
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({"error": err}), 400
    else:
        safe_chosen = secure_filename(req_data.get("filename") or request.form.get("filename") or "")
        candidate_path = os.path.join(UPLOAD_FOLDER, safe_chosen) if safe_chosen else ""
        if safe_chosen and os.path.isfile(candidate_path):
            original_filename = safe_chosen
            stored_input_filename = f"{job_id}_{safe_chosen}"
            input_path = os.path.join(UPLOAD_FOLDER, stored_input_filename)
            shutil.copy2(candidate_path, input_path)

    if not input_path or not original_filename:
        return jsonify({"error": "Không tìm thấy file PDF! Vui lòng tải lên file cần dịch."}), 400

    stored_output_filename = f"translated_{job_id}_{original_filename}"
    output_path = os.path.join(OUTPUT_FOLDER, stored_output_filename)
    raw_beams = str(request.form.get("num_beams") or req_data.get("num_beams") or 1)
    num_beams = int(raw_beams) if raw_beams in ("1", "2", "4") else 1

    _jobs[job_id] = {
        "status": "pending",
        "progress": 2,
        "message": "Đang chuẩn bị xử lý tài liệu...",
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


@app.route("/api/job/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    _cleanup_expired_jobs()
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Không tìm thấy job"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
    })


@app.route("/api/system-status", methods=["GET"])
def system_status():
    try:
        from pdf_engine import MODEL_NAME, get_device
        import torch

        dev = get_device()
        gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" and torch.cuda.is_available() else "CPU"
        return jsonify({"model": MODEL_NAME, "device": dev.upper(), "device_name": gpu_name, "status": "online"})
    except Exception as e:
        return jsonify({"status": "online", "model": "vinai/vinai-translate-en2vi", "device": "CPU", "error": str(e)})


@app.route("/view-pdf/<folder>/<filename>")
def view_pdf(folder, filename):
    folder, filename = secure_filename(folder), secure_filename(filename)
    if not filename:
        return jsonify({"error": "Tên file không hợp lệ"}), 400

    if folder in ("uploads", "outputs"):
        path = _path_in_folder(UPLOAD_FOLDER if folder == "uploads" else OUTPUT_FOLDER, filename)
        if path is None:
            return jsonify({"error": "Truy cập không hợp lệ"}), 403
    elif folder in ("job", "jobs"):
        path = (_jobs.get(filename) or {}).get("output_path")
        if not path:
            return jsonify({"error": "Không tìm thấy job"}), 404
    else:
        return jsonify({"error": "Thư mục không hợp lệ"}), 400

    if not os.path.isfile(path):
        return jsonify({"error": "File không tồn tại"}), 404
    return send_file(path, mimetype="application/pdf")


@app.route("/download/<filename>")
def download_file(filename):
    if filename in _jobs and _jobs[filename].get("output_path"):
        target_path = _jobs[filename]["output_path"]
        download_name = f"translated_{_jobs[filename].get('original_filename', 'document.pdf')}"
    else:
        clean_name = secure_filename(filename)
        if not clean_name:
            return jsonify({"error": "Tên file không hợp lệ"}), 400
        target_path = _path_in_folder(OUTPUT_FOLDER, clean_name)
        if target_path is None:
            return jsonify({"error": "Truy cập không hợp lệ"}), 403
        match = re.match(r"^translated_[0-9a-fA-F\-]{8,36}_(.+)$", clean_name)
        download_name = f"translated_{match.group(1)}" if match else clean_name

    if not target_path or not os.path.isfile(target_path):
        return jsonify({"error": "File không tồn tại"}), 404
    return send_file(target_path, as_attachment=True, download_name=download_name, mimetype="application/pdf")


if __name__ == "__main__":
    pdf_args = [arg for arg in sys.argv[1:] if arg.lower().endswith(".pdf")]
    if pdf_args:
        target = pdf_args[0]
        upload_cand = os.path.join(UPLOAD_FOLDER, target)
        target_path = target if os.path.isabs(target) else (os.path.abspath(target) if os.path.exists(target) else (upload_cand if os.path.exists(upload_cand) else os.path.abspath(target)))
        if not os.path.exists(target_path):
            print(f"[LỖI] Không tìm thấy file: {target_path}")
            sys.exit(1)

        out_path = os.path.join(OUTPUT_FOLDER, f"translated_{os.path.basename(target_path)}")
        print(f"Bắt đầu dịch file: {target_path}\nFile kết quả sẽ lưu tại: {out_path}")
        t_start = time.time()
        translate_pdf(target_path, out_path)
        print(f"Dịch hoàn tất sau {round(time.time() - t_start, 2)}s! Kết quả đã lưu tại: {out_path}")
    else:
        print("Khởi động server tại http://127.0.0.1:5000 ...\nĐang nạp model dịch trước khi mở server...")
        from pdf_engine import get_translation_model

        get_translation_model()
        print("Model dịch đã sẵn sàng.")
        debug_enabled = os.environ.get("DOCUVI_DEBUG", "").strip().lower() in {"1", "true", "yes"}
        app.run(host="127.0.0.1", port=5000, debug=debug_enabled, use_reloader=False)
