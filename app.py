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
    return (
        jsonify(
            {
                "error": (
                    "Dung lượng file vượt quá giới hạn cho phép (tối đa 50MB)! "
                    "Vui lòng chọn file có kích thước nhỏ hơn."
                )
            }
        ),
        413,
    )


def _path_in_folder(folder: str, filename: str) -> str | None:
    base_folder_path = os.path.abspath(folder)
    resolved_file_path = os.path.abspath(os.path.join(folder, filename))
    if os.path.commonpath([resolved_file_path, base_folder_path]) == base_folder_path:
        return resolved_file_path
    return None


def _cleanup_expired_jobs(current_timestamp: float | None = None):
    now = time.time() if current_timestamp is None else current_timestamp
    expired_job_ids = [
        job_id
        for job_id, job_info in list(_jobs.items())
        if job_info.get("status") in ("done", "error")
        and now - job_info.get("created_at", now) > JOB_RETENTION_SECONDS
    ]
    for expired_job_id in expired_job_ids:
        job_info = _jobs.pop(expired_job_id, {})
        for file_path in (job_info.get("input_path"), job_info.get("output_path")):
            if file_path and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError as os_error:
                    print(f"Không thể dọn file job hết hạn {expired_job_id}: {os_error}")


def _run_translation_job(
    job_id: str,
    input_path: str,
    output_path: str,
    num_beams: int,
    original_filename: str | None = None,
):
    if job_id in _jobs:
        _jobs[job_id].update(
            status="running",
            progress=5,
            message="Đang bắt đầu phân tích tài liệu...",
        )

    try:
        def _update_job_progress(progress_percent: int, status_message: str):
            if job_id in _jobs:
                _jobs[job_id].update(progress=progress_percent, message=status_message)

        translate_pdf(
            input_path,
            output_path,
            num_beams=num_beams,
            progress_callback=_update_job_progress,
        )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Quá trình dịch không tạo được file PDF kết quả hợp lệ.")

        with fitz.open(output_path) as output_document:
            if len(output_document) == 0:
                raise RuntimeError("File PDF kết quả không chứa trang nào.")

        clean_original_name = original_filename or os.path.basename(input_path)
        stored_output_name = os.path.basename(output_path)
        stored_input_name = os.path.basename(input_path)

        if job_id in _jobs:
            _jobs[job_id].update(
                status="done",
                progress=100,
                message="Dịch hoàn tất! Đang nạp bản xem trước...",
                result={
                    "success": True,
                    "job_id": job_id,
                    "download_url": f"/download/{stored_output_name}",
                    "output_name": f"translated_{clean_original_name}",
                    "original_file": clean_original_name,
                    "translated_file": stored_output_name,
                    "view_url": f"/view-pdf/outputs/{stored_output_name}",
                    "original_view_url": f"/view-pdf/uploads/{stored_input_name}",
                    "num_beams": num_beams,
                },
            )
    except Exception as error:
        sanitized_error_message = re.sub(
            r"[A-Za-z]:\\[^ \n\r\t]+|/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+",
            "[đường dẫn hệ thống]",
            str(error),
        )
        print(f"Lỗi xử lý file PDF trong job {job_id}: {error}")
        if job_id in _jobs:
            _jobs[job_id].update(status="error", error=sanitized_error_message)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/translate-pdf", methods=["POST"])
def translate_api():
    _cleanup_expired_jobs()
    request_json = request.get_json(silent=True) or {}
    uploaded_file = request.files.get("file") or request.files.get("pdfFile")
    job_id = str(uuid.uuid4())
    original_filename = None
    input_path = None
    stored_input_filename = None

    if uploaded_file and uploaded_file.filename != "":
        if not uploaded_file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "File không hợp lệ! Vui lòng chọn file có định dạng .pdf"}), 400

        header_bytes = uploaded_file.read(1024)
        uploaded_file.seek(0)
        if b"%PDF-" not in header_bytes[:1024]:
            return (
                jsonify({"error": "File không phải là định dạng PDF hợp lệ (thiếu header %PDF-)!"}),
                400,
            )

        safe_filename = secure_filename(uploaded_file.filename)
        if safe_filename and safe_filename != ".pdf":
            original_filename = safe_filename
        else:
            original_filename = f"document_{int(time.time())}.pdf"

        stored_input_filename = f"{job_id}_{original_filename}"
        input_path = os.path.join(UPLOAD_FOLDER, stored_input_filename)
        uploaded_file.save(input_path)

        validation_error = None
        try:
            with fitz.open(input_path) as check_document:
                if len(check_document) == 0:
                    validation_error = "File PDF rỗng, không chứa trang nào!"
                elif check_document.is_encrypted:
                    validation_error = (
                        "File PDF được bảo vệ bằng mật khẩu. "
                        "Vui lòng mở khóa tài liệu trước khi dịch."
                    )
        except Exception as error:
            validation_error = f"Không thể đọc file PDF (file bị hỏng hoặc sai cấu trúc): {error}"

        if validation_error:
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({"error": validation_error}), 400
    else:
        requested_filename = request_json.get("filename") or request.form.get("filename") or ""
        safe_chosen_filename = secure_filename(requested_filename)
        if safe_chosen_filename:
            candidate_path = os.path.join(UPLOAD_FOLDER, safe_chosen_filename)
        else:
            candidate_path = ""

        if safe_chosen_filename and os.path.isfile(candidate_path):
            original_filename = safe_chosen_filename
            stored_input_filename = f"{job_id}_{safe_chosen_filename}"
            input_path = os.path.join(UPLOAD_FOLDER, stored_input_filename)
            shutil.copy2(candidate_path, input_path)

    if not input_path or not original_filename:
        return jsonify({"error": "Không tìm thấy file PDF! Vui lòng tải lên file cần dịch."}), 400

    stored_output_filename = f"translated_{job_id}_{original_filename}"
    output_path = os.path.join(OUTPUT_FOLDER, stored_output_filename)
    raw_beams_value = str(request.form.get("num_beams") or request_json.get("num_beams") or 1)
    num_beams = int(raw_beams_value) if raw_beams_value in ("1", "2", "4") else 1

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
def get_job_status(job_id: str):
    _cleanup_expired_jobs()
    job_info = _jobs.get(job_id)
    if not job_info:
        return jsonify({"error": "Không tìm thấy job"}), 404

    return jsonify(
        {
            "job_id": job_id,
            "status": job_info.get("status"),
            "progress": job_info.get("progress", 0),
            "message": job_info.get("message", ""),
            "result": job_info.get("result"),
            "error": job_info.get("error"),
            "created_at": job_info.get("created_at"),
        }
    )


@app.route("/api/system-status", methods=["GET"])
def system_status():
    try:
        from pdf_engine import MODEL_NAME, get_device
        import torch

        current_device = get_device()
        if current_device == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        else:
            device_name = "CPU"

        return jsonify(
            {
                "model": MODEL_NAME,
                "device": current_device.upper(),
                "device_name": device_name,
                "status": "online",
            }
        )
    except Exception as error:
        return jsonify(
            {
                "status": "online",
                "model": "vinai/vinai-translate-en2vi",
                "device": "CPU",
                "error": str(error),
            }
        )


@app.route("/view-pdf/<folder>/<filename>")
def view_pdf(folder: str, filename: str):
    safe_folder = secure_filename(folder)
    safe_filename = secure_filename(filename)
    if not safe_filename:
        return jsonify({"error": "Tên file không hợp lệ"}), 400

    if safe_folder in ("uploads", "outputs"):
        target_directory = UPLOAD_FOLDER if safe_folder == "uploads" else OUTPUT_FOLDER
        file_path = _path_in_folder(target_directory, safe_filename)
        if file_path is None:
            return jsonify({"error": "Truy cập không hợp lệ"}), 403
    elif safe_folder in ("job", "jobs"):
        file_path = (_jobs.get(safe_filename) or {}).get("output_path")
        if not file_path:
            return jsonify({"error": "Không tìm thấy job"}), 404
    else:
        return jsonify({"error": "Thư mục không hợp lệ"}), 400

    if not os.path.isfile(file_path):
        return jsonify({"error": "File không tồn tại"}), 404
    return send_file(file_path, mimetype="application/pdf")


@app.route("/download/<filename>")
def download_file(filename: str):
    if filename in _jobs and _jobs[filename].get("output_path"):
        target_path = _jobs[filename]["output_path"]
        original_name = _jobs[filename].get("original_filename", "document.pdf")
        download_name = f"translated_{original_name}"
    else:
        clean_filename = secure_filename(filename)
        if not clean_filename:
            return jsonify({"error": "Tên file không hợp lệ"}), 400

        target_path = _path_in_folder(OUTPUT_FOLDER, clean_filename)
        if target_path is None:
            return jsonify({"error": "Truy cập không hợp lệ"}), 403

        uuid_prefix_match = re.match(r"^translated_[0-9a-fA-F\-]{8,36}_(.+)$", clean_filename)
        if uuid_prefix_match:
            download_name = f"translated_{uuid_prefix_match.group(1)}"
        else:
            download_name = clean_filename

    if not target_path or not os.path.isfile(target_path):
        return jsonify({"error": "File không tồn tại"}), 404

    return send_file(
        target_path,
        as_attachment=True,
        download_name=download_name,
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    cli_pdf_arguments = [arg for arg in sys.argv[1:] if arg.lower().endswith(".pdf")]
    if cli_pdf_arguments:
        requested_target = cli_pdf_arguments[0]
        upload_folder_candidate = os.path.join(UPLOAD_FOLDER, requested_target)

        if os.path.isabs(requested_target):
            target_path = requested_target
        elif os.path.exists(requested_target):
            target_path = os.path.abspath(requested_target)
        elif os.path.exists(upload_folder_candidate):
            target_path = upload_folder_candidate
        else:
            target_path = os.path.abspath(requested_target)

        if not os.path.exists(target_path):
            print(f"[LỖI] Không tìm thấy file: {target_path}")
            sys.exit(1)

        output_path = os.path.join(OUTPUT_FOLDER, f"translated_{os.path.basename(target_path)}")
        print(f"Bắt đầu dịch file: {target_path}\nFile kết quả sẽ lưu tại: {output_path}")
        start_time = time.time()
        translate_pdf(target_path, output_path)
        elapsed_seconds = round(time.time() - start_time, 2)
        print(f"Dịch hoàn tất sau {elapsed_seconds}s! Kết quả đã lưu tại: {output_path}")
    else:
        print("Khởi động server tại http://127.0.0.1:5000 ...\nĐang nạp model dịch trước khi mở server...")
        from pdf_engine import get_translation_model

        get_translation_model()
        print("Model dịch đã sẵn sàng.")
        debug_env_value = os.environ.get("DOCUVI_DEBUG", "").strip().lower()
        debug_enabled = debug_env_value in {"1", "true", "yes"}
        app.run(host="127.0.0.1", port=5000, debug=debug_enabled, use_reloader=False)

