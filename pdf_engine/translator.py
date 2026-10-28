import re
import sys
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


MODEL_NAME = "vinai/vinai-translate-en2vi"
_device = None
_tokenizer = None
_model = None
_vi_lang_id = None
_model_init_lock = threading.Lock()
model_lock = threading.Lock()


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        if _device == "cuda":
            # Tự động benchmark để chọn thuật toán convolution/attention tối ưu cho GPU
            torch.backends.cudnn.benchmark = True
    return _device


def get_translation_model():
    """Khởi tạo lười và nạp mô hình dịch vào bộ nhớ với cơ chế thread-safe."""
    global _model, _tokenizer, _vi_lang_id, _device
    if _model is None or _tokenizer is None:
        with _model_init_lock:
            if _model is None or _tokenizer is None:
                cur_device = get_device()
                print(
                    f"[Khởi tạo AI] Đang nạp mô hình dịch {MODEL_NAME} trên {cur_device.upper()}..."
                )
                tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="en_XX")
                if cur_device == "cuda":
                    try:
                        mdl = AutoModelForSeq2SeqLM.from_pretrained(
                            MODEL_NAME, torch_dtype=torch.float16
                        ).to(cur_device)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        print(f"[CẢNH BÁO] Không đủ VRAM trên GPU ({e}), tự động chuyển sang nạp trên CPU...")
                        cur_device = "cpu"
                        _device = "cpu"
                        mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")
                else:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(cur_device)
                mdl.eval()
                _tokenizer = tok
                _model = mdl
                _vi_lang_id = tok.lang_code_to_id["vi_VN"]
                print(f"[Khởi tạo AI] Nạp mô hình hoàn tất trên {cur_device.upper()}!")
    return _model, _tokenizer, _vi_lang_id


def __getattr__(name: str):
    if name == "model":
        mdl, _, _ = get_translation_model()
        return mdl
    elif name == "tokenizer":
        _, tok, _ = get_translation_model()
        return tok
    elif name == "device":
        return get_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _chunk_text_if_needed(text: str, max_chars: int = 500) -> list[str]:
    """Phân tách văn bản quá dài tại các mốc dấu câu tự nhiên để tránh bị tokenizer cắt cụt (truncation)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Thử tách theo dấu chấm phẩy, hai chấm, gạch ngang dài
    sub_parts = []
    candidates = re.split(r"(?<=[;:—])\s+", text)
    if len(candidates) > 1:
        current = ""
        for cand in candidates:
            if current and len(current) + len(cand) > max_chars:
                sub_parts.append(current.strip())
                current = cand
            else:
                current = f"{current} {cand}".strip() if current else cand
        if current:
            sub_parts.append(current.strip())
        return sub_parts

    # Thử tách theo dấu phẩy
    candidates = re.split(r"(?<=,)\s+", text)
    if len(candidates) > 1:
        current = ""
        for cand in candidates:
            if current and len(current) + len(cand) > max_chars:
                sub_parts.append(current.strip())
                current = cand
            else:
                current = f"{current} {cand}".strip() if current else cand
        if current:
            sub_parts.append(current.strip())
        return sub_parts

    # Tách theo ranh giới từ nếu không có dấu câu
    words = text.split()
    current = ""
    for w in words:
        if current and len(current) + len(w) + 1 > max_chars:
            sub_parts.append(current.strip())
            current = w
        else:
            current = f"{current} {w}".strip() if current else w
    if current:
        sub_parts.append(current.strip())
    return sub_parts if sub_parts else [text]


def _translate_raw_batch(
    model, tokenizer, vi_lang_id, batch: list[str], cur_device: str, num_beams: int
) -> list[str]:
    """Hàm phụ thực hiện suy luận một batch nhỏ với bộ tokenizer và model."""
    encoded = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(cur_device)

    gen_kwargs = {
        "decoder_start_token_id": vi_lang_id,
        "num_beams": num_beams,
        "max_length": 512,
    }
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True

    generated = model.generate(**encoded, **gen_kwargs)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return decoded



def translate_batch(
    texts: list[str],
    batch_size: int = 32,
    num_beams: int = 1,
    clear_cache: bool = False,
) -> list[str]:
    """
    Dịch batch câu en-vi trên GPU/CPU:
    - Gom câu theo độ dài giảm padding thừa.
    - Chia nhỏ câu quá dài tránh truncation làm mất văn bản.
    - Xử lý Out-of-Memory (OOM) tự động giảm batch size hoặc fallback CPU.
    - Bảo toàn câu gốc tiếng Anh nếu mô hình trả về chuỗi rỗng.
    """
    if not texts:
        return []

    model, tokenizer, vi_lang_id = get_translation_model()
    cur_device = get_device()

    # Bước 1: Kiểm tra và chia nhỏ các câu quá dài để không bị mất nội dung do truncation
    flat_units = []
    unit_map = []  # mapping từ flat_units về index gốc trong texts
    for orig_idx, t in enumerate(texts):
        chunks = _chunk_text_if_needed(t, max_chars=450)
        for c in chunks:
            flat_units.append(c)
            unit_map.append(orig_idx)

    # Bước 2: Sắp xếp theo độ dài để tối ưu gom nhóm tensor khi padding
    indexed_units = sorted(enumerate(flat_units), key=lambda x: len(x[1]))
    sorted_unit_indices = [x[0] for x in indexed_units]
    sorted_units = [x[1] for x in indexed_units]

    sorted_results = [""] * len(sorted_units)

    with model_lock:
        with torch.inference_mode():
            i = 0
            current_batch_size = max(1, batch_size)
            while i < len(sorted_units):
                batch = sorted_units[i : i + current_batch_size]
                try:
                    decoded = _translate_raw_batch(
                        model, tokenizer, vi_lang_id, batch, cur_device, num_beams
                    )
                    for b_offset, res_text in enumerate(decoded):
                        sorted_results[i + b_offset] = res_text
                    i += len(batch)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    err_msg = str(e).lower()
                    if "out of memory" in err_msg or "cuda" in err_msg:
                        print(f"[CẢNH BÁO] Tràn VRAM CUDA tại batch kích thước {current_batch_size}: {e}")
                        if cur_device == "cuda":
                            torch.cuda.empty_cache()
                        if current_batch_size > 1:
                            # Thử lại batch hiện tại với batch size nhỏ hơn
                            current_batch_size = max(1, current_batch_size // 2)
                            print(f"[TỰ PHỤC HỒI] Giảm batch_size xuống {current_batch_size} và tiếp tục...")
                            continue
                        else:
                            # Khi batch_size = 1 vẫn OOM, tạm đưa câu hiện tại sang CPU dịch
                            print(f"[TỰ PHỤC HỒI] Dịch câu đơn trên CPU để vượt qua giới hạn bộ nhớ...")
                            try:
                                cpu_decoded = _translate_raw_batch(
                                    model.to("cpu"), tokenizer, vi_lang_id, batch, "cpu", 1
                                )
                                sorted_results[i] = cpu_decoded[0]
                                if get_device() == "cuda":
                                    model.to("cuda")
                            except Exception as cpu_err:
                                print(f"[CẢNH BÁO] Không thể dịch câu: {cpu_err}. Giữ nguyên câu gốc.")
                                sorted_results[i] = batch[0]
                            i += 1
                    else:
                        print(f"[LỖI BATCH DỊCH] {e}. Giữ nguyên câu gốc cho batch này.")
                        for b_offset, orig_text in enumerate(batch):
                            sorted_results[i + b_offset] = orig_text
                        i += len(batch)

            if clear_cache and cur_device == "cuda":
                torch.cuda.empty_cache()

    # Bước 3: Ánh xạ kết quả dịch đã sắp xếp về flat_units ban đầu
    flat_results = [""] * len(flat_units)
    for sorted_idx, res in zip(sorted_unit_indices, sorted_results):
        orig_src = flat_units[sorted_idx]
        # Nếu mô hình trả về chuỗi rỗng thì giữ nguyên câu gốc
        cleaned_res = res.strip() if res else ""
        if not cleaned_res and orig_src.strip():
            print(f"[CẢNH BÁO] Dịch trả về rỗng cho: '{orig_src[:50]}...', giữ nguyên văn bản gốc.")
            cleaned_res = orig_src
        flat_results[sorted_idx] = cleaned_res

    # Bước 4: Ghép các chunk con về đúng texts ban đầu
    grouped_results: dict[int, list[str]] = {idx: [] for idx in range(len(texts))}
    for u_idx, orig_text_idx in enumerate(unit_map):
        grouped_results[orig_text_idx].append(flat_results[u_idx])

    final_results = []
    for idx in range(len(texts)):
        joined = " ".join(part for part in grouped_results[idx] if part).strip()
        if not joined and texts[idx].strip():
            joined = texts[idx].strip()
        final_results.append(joined)

    return final_results

