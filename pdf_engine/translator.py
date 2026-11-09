import re
import sys
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_NAME = "vinai/vinai-translate-en2vi"
_device = _tokenizer = _model = _vi_lang_id = None
_model_init_lock = threading.Lock()
model_lock = threading.Lock()


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        if _device == "cuda":
            torch.backends.cudnn.benchmark = True
    return _device


def get_translation_model():
    global _model, _tokenizer, _vi_lang_id, _device
    if _model is None or _tokenizer is None:
        with _model_init_lock:
            if _model is None or _tokenizer is None:
                cur_device = get_device()
                print(f"[Khởi tạo AI] Đang nạp mô hình dịch {MODEL_NAME} trên {cur_device.upper()}...")
                tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="en_XX")
                if cur_device == "cuda":
                    try:
                        mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(cur_device)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        print(f"[CẢNH BÁO] Không đủ VRAM trên GPU ({e}), tự động chuyển sang nạp trên CPU...")
                        cur_device = _device = "cpu"
                        mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")
                else:
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(cur_device)
                mdl.eval()
                _tokenizer, _model, _vi_lang_id = tok, mdl, tok.lang_code_to_id["vi_VN"]
                print(f"[Khởi tạo AI] Nạp mô hình hoàn tất trên {cur_device.upper()}!")
    return _model, _tokenizer, _vi_lang_id


def __getattr__(name: str):
    if name == "model":
        return get_translation_model()[0]
    if name == "tokenizer":
        return get_translation_model()[1]
    if name == "device":
        return get_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _chunk_text_if_needed(text: str, max_chars: int = 500) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    for pattern in (r"(?<=[;:—])\s+", r"(?<=,)\s+"):
        candidates = re.split(pattern, text)
        if len(candidates) > 1:
            parts, pad = candidates, 0
            break
    else:
        parts, pad = text.split(), 1

    chunks, current = [], ""
    for item in parts:
        if current and len(current) + len(item) + pad > max_chars:
            chunks.append(current.strip())
            current = item
        else:
            current = f"{current} {item}".strip() if current else item
    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _translate_raw_batch(model, tokenizer, vi_lang_id, batch: list[str], cur_device: str, num_beams: int) -> list[str]:
    encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(cur_device)
    gen_kwargs = {"decoder_start_token_id": vi_lang_id, "num_beams": num_beams, "max_length": 512}
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
    return tokenizer.batch_decode(model.generate(**encoded, **gen_kwargs), skip_special_tokens=True)


def translate_batch(
    texts: list[str], batch_size: int = 32, num_beams: int = 1, clear_cache: bool = False, progress_callback=None
) -> list[str]:
    if not texts:
        return []
    model, tokenizer, vi_lang_id = get_translation_model()
    cur_device = get_device()

    flat_units, unit_map = [], []
    for orig_idx, t in enumerate(texts):
        for c in _chunk_text_if_needed(t, max_chars=450):
            flat_units.append(c)
            unit_map.append(orig_idx)

    indexed_units = sorted(enumerate(flat_units), key=lambda x: len(x[1]))
    sorted_unit_indices = [x[0] for x in indexed_units]
    sorted_units = [x[1] for x in indexed_units]
    sorted_results = [""] * len(sorted_units)

    with model_lock, torch.inference_mode():
        i, current_batch_size = 0, max(1, batch_size)
        while i < len(sorted_units):
            batch = sorted_units[i : i + current_batch_size]
            try:
                decoded = _translate_raw_batch(model, tokenizer, vi_lang_id, batch, cur_device, num_beams)
                sorted_results[i : i + len(decoded)] = decoded
                i += len(batch)
                if progress_callback:
                    progress_callback(min(i, len(sorted_units)), len(sorted_units))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg:
                    print(f"[CẢNH BÁO] Tràn VRAM CUDA tại batch kích thước {current_batch_size}: {e}")
                    if cur_device == "cuda":
                        torch.cuda.empty_cache()
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"[TỰ PHỤC HỒI] Giảm batch_size xuống {current_batch_size} và tiếp tục...")
                        continue
                    print("[TỰ PHỤC HỒI] Dịch câu đơn trên CPU để vượt qua giới hạn bộ nhớ...")
                    try:
                        sorted_results[i] = _translate_raw_batch(model.to("cpu"), tokenizer, vi_lang_id, batch, "cpu", 1)[0]
                        if get_device() == "cuda":
                            model.to("cuda")
                    except Exception as cpu_err:
                        print(f"[CẢNH BÁO] Không thể dịch câu: {cpu_err}. Giữ nguyên câu gốc.")
                        sorted_results[i] = batch[0]
                    i += 1
                else:
                    print(f"[LỖI BATCH DỊCH] {e}. Giữ nguyên câu gốc cho batch này.")
                    sorted_results[i : i + len(batch)] = batch
                    i += len(batch)
                if progress_callback:
                    progress_callback(min(i, len(sorted_units)), len(sorted_units))

        if clear_cache and cur_device == "cuda":
            torch.cuda.empty_cache()

    flat_results = [""] * len(flat_units)
    for sorted_idx, res in zip(sorted_unit_indices, sorted_results):
        orig_src = flat_units[sorted_idx]
        cleaned_res = (res or "").strip()
        if not cleaned_res and orig_src.strip():
            print(f"[CẢNH BÁO] Dịch trả về rỗng cho: '{orig_src[:50]}...', giữ nguyên văn bản gốc.")
            cleaned_res = orig_src
        flat_results[sorted_idx] = cleaned_res

    grouped_results: dict[int, list[str]] = {idx: [] for idx in range(len(texts))}
    for u_idx, orig_text_idx in enumerate(unit_map):
        grouped_results[orig_text_idx].append(flat_results[u_idx])

    return [
        " ".join(p for p in grouped_results[idx] if p).strip() or texts[idx].strip()
        for idx in range(len(texts))
    ]
