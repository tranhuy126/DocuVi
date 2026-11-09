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
            torch.backends.cudnn.benchmark = True
    return _device


def get_translation_model():
    global _model, _tokenizer, _vi_lang_id, _device
    if _model is None or _tokenizer is None:
        with _model_init_lock:
            if _model is None or _tokenizer is None:
                current_device = get_device()
                print(f"[Khởi tạo AI] Đang nạp mô hình dịch {MODEL_NAME} trên {current_device.upper()}...")
                loaded_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="en_XX")

                if current_device == "cuda":
                    try:
                        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(
                            MODEL_NAME,
                            torch_dtype=torch.float16,
                        ).to(current_device)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as gpu_error:
                        print(
                            f"[CẢNH BÁO] Không đủ VRAM trên GPU ({gpu_error}), "
                            "tự động chuyển sang nạp trên CPU..."
                        )
                        current_device = _device = "cpu"
                        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")
                else:
                    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(current_device)

                loaded_model.eval()
                _tokenizer = loaded_tokenizer
                _model = loaded_model
                _vi_lang_id = loaded_tokenizer.lang_code_to_id["vi_VN"]
                print(f"[Khởi tạo AI] Nạp mô hình hoàn tất trên {current_device.upper()}!")

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
    stripped_text = text.strip()
    if len(stripped_text) <= max_chars:
        return [stripped_text]

    for split_pattern in (r"(?<=[;:—])\s+", r"(?<=,)\s+"):
        candidate_parts = re.split(split_pattern, stripped_text)
        if len(candidate_parts) > 1:
            segments = candidate_parts
            space_padding = 0
            break
    else:
        segments = stripped_text.split()
        space_padding = 1

    chunks = []
    current_chunk = ""
    for segment in segments:
        if current_chunk and len(current_chunk) + len(segment) + space_padding > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = segment
        else:
            current_chunk = f"{current_chunk} {segment}".strip() if current_chunk else segment

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks or [stripped_text]


def _translate_raw_batch(
    model,
    tokenizer,
    vi_lang_id: int,
    batch: list[str],
    current_device: str,
    num_beams: int,
) -> list[str]:
    encoded_inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(current_device)

    generation_kwargs = {
        "decoder_start_token_id": vi_lang_id,
        "num_beams": num_beams,
        "max_length": 512,
    }
    if num_beams > 1:
        generation_kwargs["early_stopping"] = True

    generated_tokens = model.generate(**encoded_inputs, **generation_kwargs)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate_batch(
    texts: list[str],
    batch_size: int = 32,
    num_beams: int = 1,
    clear_cache: bool = False,
    progress_callback=None,
) -> list[str]:
    if not texts:
        return []

    model, tokenizer, vi_lang_id = get_translation_model()
    current_device = get_device()

    flat_units = []
    unit_to_text_index = []
    for original_index, source_text in enumerate(texts):
        for chunk in _chunk_text_if_needed(source_text, max_chars=450):
            flat_units.append(chunk)
            unit_to_text_index.append(original_index)

    indexed_units = sorted(enumerate(flat_units), key=lambda item: len(item[1]))
    sorted_unit_indices = [item[0] for item in indexed_units]
    sorted_units = [item[1] for item in indexed_units]
    sorted_results = [""] * len(sorted_units)

    with model_lock, torch.inference_mode():
        batch_start_index = 0
        current_batch_size = max(1, batch_size)
        total_units = len(sorted_units)

        while batch_start_index < total_units:
            current_batch = sorted_units[batch_start_index : batch_start_index + current_batch_size]
            try:
                decoded_translations = _translate_raw_batch(
                    model,
                    tokenizer,
                    vi_lang_id,
                    current_batch,
                    current_device,
                    num_beams,
                )
                sorted_results[batch_start_index : batch_start_index + len(decoded_translations)] = decoded_translations
                batch_start_index += len(current_batch)

                if progress_callback:
                    progress_callback(min(batch_start_index, total_units), total_units)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as runtime_error:
                error_message_lower = str(runtime_error).lower()
                is_cuda_oom = "out of memory" in error_message_lower or "cuda" in error_message_lower

                if is_cuda_oom:
                    print(
                        f"[CẢNH BÁO] Tràn VRAM CUDA tại batch kích thước "
                        f"{current_batch_size}: {runtime_error}"
                    )
                    if current_device == "cuda":
                        torch.cuda.empty_cache()

                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"[TỰ PHỤC HỒI] Giảm batch_size xuống {current_batch_size} và tiếp tục...")
                        continue

                    print("[TỰ PHỤC HỒI] Dịch câu đơn trên CPU để vượt qua giới hạn bộ nhớ...")
                    try:
                        cpu_translation = _translate_raw_batch(
                            model.to("cpu"),
                            tokenizer,
                            vi_lang_id,
                            current_batch,
                            "cpu",
                            1,
                        )[0]
                        sorted_results[batch_start_index] = cpu_translation
                        if get_device() == "cuda":
                            model.to("cuda")
                    except Exception as cpu_error:
                        print(f"[CẢNH BÁO] Không thể dịch câu: {cpu_error}. Giữ nguyên câu gốc.")
                        sorted_results[batch_start_index] = current_batch[0]
                    batch_start_index += 1
                else:
                    print(f"[LỖI BATCH DỊCH] {runtime_error}. Giữ nguyên câu gốc cho batch này.")
                    sorted_results[batch_start_index : batch_start_index + len(current_batch)] = current_batch
                    batch_start_index += len(current_batch)

                if progress_callback:
                    progress_callback(min(batch_start_index, total_units), total_units)

        if clear_cache and current_device == "cuda":
            torch.cuda.empty_cache()

    flat_results = [""] * len(flat_units)
    for original_unit_index, raw_translation in zip(sorted_unit_indices, sorted_results):
        original_unit_text = flat_units[original_unit_index]
        cleaned_translation = (raw_translation or "").strip()
        if not cleaned_translation and original_unit_text.strip():
            print(
                f"[CẢNH BÁO] Dịch trả về rỗng cho: '{original_unit_text[:50]}...', "
                "giữ nguyên văn bản gốc."
            )
            cleaned_translation = original_unit_text
        flat_results[original_unit_index] = cleaned_translation

    grouped_results: dict[int, list[str]] = {index: [] for index in range(len(texts))}
    for unit_index, original_text_index in enumerate(unit_to_text_index):
        grouped_results[original_text_index].append(flat_results[unit_index])

    final_translations = []
    for index, original_text in enumerate(texts):
        joined_translation = " ".join(part for part in grouped_results[index] if part).strip()
        final_translations.append(joined_translation or original_text.strip())

    return final_translations

