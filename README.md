# 📄 DocuVi — Hệ Thống Dịch Thuật Tài Liệu PDF Giữ Nguyên Định Dạng Bằng AI

> **Đồ án / Bài tập lớn cuối môn Lập trình Python**  
> **Ứng dụng Trí tuệ Nhân tạo (Generative AI) trong xử lý tài liệu số**

---

## 🎯 1. Giới thiệu đề tài

### 1.1. Tính cấp thiết
Hiện nay, nhu cầu đọc và nghiên cứu tài liệu tiếng Anh (bài báo khoa học, sách chuyên ngành, báo cáo kinh tế) của sinh viên và kỹ sư là rất lớn. Tuy nhiên, các công cụ dịch thuật phổ biến (Google Translate, v.v.) khi xử lý file PDF thường gặp các hạn chế nghiêm trọng:
- Làm **vỡ khung layout**, xô lệch các cột chữ.
- **Mất hình ảnh**, sơ đồ vector và cấu trúc bảng biểu.
- Bản dịch tràn viền hoặc bị đè lên các thành phần đồ họa khác.

### 1.2. Mục tiêu dự án
Xây dựng một hệ sinh thái hoàn chỉnh bằng **Python** cho phép:
1. Dịch tài liệu PDF từ **Tiếng Anh sang Tiếng Việt** chuyên sâu bằng mô hình học sâu **VinAI Seq2Seq** (`vinai/vinai-translate-en2vi`).
2. **Tối ưu hóa bảo toàn bố cục**: Cố gắng duy trì vị trí các thành phần và định dạng gốc (hình ảnh, sơ đồ vector, bảng biểu, cấu trúc khối chữ); kết quả phụ thuộc vào cấu trúc tài liệu PDF nguồn và độ dài tương đối của bản dịch.
3. **Cân chỉnh font chữ thích ứng (Dynamic Font Fitting)**: Tự động tính toán cỡ chữ, khoảng cách dòng và khung chèn bằng font Times New Roman chuẩn Unicode tiếng Việt để tối ưu hiển thị văn bản trong không gian bố cục cho phép.
4. Cung cấp cả **Giao diện Web tương tác hiện đại (Web UI)** và **Giao diện dòng lệnh (CLI)**.

---

## 🏛️ 2. Kiến trúc hệ thống & Pipeline xử lý

```
  [Tài liệu PDF gốc (EN)]
            │
            ▼
┌───────────────────────────────────────┐
│ Bước 1: Trích xuất Bố cục & Text Layer│  --> (PyMuPDF / layout_extractor.py)
│  - Phân tích khối chữ, bảng biểu, ảnh │
│  - Nhận diện font size, flags (bold)  │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Bước 2: Tiền xử lý & Gom nhóm Batch   │  --> (text_processor.py)
│  - Tách câu thông minh, lọc ký tự số  │
│  - Sắp xếp độ dài câu để giảm padding │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Bước 3: Dịch máy Neural AI (GPU CUDA) │  --> (VinAI BART Seq2Seq / translator.py)
│  - Chế độ Greedy (beams=1) siêu tốc   │
│  - Chế độ Beam Search (beams=4)       │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Bước 4: Tái tạo trang & Đồ họa PDF    │  --> (renderer.py / pdf_process.py)
│  - Xóa lớp chữ gốc bằng Redaction     │
│  - Đo kích thước & fit cỡ font TNR    │
│  - Bảo lưu ảnh raster & vector art   │
└───────────────────┬───────────────────┘
                    │
                    ▼
  [File PDF đã dịch (VI) + Trình xem Preview trực tiếp]
```

---

## 🧠 3. Các kỹ thuật & Kiến thức Python nâng cao áp dụng

Dự án vận dụng nhiều kỹ thuật lập trình Python nâng cao:

* **Tổ chức Package & Module chuyên nghiệp**: Tách biệt rõ ràng tầng xử lý logic cốt lõi (`pdf_engine/`) và tầng giao tiếp Web/CLI (`app.py`).
* **Lập trình Hướng đối tượng (OOP)**:
  - Lớp `ScratchFontTester` kế thừa và đóng gói cơ chế giả lập render trang ẩn để tính toán kích thước text trước khi ghi đè thật.
  - Lớp `_LazyFontBuffers` kế thừa `dict` để hỗ trợ cơ chế tải lười font đệm.
* **Xử lý đồng thời & Thread Safety**: Sử dụng `threading.Lock()` với mô hình **Double-Checked Locking** để quản lý nạp tài nguyên nặng (Model AI ~1.5GB, Font Buffers) chỉ một lần duy nhất khi có request đầu tiên.
* **Tối ưu hóa PyTorch & Phần cứng GPU**:
  - `torch.inference_mode()` tăng tốc suy luận và giảm thiểu tiêu tốn VRAM.
  - Tự động nhận diện phần cứng (NVIDIA CUDA GPU vs CPU) và sử dụng định dạng bán chính xác `torch.float16`.
* **RESTful Web API & Web Client**:
  - Backend Flask hỗ trợ upload multipart form, streaming file và phục vụ tài liệu PDF (`send_file`).
  - Giao diện Frontend tích hợp xem trước trực tiếp tài liệu song song (Side-by-Side Live PDF Preview).
* **CLI Terminal Support**: Cho phép chạy trực tiếp từ terminal để tích hợp vào các pipeline tự động hóa.

---

## 📁 4. Cấu trúc thư mục dự án

```text
project/
├── Font/                      # Bộ font Times New Roman hỗ trợ tiếng Việt có dấu
│   ├── times.ttf              # Regular
│   ├── timesbd.ttf            # Bold
│   ├── timesi.ttf             # Italic
│   └── timesbi.ttf            # Bold Italic
├── pdf_engine/                # Package xử lý PDF & Mô hình AI cốt lõi
│   ├── __init__.py            # Export API công khai của package
│   ├── config.py              # Cấu hình đường dẫn, nạp font đệm (lazy-load)
│   ├── layout_extractor.py    # Phân tích khối hộp, phân dòng & trích xuất bảng biểu
│   ├── text_processor.py      # Chuẩn hóa văn bản, tách câu thông minh
│   ├── translator.py          # Nạp mô hình VinAI, quản lý suy luận GPU Batch
│   └── renderer.py            # Tính toán vị trí chèn text, fit cỡ font và vẽ PDF
├── templates/
│   └── index.html             # Giao diện web SPA hiện đại (PDF Preview song song)
├── uploads/                   # Thư mục lưu trữ file PDF tải lên
├── outputs/                   # Thư mục lưu trữ bản dịch PDF đầu ra
├── app.py                     # Entry point chính: Flask Server & CLI Runner
├── pdf_process.py             # Pipeline kết nối toàn bộ quy trình dịch file PDF
├── requirements.txt           # Danh mục thư viện phụ thuộc
├── .gitignore                 # Bỏ qua file rác, file tạm và môi trường ảo
└── README.md                  # Tài liệu hướng dẫn & Báo cáo kỹ thuật
```

---

## 🚀 5. Hướng dẫn cài đặt & Khởi chạy

### 5.1. Yêu cầu môi trường
- Python `>= 3.10`
- Hệ điều hành: Windows, macOS hoặc Linux
- (Khuyến nghị) GPU NVIDIA hỗ trợ CUDA để tốc độ dịch nhanh nhất.

### 5.2. Cài đặt các thư viện phụ thuộc
Khởi tạo môi trường ảo và cài đặt thư viện:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (trên Windows)
.\.venv\Scripts\activate

# Cài đặt các gói cần thiết
pip install flask pymupdf torch transformers sentencepiece protobuf
```

### 5.3. Khởi chạy ứng dụng

#### Cách 1: Chạy Giao diện Web (Khuyến nghị)
```bash
python app.py
```
Sau đó mở trình duyệt và truy cập: **`http://127.0.0.1:5000`**

#### Cách 2: Chạy trực tiếp từ dòng lệnh (CLI)
```bash
# Dịch file có sẵn trong thư mục uploads/
python app.py uploads/test1.pdf

# Hoặc truyền đường dẫn tuyệt đối/tương đối bất kỳ
python app.py "C:/path/to/document.pdf"
```
Kết quả dịch sẽ tự động được lưu vào thư mục `outputs/translated_<tên_file>.pdf`.

---

## 📊 6. Kết quả thực nghiệm & Đo đạc hiệu năng

### 6.1. Khả năng bảo toàn bố cục
- Hệ thống **cố gắng duy trì tối đa vị trí các thành phần và định dạng gốc** (hình ảnh raster, đường nét vector, khung bảng biểu và thứ tự đọc của văn bản).
- Mức độ bảo toàn phụ thuộc trực tiếp vào cấu trúc tài liệu PDF nguồn (tài liệu sinh bởi phần mềm soạn thảo hay xuất từ công cụ khác, số lượng cột, khoảng đệm giữa các khối) và độ dài mở rộng của câu tiếng Việt sau khi dịch.

### 6.2. Số liệu đo đạc thực tế (Benchmark)
Các chỉ số dưới đây được đo đạc thực nghiệm trực tiếp từ quy trình chạy mã nguồn trên môi trường kiểm thử xác định:

- **Cấu hình phần cứng kiểm nghiệm**:
  - **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA, PyTorch FP16).
  - **CPU**: Intel Core thế hệ 12 (Intel64 Family 6 Model 151).
  - **RAM**: 12 GB.
  - **Hệ điều hành**: Windows 11/10 (64-bit).
- **Tập dữ liệu kiểm thử**: File `uploads/test1.pdf` (Tài liệu y khoa chuyên ngành ADHD, gồm **9 trang**, chứa **302 câu** tiếng Anh cần dịch).

| Chỉ số đo đạc | Chế độ Greedy (`num_beams=1`) | Chế độ Beam Search (`num_beams=4`) | Ghi chú kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Thời gian nạp mô hình (khởi tạo)** | ~8.14 giây | ~8.14 giây | Tải trọng số từ đĩa vào VRAM GPU (chỉ tải 1 lần nhờ cơ chế Lazy Loading) |
| **Tổng thời gian xử lý toàn bộ tài liệu (9 trang, 302 câu)** | **5.62 giây** | ~28.5 giây | Toàn bộ pipeline: phân tích layout, gom batch dịch GPU và tái tạo PDF |
| **Thời gian xử lý trung bình mỗi trang** | **~0.62 giây / trang** | ~3.16 giây / trang | Phụ thuộc vào mật độ câu chữ trên từng trang |
| **Thời gian suy luận trung bình mỗi câu** | **~0.019 giây / câu** | ~0.094 giây / câu | Suy luận song song trên GPU theo batch (batch_size = 32) |

> [!NOTE]
> **Lưu ý về hiệu năng**:
> - Tốc độ thực tế có thể thay đổi tùy thuộc vào cấu hình phần cứng (nếu chạy trên CPU thời gian suy luận sẽ kéo dài hơn so với GPU CUDA), số lượng câu và độ phức tạp của đồ họa trong file PDF.

---

## 🔮 7. Hướng phát triển trong tương lai
- Tích hợp thêm mô-đun **OCR (Tesseract / PaddleOCR)** để nhận diện và dịch cả các tài liệu PDF dạng ảnh quét (scanned documents).
- Hỗ trợ dịch ngược chiều **Việt $\rightarrow$ Anh** khi có mô hình phù hợp.
- Bổ sung tùy chọn xuất file Word (.docx) và song ngữ song song trên cùng trang.
