# duancntt1 — SD-TokenSkip evaluation

Repository này lưu phần chạy thực nghiệm cần thiết cho đề tài Adaptive Compression of Chain-of-Thought.

Hướng dẫn đầy đủ dạng HTML: [HUONG-DAN-CHAY-SD-TOKENSKIP.html](./HUONG-DAN-CHAY-SD-TOKENSKIP.html)

Phương pháp đang chạy là **Selective Difficulty-Conditioned TokenSkip (SD-TokenSkip)** trên:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- TokenSkip adapter: `hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K`
- Dataset: GSM8K train/test theo dữ liệu chính thức đi kèm TokenSkip
- Policy đã khóa: `easy -> 0.5`, `medium -> 1.0`, `hard -> 1.0`
- So sánh: Full-CoT, fixed `0.5/0.7/0.9`, SD-TokenSkip

## Chạy tiếp trên máy mới

Clone repository rồi chạy từ thư mục gốc:

```powershell
git clone https://github.com/kiet1i38/duancntt1.git
cd duancntt1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài PyTorch bản CUDA phù hợp với GPU/driver trước. Môi trường máy cũ đã dùng `torch 2.5.1+cu121`; sau đó cài các thư viện còn lại:

```powershell
python -m pip install --upgrade pip
pip install -r test\requirements-sd-run.txt
```

Chạy tiếp phép đo full; runner tự đọc các dòng đã ghi trong `full-sd-tokenskip-gsm8k-512` và bỏ qua chúng:

```powershell
python test\run_full_sd_tokenskip.py `
  --base-model Qwen/Qwen2.5-3B-Instruct `
  --adapter-path hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K `
  --cal-data test\TokenSkip-official\datasets\gsm8k\train.jsonl `
  --eval-data test\TokenSkip-official\datasets\gsm8k\test.jsonl `
  --output-dir test\full-sd-tokenskip-gsm8k-512 `
  --cal-pool 0 --n-cal 9 --n-eval 1319 --max-new-tokens 512
```

Model và adapter sẽ được tải từ Hugging Face ở lần chạy đầu; không commit cache model vào Git.

## Trạng thái đã lưu

Snapshot hiện tại đã ghi hoàn chỉnh **8/9 dòng calibration**, chưa có evaluation row nào. Runner có resume theo `id`, nên không cần xóa thư mục output trước khi chạy tiếp.

`run_meta.json` giữ hash snapshot của máy cũ để truy vết. Khi chạy bằng model ID trên máy mới, runner sẽ ghi lại metadata với đường dẫn mới.

## Phân loại kết quả

Kết quả full split của SD-TokenSkip là đánh giá phương pháp nhóm đề xuất. Không gọi là `Reproduced result` nếu chưa khớp đầy đủ checkpoint, revision, precision, prompt, decoding, evaluator và protocol của bài báo TokenSkip.

## Tài nguyên máy khuyến nghị

- Tối thiểu thực dụng cho runner hiện tại: NVIDIA GPU 8 GB VRAM, RAM 16 GB, SSD trống 30 GB.
- Khuyến nghị: GPU 12–16 GB VRAM, RAM 32 GB, SSD NVMe trống ít nhất 50 GB.
- Nếu muốn chạy FP16 hoặc evaluation batch lớn gần protocol vLLM chính thức: GPU 16–24 GB VRAM.

RTX 3060 12 GB là mức tiết kiệm có thể dùng; RTX 4070 12 GB hoặc RTX 4060 Ti 16 GB phù hợp hơn; RTX 3090 24 GB là lựa chọn dư dả nếu ưu tiên chạy nhanh/batch lớn.
