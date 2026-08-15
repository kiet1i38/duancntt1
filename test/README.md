# Test assets

Các file active cho phép chạy SD-TokenSkip:

- `run_full_sd_tokenskip.py`: full GSM8K runner có resume theo từng dòng.
- `tokenskip_pilot_core.py`: prompt, evaluator, model loader và metrics dùng chung.
- `TokenSkip-official/`: phần code/evaluator/dataset GSM8K cần thiết từ repository TokenSkip chính thức.
- `full-sd-tokenskip-gsm8k-512/`: trạng thái resume hiện tại.
- `pilot-sd-tokenskip-fresh-27-512/`: kết quả screening 27 bài, không phải reproduced result.

Nguồn TokenSkip chính thức tại snapshot local: `7b958cdb98e0b2a84056e26e885b260987f2e64f`.

Không đưa vào repository: `.venv`, Hugging Face model cache, browser profile, bytecode và các archive của phương pháp đã loại.
