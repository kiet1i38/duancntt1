# SD-TokenSkip fresh pilot

Đây là pilot inference-only, chưa phải `Reproduced result`.

- Base: `Qwen/Qwen2.5-3B-Instruct` snapshot đã khóa trong lệnh chạy.
- Adapter: `TokenSkip-Qwen2.5-3B-Instruct-GSM8K` snapshot chính thức đã khóa.
- Dataset/protocol: GSM8K chính thức, prompt/evaluator của TokenSkip, greedy decoding, seed 42, `max_new_tokens=512`, 4-bit inference vì GPU 4GB.
- Calibration: 9 mẫu lấy theo input-complexity từ train, không dùng test answer.
- Evaluation: 27 test IDs độc lập, 9 easy / 9 medium / 9 hard; danh sách nằm trong `pilot_rows.jsonl`.
- Policy đã khóa trước evaluation: `easy→ratio 0.5`, `medium→Full-CoT`, `hard→Full-CoT`.

Kết quả test: Full-CoT `26/27`, `297.67` reasoning tokens/câu, `356.681s`; SD-TokenSkip `26/27`, `253.56` tokens/câu, `362.087s`; giảm `14.82%` token, accuracy delta `0`, latency tăng `1.52%`. Theo bin: easy `9/9`, medium `9/9`, hard `8/9`. Fixed ratio `0.9` đạt `26/27`, `216.59` tokens/câu nhưng latency `518.651s`.

Không được dùng các số này để gọi là official repository result hoặc reproduced result; cần chạy protocol đầy đủ với revision/hash và toàn bộ split sau khi thầy duyệt.
