# Full SD-TokenSkip resume state

Trạng thái snapshot trước khi đổi máy:

- Calibration đã ghi: 8/9.
- Evaluation đã ghi: 0/1319.
- Dòng đang chờ: calibration còn lại 1 dòng; sau đó chạy evaluation 1.319 dòng.
- `calibration_rows.jsonl` là JSONL append-only và được runner đọc để resume theo `id`.

Không xóa thư mục này trước khi chạy tiếp. Nếu dùng model ID thay cho đường dẫn cache cũ, runner sẽ cập nhật `run_meta.json`.
