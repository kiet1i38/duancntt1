# Dự án CNTT 1 - Kế hoạch nghiên cứu chủ đề 3

Kế hoạch hiện tại dành cho chủ đề **Adaptive Compression of Chain-of-Thought for Efficient Reasoning**. Bối cảnh và các quyết định đã chốt được ghi trong `flag.md`; tài liệu nghiên cứu chi tiết nằm ở `nghien-cuu-de-tai.md`.

## Mục tiêu nghiên cứu

1. Đọc và hệ thống hóa các công trình về Long-CoT, Short-CoT, CoT distillation và adaptive compression.
2. Chọn một bài báo nền có code/dataset/model đủ rõ để tái lập.
3. Tái lập baseline trên đúng model và dataset của bài báo nền.
4. Đề xuất một cải tiến có giả thuyết, thuật toán, calibration, ablation và tiêu chí đánh giá rõ ràng.
5. So sánh công bằng, phân tích lỗi và viết bài báo khoa học.

## Mốc hiện tại: đã có run FP16, chờ thầy duyệt cách báo cáo

1. Trình bày nguyên văn câu hỏi nghiên cứu của thầy: giảm reasoning tokens nhưng giữ accuracy; compression ratio thay đổi theo difficulty.
2. Xin thầy duyệt **TokenSkip** làm bài báo nền duy nhất và cấu hình `Qwen/Qwen2.5-3B-Instruct` + adapter `TokenSkip-Qwen2.5-3B-Instruct-GSM8K` trên `GSM8K` làm reproduction chính; PDF Appendix B.2/Table 2 xác nhận đây là dòng model được bài báo báo cáo.
3. Xin thầy duyệt ứng viên **Selective Difficulty-Conditioned TokenSkip (SD-TokenSkip)**: difficulty từ input chia easy/medium/hard; policy khóa trước test `easy→0.5`, `medium/hard→Full-CoT`, sau đó sinh một lần bằng adapter 3B; không dùng verifier/continuation/teacher trong phương pháp chính.
4. Hỏi rõ accuracy tolerance `δ`, GPU/thời gian, có bắt buộc MATH-500 hay không và có cần extension hay không; mặc định hiện tại không chạy extension.
5. Nêu rõ adaptive/difficulty-aware đã có trong MACC, TALE, TokenSqueeze, DAD/LiteCoT, TRAAC và DiffAdapt; nhóm chỉ bảo vệ claim hẹp về calibration difficulty–budget trên cùng model/dataset, fair protocol và overhead-aware evaluation.

### Tài liệu đã chuẩn bị ngày 2026-08-14

- Đã tạo [bai-bao-chu-de-3.html](./bai-bao-chu-de-3.html) với 50 công trình, link trực tiếp và nhãn Published/Preprint/Workshop.
- Đã tạo [phan-tich-chi-tiet-50-bai-chu-de-3.html](./phan-tich-chi-tiet-50-bai-chu-de-3.html) với ma trận 40 trường/bài, bảng tổng hợp, tìm kiếm/lọc, khoảng trống và phương pháp.
- Đã cập nhật `nghien-cuu-de-tai.md` theo kết luận đủ 50 bài và SD-TokenSkip; các pilot inference-only đã có nhưng chưa có reproduced result.
- Đã sao lưu mô tả cũ tại `mo-ta-chi-tiet-de-tai-truoc-khi-chot-50-bai.md` trước khi viết lại.
- Việc còn chờ thầy: xác nhận chọn row 3B và adapter/revision cụ thể, dataset/protocol, `δ`, difficulty bins và việc có cần extension hay không.
- Cập nhật sau pilot SD-TokenSkip: DC-TokenSkip và TS-DC-TokenSkip đã rời active, lưu tại `test/archive-dc-tokenskip-rejected/` và `test/archive-ts-dc-tokenskip-rejected/`. Pilot SD mới tại `test/pilot-sd-tokenskip-fresh-27-512/` giữ `26/27` như Full-CoT và giảm `14.8%` reasoning tokens; đây vẫn chỉ là screening evidence, chưa phải reproduced result.

### Kết quả run đã xác minh ngày 2026-08-16

- Đã chạy đúng base `Qwen/Qwen2.5-3B-Instruct`, adapter `hemingkx/TokenSkip-Qwen2.5-3B-Instruct-GSM8K`, GSM8K chính thức, prompt/evaluator TokenSkip, `max_new_tokens=512`, FP16, batch `288`; không dùng 4-bit. Mỗi run có `9` calibration và `1.319` evaluation rows với ID duy nhất.
- Fresh original-ratio baseline: ratio `0.5` trên toàn bộ 1.319 câu đạt accuracy `74.8294%`, `169.874905` reasoning tokens/câu, latency `1216.081185s`, `227.016211` tokens/correct. Đây là fresh inference của protocol ratio gốc, không phải số paper reported.
- Full-CoT baseline tham chiếu trong cùng môi trường đạt accuracy `83.4723%`, `316.920394` reasoning tokens/câu, latency `1113.195018s`, `379.671208` tokens/correct.
- Extension gate: `difficulty_score <= 46` dùng ratio `0.5` cho `385` câu, còn `934` câu chạy Full-CoT; đạt accuracy `81.5770%`, `280.696740` reasoning tokens/câu, latency `1095.655628s`, `344.088290` tokens/correct. So với Full-CoT: giảm `11.42989%` reasoning tokens, giảm `1.8953` điểm accuracy và latency giảm `1.57559%`.
- Extension gate là `Extension result; fresh inference`, không gọi là phương pháp gốc hay paper reported result. Original-ratio baseline giảm nhiều token hơn nhưng mất `8.6429` điểm accuracy, nên cần báo cáo riêng hai kết quả này.
- Artifact đã lưu tại `test/full-sd-tokenskip-gsm8k-512-fp16/` và `test/full-sd-tokenskip-gsm8k-512-original-ratio-gate-fp16/`; commit GitHub mới là `17b7eeadc8eab765a20a3f7b50f3668ab3b0d954`. Thư mục 4-bit `test/full-sd-tokenskip-gsm8k-512/` được giữ nguyên.

### Kết quả run 3B multi-level và 7B Full-CoT đã xác minh ngày 2026-08-17

- Đã chạy 3B multi-level SD-TokenSkip bằng FP16, batch `288`, cùng model `Qwen/Qwen2.5-3B-Instruct`, adapter TokenSkip 3B, GSM8K chính thức, `9` calibration và `1.319` evaluation rows. Artifact local tại `test/full-sd-tokenskip-gsm8k-512-multilevel-fp16/`; có `1.319/1.319` ID duy nhất và đủ `calibration_rows.jsonl`, `evaluation_rows.jsonl`, `full_results.json`, `run_meta.json`, `run.stdout.log`.
- Calibration với các ratio `0.5/0.7/0.8/1.0` đã chọn policy thực tế `easy→1.0`, `medium→0.5`, `hard→1.0`; policy này khác policy cố định ban đầu vì bin easy không đạt tolerance trên calibration. Multi-level đạt accuracy `80.2881%`, `265.347991` reasoning tokens/câu, latency `1080.570016s`, `330.494806` tokens/correct. So với Full-CoT 3B tham chiếu: giảm `16.272983%` token, mất `3.1842` điểm accuracy, latency giảm `2.930754%`.
- Chạy 7B Full-CoT bằng FP16, batch `32` vì giới hạn VRAM 24GB, cùng adapter TokenSkip 7B nhưng adapter được disable ở ratio `1.0`; model `Qwen/Qwen2.5-7B-Instruct`, cùng GSM8K, `9` calibration và `1.319` evaluation rows. Artifact local tại `test/full-cot-qwen2.5-7b-gsm8k-512-fp16/`; có `1.319/1.319` ID duy nhất, không trùng và đủ 5 file. Kết quả fresh baseline: accuracy `91.5845%`, `298.978014` reasoning tokens/câu, latency `2114.559231s`, `326.450331` tokens/correct.
- 7B Full-CoT chỉ là baseline cùng dataset/model scale để đối chiếu mô tả; chưa chạy SD-TokenSkip 7B nên không được dùng để kết luận compression của 7B tốt hơn. Cả số 3B và 7B ở trên đều là fresh inference, không phải paper reported result.
- Đã tạo `test/comparison_3b_7b.json` và script kiểm tra `test/summarize_sd_runs.py`. Chẩn đoán theo ví dụ `20%` token và tối đa `2` điểm accuracy loss: multi-level 3B **chưa đạt** (`16.272983%`, `3.1842` điểm). Gate trước đó đạt mất `1.8953` điểm nhưng chỉ giảm `11.42989%` token; đây là trade-off tốt hơn về accuracy nhưng vẫn chưa đạt mức giảm token 20%.
- Kết luận dừng của đợt này: SD-TokenSkip hiện chưa chứng minh được là phương pháp tốt hơn Full-CoT theo cả accuracy và token. Không chỉnh test set, không đổi model/dataset/precision để làm đẹp kết quả; cần báo cáo negative result và xin thầy duyệt hướng tiếp theo.

### Lựa chọn 2: protocol bốn run cho mỗi model ngày 2026-08-17

- Để so sánh gọn và đúng phạm vi đã chọn, mỗi model `Qwen2.5-3B` và `Qwen2.5-7B` chạy đúng **4 điều kiện độc lập**: (1) Full-CoT `γ=1.0`, (2) TokenSkip `γ=0.5`, (3) TokenSkip `γ=0.7`, (4) SD-TokenSkip.
- Full-CoT là mốc baseline phần cứng; γ=`0.5` là mức nén sâu; γ=`0.7` là mức nén trung gian đại diện; SD là phương pháp đề xuất. Đây là **selected comparison protocol**, không gọi là full sweep mọi ratio trong Table 2.
- **SD-TokenSkip** sau calibration chỉ chọn một ratio cho mỗi câu test và sinh một lần trên evaluation. Calibration có thể thử nhiều ratio trên tập nhỏ; chi phí phải ghi riêng như policy/router overhead. Chạy mọi ratio trên cùng câu test chỉ dùng cho fixed-ratio baseline/ablation, không phải main SD one-pass.
- Run 7B Full-CoT đã có chỉ là một trong bốn điều kiện. Việc bắt buộc tiếp theo là chạy 7B γ=`0.5`, 7B γ=`0.7` và 7B SD; 3B cũng phải có đủ bốn điều kiện để so sánh cùng cấu trúc.
- Các kết quả 3B/7B hiện có không bị xóa; run mới phải dùng thư mục riêng và không trộn với thư mục 4-bit.

## Thứ tự thực hiện bắt buộc

1. Xin thầy duyệt kết luận và scope trước khi chạy bất kỳ model nào.
2. Khóa model/tokenizer revision, adapter hash, dataset version/split/hash, prompt, preprocessing, seed, decoding, max tokens, answer extraction, evaluator và môi trường.
3. Chạy official repository result rồi bốn điều kiện đã chọn `Full-CoT/γ0.5/γ0.7/SD` trên cùng split cho từng model; tách `Reported`, `Official`, `Reproduced` và `Extension`.
4. Tạo difficulty label/calibrator chỉ từ train/validation; xây frontier accuracy–tokens–latency và khóa ratio policy trước test, không dùng test answer.
5. Đánh giá SD-TokenSkip: difficulty-first, policy khóa `easy→0.5`, `medium/hard→Full-CoT` và sinh một lần; cộng chi phí router/điều tra calibration khi báo cáo end-to-end.
6. Chạy ablation fixed, difficulty-only, random router, oracle, no-fallback, estimator/calibration và lỗi theo bin.
7. Đánh giá accuracy/EM, reasoning tokens, actual ratio, latency, tokens-per-correct, fallback rate, router overhead và Pareto theo từng bin.
8. Chỉ sau reproduction mới xét extension model/dataset khác nếu còn tài nguyên và được thầy duyệt; nếu không, không mở rộng.
9. Viết bài báo với related work, method, protocol audit, results, negative result, error analysis, limitation, reproducibility và ethics.

## Cấu hình ưu tiên tạm thời

- Bài báo nền: TokenSkip, EMNLP 2025.
- Model: `Qwen/Qwen2.5-3B-Instruct` + adapter `TokenSkip-Qwen2.5-3B-Instruct-GSM8K` (được repository chính thức liệt kê và phù hợp giới hạn phần cứng hiện tại).
- Dataset chính: GSM8K, giữ đúng split và format của baseline.
- Dataset mở rộng: MATH-500 chỉ chạy sau khi baseline chính tái lập ổn định.
- Baseline tối thiểu theo Lựa chọn 2: Full-CoT, TokenSkip fixed ratio `0.5` và `0.7` trên cả 3B/7B, SD-TokenSkip, random router và oracle control.
- Không dùng Qwen2.5-14B làm cấu hình chính vì vượt giới hạn 13B, dù TokenSkip có báo cáo kết quả trên model này.

## Tiêu chí dừng hoặc đổi hướng

- Nếu không tái lập được TokenSkip trong thời gian/tài nguyên đã thống nhất, báo thầy và chuyển sang cấu hình nhỏ hơn hoặc bài báo nền có mã nguồn rõ hơn; không tự thay model/dataset để tạo so sánh không công bằng.
- Nếu SD-TokenSkip không giữ accuracy trong `δ` hoặc không tạo Pareto trade-off so với fixed-ratio baseline, vẫn báo cáo kết quả âm cùng phân tích difficulty error và overhead; không điều chỉnh test set để có kết quả đẹp.
- Nếu literature cho thấy ý tưởng router difficulty/uncertainty đã trùng phương pháp, phải đổi claim hoặc đổi phương pháp trước khi viết bài báo.
- Chưa mở rộng sang chatbot, hệ thống quản lý dự án, sandbox chạy code, tự commit Git, tự deploy hoặc microservice sản phẩm. Các nội dung đó thuộc hướng DevFlow cũ và đang được lưu lịch sử.

## Phân chia việc cho hai thành viên

- Thành viên 1: TokenSkip paper/repository, model/dataset/evaluator revision, reported/official/reproduced bookkeeping.
- Thành viên 2: difficulty definition, SD-TokenSkip policy, metrics, ablation, error analysis và literature matrix.
- Cùng làm: xin thầy duyệt, kiểm tra leakage/license, review config/log và viết bài báo.

## Trạng thái tài liệu

- `flag.md`: đã cập nhật hướng nghiên cứu mới ngày 2026-08-14.
- `nghien-cuu-de-tai.md`: đã cập nhật kết luận đủ 50 bài, gap, các phương án, SD-TokenSkip, baseline/model/dataset/protocol và câu hỏi thầy.
- `phan-tich-chi-tiet-50-bai-chu-de-3.html`: đã tạo; đã kiểm tra JS/DOM/search/filter/card bằng browser.
- `mo-ta-chi-tiet-de-tai-truoc-khi-chot-50-bai.md`: bản sao lưu trước khi chốt.
- `nghien-cuu-de-tai-agent-cu.md`: bản nghiên cứu DevFlow cũ, lưu để tham chiếu lịch sử.
- `mo-ta-chi-tiet-de-tai.md`: đã viết lại thành mô tả đề tài nghiên cứu chủ đề 3; bản cũ ở `mo-ta-chi-tiet-de-tai-agent-cu.md`.
- `test/full-sd-tokenskip-gsm8k-512-fp16/`: fresh FP16 original-ratio baseline, đủ 1.319 evaluation rows.
- `test/full-sd-tokenskip-gsm8k-512-original-ratio-gate-fp16/`: fresh FP16 extension gate, đủ 1.319 evaluation rows.

## Quy tắc cập nhật

- `flag.md` là nguồn trạng thái; cập nhật trước khi thay đổi phạm vi hoặc quyết định nghiên cứu.
- `plan.md` chỉ thay đổi khi roadmap, mốc hoặc tiêu chí thực nghiệm thay đổi.
- Mọi kết quả phải ghi rõ nguồn: paper, official repository hay thực nghiệm của nhóm.
- Không ghi kết quả là đã tái lập nếu chưa có log/config/seed và kiểm tra độc lập.
