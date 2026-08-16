# FLAG - Bối cảnh hiện tại của Dự án CNTT 1

**Cập nhật lần cuối:** 2026-08-17  
**Trạng thái:** Đã hoàn thành đợt nghiên cứu tài liệu 50 bài, các fresh FP16 baseline/extension trên GSM8K và run 7B Full-CoT đối chiếu; đang chờ thầy duyệt cách gọi và cách trình bày kết quả âm/trade-off. Hướng DevFlow Assistant bên dưới chỉ là lịch sử, không còn là roadmap hiện tại.

## Quyết định phạm vi mới 2026-08-14

- Nhóm chọn chủ đề số 3: **Adaptive Compression of Chain-of-Thought for Efficient Reasoning**, trọng tâm **Long-CoT → Short-CoT**.
- Câu hỏi nghiên cứu trung tâm phải giữ nguyên: **“Có thể giảm reasoning tokens mà không làm giảm accuracy, và mức compression có thể thay đổi theo độ khó bài toán không?”** Difficulty là biến trung tâm; uncertainty/entropy/confidence/verification chỉ là tín hiệu phụ để ước lượng rủi ro.
- Quy tắc bắt buộc của thầy: phương pháp nhóm phải dùng **cùng model và cùng dataset với bài báo nền** để so sánh công bằng; phải có thực nghiệm, chỉ số đánh giá và bài báo khoa học.
- Bài báo nền duy nhất được chọn để xin thầy duyệt: **TokenSkip: Controllable Chain-of-Thought Compression in LLMs**, EMNLP 2025. PDF của chính bài báo có báo cáo riêng dòng **Qwen2.5-3B trên GSM8K** ở Appendix B.2/Table 2; reproduction chính dùng đúng `Qwen/Qwen2.5-3B-Instruct` + adapter `TokenSkip-Qwen2.5-3B-Instruct-GSM8K` trên `GSM8K`. Các số 7B/14B chỉ là các dòng reported tham chiếu khác, không thay cho dòng 3B của nhóm.
- DAPB-VC, DC-Token và DC-TokenSkip/TS-DC-TokenSkip đã bị loại khỏi vùng active, lần lượt được lưu tại `test/archive-dapb-vc-rejected/`, `test/archive-dc-token-rejected/`, `test/archive-dc-tokenskip-rejected/` và `test/archive-ts-dc-tokenskip-rejected/`. Phương pháp active hiện tại là **Selective Difficulty-Conditioned TokenSkip (SD-TokenSkip)**: difficulty từ input chia easy/medium/hard; policy khóa trước test `easy→ratio 0.5`, `medium→Full-CoT`, `hard→Full-CoT`. Đây là policy bảo toàn rủi ro/empirical extension, không claim novelty tuyệt đối.
- Pilot SD-TokenSkip inference-only dùng đúng base `Qwen/Qwen2.5-3B-Instruct`, adapter `TokenSkip-Qwen2.5-3B-Instruct-GSM8K`, prompt/evaluator TokenSkip và GSM8K chính thức trên 27 test IDs mới, 9 calibration IDs, trần 512. Full-CoT đạt `26/27`, `297.67` reasoning tokens/câu; SD đạt `26/27`, `253.56` tokens/câu, giảm `14.8%`, latency `356.681s→362.087s`. Fixed `0.9` cũng đạt `26/27` và chỉ `216.59` tokens/câu nhưng latency `518.651s`; vì vậy SD có Pareto trade-off accuracy/token/latency trong pilot, chưa phải reproduced result.
- Model extension: **không bắt buộc trong phạm vi hiện tại** vì giới hạn phần cứng; chỉ xem xét model khác sau khi reproduction 3B hoàn tất và được thầy duyệt riêng.
- Đã kiểm tra đủ 50 bản ghi: 45 published, 4 preprint, 1 workshop/preprint. #35 và #38 là hai phiên bản của một canonical work và được đánh dấu để không đếm đôi trong gap.
- Đã chạy các pilot inference-only trong `test`; chưa fine-tune và chưa gọi các số fresh inference là paper reported result. Đã có original-ratio baseline và extension gate FP16 trên đủ 1.319 test rows; việc này không thay thế một lần chạy nguyên repository chính thức nếu thầy yêu cầu đúng entrypoint của paper.
- Đã tạo `phan-tich-chi-tiet-50-bai-chu-de-3.html` với ma trận 40 trường/bài, tìm kiếm, bộ lọc, card, bảng tổng hợp, khoảng trống, phương án, baseline/model/protocol và câu hỏi cần xin thầy.
- `nghien-cuu-de-tai.md` là kết luận nghiên cứu hiện tại. `bai-bao-chu-de-3.html` là danh mục nguồn gốc không ghi kết quả nhóm. `mo-ta-chi-tiet-de-tai.md` được cập nhật theo SD-TokenSkip sau khi sao lưu trạng thái DC-TokenSkip tại `mo-ta-chi-tiet-de-tai-truoc-khi-chot-SD-TokenSkip.md`.

## Trạng thái thực nghiệm đã xác minh 2026-08-16

- Protocol giữ nguyên model/adapter/dataset/evaluator TokenSkip, dùng FP16, batch `288`, `max_new_tokens=512`, `9` calibration rows và `1.319` evaluation rows; không dùng nén 4-bit.
- Original TokenSkip fixed ratio `0.5` trên toàn bộ test: accuracy `74.8294%`, `169.874905` reasoning tokens/câu, latency `1216.081185s`, `227.016211` tokens/correct. Đây là fresh inference, không phải paper reported result.
- Full-CoT baseline tham chiếu: accuracy `83.4723%`, `316.920394` reasoning tokens/câu, latency `1113.195018s`, `379.671208` tokens/correct.
- Extension gate dùng ratio `0.5` cho `385` câu có difficulty score `<=46`, Full-CoT cho `934` câu: accuracy `81.5770%`, `280.696740` reasoning tokens/câu, latency `1095.655628s`, `344.088290` tokens/correct; so với Full-CoT giảm `11.42989%` token, mất `1.8953` điểm accuracy và latency giảm `1.57559%`.
- Kết luận hiện tại: ratio gốc tiết kiệm token mạnh nhưng accuracy giảm `8.6429` điểm; gate là extension có trade-off cân bằng hơn nhưng chưa đạt mục tiêu 20% token/2% accuracy. Không gọi gate là phương pháp gốc.
- File kết quả: `test/full-sd-tokenskip-gsm8k-512-fp16/` và `test/full-sd-tokenskip-gsm8k-512-original-ratio-gate-fp16/`. GitHub đã push commit `17b7eeadc8eab765a20a3f7b50f3668ab3b0d954`; thư mục 4-bit cũ vẫn được giữ.

## Trạng thái thực nghiệm bổ sung 2026-08-17

- Đã chạy multi-level SD-TokenSkip 3B bằng FP16, batch `288`, cùng model/adapter/dataset/evaluator, `9` calibration và `1.319` evaluation. Policy được chọn chỉ từ calibration là `easy→1.0`, `medium→0.5`, `hard→1.0`; kết quả `80.2881%` accuracy, `265.347991` reasoning tokens/câu, giảm `16.272983%` token nhưng mất `3.1842` điểm accuracy so với Full-CoT 3B (`83.4723%`). Artifact tại `test/full-sd-tokenskip-gsm8k-512-multilevel-fp16/`.
- Đã chạy 7B Full-CoT bằng FP16, batch `32`, cùng GSM8K và adapter 7B được disable ở Full-CoT. Kết quả fresh baseline trên đủ `1.319` rows: `91.5845%` accuracy, `298.978014` reasoning tokens/câu, `2114.559231s` latency, `326.450331` tokens/correct. Artifact tại `test/full-cot-qwen2.5-7b-gsm8k-512-fp16/`.
- Cả hai run đều được kiểm tra `1.319` rows, `1.319` unique IDs, `0` duplicate và đủ 5 artifact. Đây là fresh inference; không gọi là paper reported result hoặc official reproduction. 7B hiện chỉ có Full-CoT baseline, chưa có 7B SD nên không kết luận compression 7B.
- So sánh chẩn đoán với ví dụ của thầy (`20%` token, tối đa `2` điểm accuracy loss): multi-level 3B chưa đạt. Gate trước đó có loss `1.8953` điểm nhưng chỉ giảm `11.42989%` token. Kết quả hiện tại là negative/trade-off evidence, không được làm đẹp bằng cách đổi model, dataset, precision hoặc test set.
- Summary lưu tại `test/comparison_3b_7b.json`; checker là `test/summarize_sd_runs.py`.

## Cách dùng file này khi mở chat mới

Đọc `flag.md` trước để biết trạng thái thật, sau đó đọc `plan.md` để biết các việc sắp làm. `nghien-cuu-de-tai.md` và `mo-ta-chi-tiet-de-tai.md` là tài liệu hiện tại; `bai-bao-chu-de-3.html` là danh mục 50 nguồn; các file có hậu tố `-agent-cu` là lịch sử. Không suy đoán lại đề tài hoặc quay về hướng DevFlow trừ khi có thay đổi mới từ thầy.

## Xác nhận đề tài ban đầu (đã chuyển hướng)

- Đề tài đã được nhận: `Xây dựng agent để tự động hoá trong một lĩnh vực (viết code, giải toán, ...)`.
- Giảng viên hướng dẫn: Thầy Vũ Đình Hồng.
- Xác nhận từ người dùng: Châu Phạm Tuấn Kiệt thông báo nhóm được nhận Top 4 vào ngày 2026-07-26.
- Đây là đề tài khung ban đầu. Ngày 2026-08-14, nhóm báo đã chọn chủ đề nghiên cứu số 3 trong danh sách mới của thầy; phần quyết định mới ở đầu file được ưu tiên.

## Vai trò nghiên cứu hiện tại

- Châu Phạm Tuấn Kiệt, MSSV 523K0011: trưởng nhóm; điều phối source audit, bài báo nền, model/dataset/evaluator revision và reproduction bookkeeping.
- Nguyễn Bá Hùng, MSSV 523K0006: thành viên; literature matrix, difficulty estimator, SD-TokenSkip policy, metrics và error analysis.
- Cả nhóm cùng xin thầy duyệt scope, kiểm tra leakage/license, review config/log và viết bài báo.

## Nền tảng hiện có

- Cả nhóm đã học Lập trình hướng đối tượng, Cấu trúc dữ liệu và giải thuật, Cơ sở dữ liệu và Lập trình web.
- Châu Phạm Tuấn Kiệt đã hoàn thành học phần DevOps, có kiến thức cơ bản về Git, Docker và CI/CD.
- Không tự nhận kinh nghiệm chuyên nghiệp hoặc kỹ năng chưa được xác nhận.

## Lời giới thiệu hướng sản phẩm cũ (lịch sử)

Nhóm em đề xuất xây dựng một hệ thống web có tích hợp AI Agent để hỗ trợ một số công việc lặp lại trong quá trình phát triển phần mềm. Hệ thống tập trung vào giai đoạn từ khi tiếp nhận yêu cầu đến khi lập danh sách công việc và chuẩn bị kiểm thử cơ bản.

Trong quá trình làm dự án, sinh viên hoặc nhóm phát triển nhỏ thường phải tự đọc yêu cầu, chia nhỏ chức năng, lập đầu việc và nghĩ ra các trường hợp kiểm thử. Nếu chưa có nhiều kinh nghiệm, nhóm dễ bỏ sót việc cần làm hoặc tạo test case chưa đầy đủ. Sản phẩm của nhóm em được xây dựng để hỗ trợ các bước này.

Người dùng sẽ tạo dự án, nhập yêu cầu hoặc issue, sau đó hệ thống dùng AI Agent để tạo gợi ý về đầu việc, mức độ ưu tiên, tiêu chí hoàn thành và test case. Kết quả chỉ là đề xuất. Người dùng có thể sửa, phê duyệt hoặc từ chối trước khi lưu vào dự án. Nhóm không xây dựng một chatbot trả lời chung chung, mà xây dựng một luồng hỗ trợ gắn với dữ liệu cụ thể của từng dự án.

## Hướng sản phẩm đã chốt trước đây (lịch sử, không phải roadmap hiện tại)

Tên sản phẩm làm việc là DevFlow Assistant. Đây là tên sản phẩm nội bộ; tên đề tài nộp cho thầy vẫn giữ nguyên theo danh sách chính thức.

Sản phẩm là nền tảng web hỗ trợ một phần quy trình phát triển phần mềm. Khi người dùng nhập yêu cầu hoặc issue, Agent có thể:

1. Tóm tắt yêu cầu và xác định các chức năng chính.
2. Gợi ý user story, danh sách công việc, độ ưu tiên và tiêu chí hoàn thành.
3. Gợi ý test case cơ bản, gồm trường hợp hợp lệ, không hợp lệ và tình huống biên.
4. Tóm tắt tiến độ dự án dựa trên trạng thái công việc đã lưu.
5. Lưu lịch sử đề xuất để người dùng xem lại.

Mọi đề xuất phải được người dùng xem, chỉnh sửa và phê duyệt trước khi trở thành dữ liệu chính thức của dự án.

## Phạm vi MVP trước đây (lịch sử)

- Đăng ký, đăng nhập và phân quyền cơ bản.
- Tạo dự án, yêu cầu, công việc và issue.
- Bảng theo dõi trạng thái công việc và tiến độ dự án.
- AI hỗ trợ phân rã yêu cầu thành đầu việc và gợi ý test case.
- Màn hình xem, sửa, phê duyệt hoặc từ chối đề xuất của AI.
- Lịch sử đề xuất và dashboard thống kê cơ bản.
- Docker hoá ứng dụng và một quy trình CI/CD tối thiểu.

## Ngoài phạm vi phiên bản đầu

- Không để AI tự ý commit code, chạy lệnh hệ thống, triển khai ứng dụng hoặc gọi dịch vụ bên ngoài mà không có phê duyệt.
- Không làm trình chạy mã nguồn hoặc sandbox thực thi.
- Không hứa hẹn tự viết code hoàn chỉnh, giải toán tổng quát hoặc thay thế người quản lý dự án.
- Không biến sản phẩm thành chatbot chung chung; mỗi thao tác AI phải gắn với dự án, yêu cầu, công việc hoặc kiểm thử.
- Tích hợp GitHub/GitLab, tạo release note hoặc đọc repository chỉ là hướng mở rộng sau MVP.

## Công nghệ dự kiến của hướng sản phẩm trước đây (lịch sử)

- Frontend dự kiến: ReactJS hoặc NextJS.
- Backend dự kiến: NodeJS với NestJS hoặc ExpressJS.
- Database dự kiến: PostgreSQL.
- API dự kiến: RESTful API.
- DevOps dự kiến: Git, GitHub/GitLab, Docker, Docker Compose và GitHub Actions hoặc GitLab CI/CD.
- Kiểm thử dự kiến: unit test cho nghiệp vụ quan trọng và API test cơ bản.

Phần AI giữ hướng gọi API như mô tả ban đầu. GPT hoặc Gemini là các lựa chọn chính, đồng thời nhóm có thể tích hợp thêm Kimi, GLM và DeepSeek để thử nghiệm trong quá trình phát triển.

Backend cần có lớp tích hợp AI riêng để phần quản lý dự án, yêu cầu, công việc và kiểm thử không bị phụ thuộc vào một mô hình cụ thể. Khi cần thử mô hình khác, chỉ thay đổi cấu hình hoặc phần tích hợp AI mà không sửa các chức năng nghiệp vụ chính.

## Lịch sử quyết định

- Ban đầu, các đề tài được đánh giá theo khả năng hoàn thiện sản phẩm, giá trị CV và cơ hội thực tập phần mềm tại Việt Nam. Kết luận chung là một dự án full-stack có backend, database, kiểm thử, Docker và CI/CD sẽ thực dụng hơn một chatbot chỉ gọi API.
- Đề tài website/app học lập trình tích hợp AI phân tích code của thầy Dương Hữu Phước từng được ưu tiên, nhưng không còn là lựa chọn hiện tại.
- Đề tài phân tích chương trình, kiểm thử và phân tích tĩnh mã nguồn của thầy Võ Văn Thành đã được chuẩn bị nội dung đăng ký, sau đó không còn khả dụng cho nhóm.
- Đề tài phát triển phần mềm đa nền tảng được dẫn dắt bởi AI của thầy Lê Văn Vang đã được chuẩn bị nội dung đăng ký, sau đó không còn khả dụng cho nhóm.
- Khi danh sách thu hẹp, thứ tự dự phòng đã được xác định là: hệ thống thời gian thực và truyền thông đa phương tiện; hệ thống gợi ý; ứng dụng thực tiễn kết hợp AI; agent tự động hoá; trợ lý pháp lý RAG.
- Nhóm đã soạn mô tả cho các hướng dự phòng Top 2, Top 3 và Top 4.
- Ngày 2026-07-26, nhóm được nhận Top 4: Agent tự động hoá của thầy Vũ Đình Hồng. Từ thời điểm này, không tiếp tục theo các đề tài dự phòng nếu chưa có thay đổi chính thức.

## Tài liệu hiện có

- `nghien-cuu-de-tai.md`: kết luận nghiên cứu 50 bài, khoảng trống, các phương án, DC-TokenSkip, baseline/model/dataset/protocol và các điểm cần xin duyệt.
- `bai-bao-chu-de-3.html`: danh mục 50 bài báo có thể click, tìm kiếm/lọc và phân biệt Published/Preprint/Workshop.
- `phan-tich-chi-tiet-50-bai-chu-de-3.html`: báo cáo HTML chi tiết 50 bản ghi và bảng phân tích tương tác.
- `nghien-cuu-de-tai-agent-cu.md`: tài liệu nghiên cứu DevFlow cũ, được đổi tên để không bị ghi đè.
- `mo-ta-chi-tiet-de-tai.md`: bản mô tả đầy đủ đã viết lại cho chủ đề 3.
- `mo-ta-chi-tiet-de-tai-agent-cu.md`: bản mô tả sản phẩm DevFlow cũ, lưu lịch sử.
- `plan.md`: roadmap nghiên cứu, tái lập baseline, thực nghiệm và viết bài báo.
- `flag.md`: file bối cảnh và trạng thái hiện tại này.

## Việc kế tiếp của hướng nghiên cứu hiện tại

1. Gửi thầy bản tóm tắt chủ đề 3, TokenSkip, Qwen2.5-3B/GSM8K và bảng so sánh fresh baseline/extension; ghi rõ reported result và fresh inference là hai loại bằng chứng khác nhau.
2. Xin thầy xác nhận baseline duy nhất, ngưỡng accuracy `δ`, difficulty bins, budget compute và việc có chấp nhận extension gate hay không.
3. Nếu thầy yêu cầu, chạy thêm đúng entrypoint/revision của official repository để hoàn tất audit reproduction; không đổi model, dataset hoặc precision để làm đẹp kết quả.
4. Sau khi scope được duyệt, thực hiện ablation fixed-ratio/random/oracle và error analysis theo difficulty; nếu không tạo Pareto hợp lệ thì báo cáo kết quả âm.
5. Nếu SD không được thầy chấp nhận, các hướng thay thế cần xin duyệt trước khi chạy GPU: risk-gated TokenSkip với Full-CoT fallback, verifier/uncertainty-triggered retry, staged budget continuation, và difficulty-to-budget policy học trên calibration có held-out validation. Mỗi hướng phải giữ cùng model/dataset/protocol và báo cáo riêng, không trộn với kết quả SD hiện tại.

## Quy tắc cập nhật flag

- `flag.md` là nguồn bối cảnh sống của dự án.
- Khi có phản hồi của thầy, thay đổi đề tài, thay đổi phạm vi, công nghệ đã chốt, mốc hoàn thành hoặc quyết định quan trọng, cập nhật file này trước.
- Mỗi cập nhật cần ghi ngày, nội dung thay đổi, lý do và ảnh hưởng.
- Không ghi chức năng là hoàn thành nếu chưa có bằng chứng triển khai hoặc kiểm thử.
- Trong các lượt làm việc sau, khi người dùng báo thay đổi hoặc yêu cầu tiếp tục dự án, phải đọc và đồng bộ `flag.md` trước. File không thể tự thay đổi khi không có thông tin mới hoặc không có phiên làm việc đang chạy.

## Nhật ký cập nhật

- 2026-07-26: Nhóm được nhận đề tài Agent tự động hoá của thầy Vũ Đình Hồng.
- 2026-07-26: Đã soạn bản mô tả chi tiết đầu tiên tại `mo-ta-chi-tiet-de-tai.md`.
- 2026-08-14: Đọc theo đúng thứ tự `flag.md` → `plan.md` → `mo-ta-chi-tiet-de-tai.md` → `bai-bao-chu-de-3.html` → `nghien-cuu-de-tai.md`; chốt phạm vi CoT compression, không quay lại DevFlow.
- 2026-08-14: Hoàn thành phân tích 50 bản ghi; trước đó đã ghi tạm TokenSkip/Qwen2.5-7B/GSM8K làm reproduction và DRG-TokenSkip làm phương án cũ.
- 2026-08-14: Theo giới hạn phần cứng, chuyển reproduction chính sang `Qwen/Qwen2.5-3B-Instruct` + adapter TokenSkip 3B chính thức trên GSM8K; không gọi 7B/14B reported result là reproduced.
- 2026-08-14: Đọc lại PDF TokenSkip; xác nhận Appendix B.2/Table 2 có kết quả Qwen2.5-3B GSM8K, nên 3B là một cấu hình được bài báo báo cáo, không chỉ là lựa chọn extension từ repository.
- 2026-08-14: Theo yêu cầu mới, loại phương án trước khỏi phương pháp active và chuyển sang DAPB-VC: difficulty chọn budget/ratio khởi đầu, tiếp tục tăng budget/ratio khi trace chưa đủ hoặc verifier phụ không đạt. Pilot base 3×3 giữ 3/3 nhưng tốn 378 so với Full-CoT 260 token/câu; pilot TokenSkip-ratio fallback đạt 2/3 do false-pass, nên chưa được gọi là thành công.
- 2026-08-14: Tạo HTML chi tiết; sao lưu mô tả cũ trước khi viết lại; chưa có kết quả chạy model.
- 2026-07-27: Tạo `flag.md`, chuyển toàn bộ bối cảnh từ `plan.md` sang đây và rút gọn `plan.md` thành roadmap. Viết lại mô tả chi tiết theo văn phong đơn giản hơn, bỏ bảng và chữ in đậm.
- 2026-07-27: Điều chỉnh phần AI theo yêu cầu của nhóm: giữ hướng gọi API GPT/Gemini như ban đầu và bổ sung Kimi, GLM, DeepSeek là các lựa chọn tích hợp thêm.
- 2026-08-14: Nhóm báo đã chọn chủ đề nghiên cứu số 3 về adaptive CoT compression trong danh sách chủ đề mới của thầy. Tạo `nghien-cuu-de-tai.md`, lưu bản nghiên cứu cũ thành `nghien-cuu-de-tai-agent-cu.md`, chuyển roadmap sang nghiên cứu/tái lập/thực nghiệm, giữ `mo-ta-chi-tiet-de-tai.md` nguyên trạng.
- 2026-08-14: Theo yêu cầu mới của nhóm, tạo danh mục 50 bài tại `bai-bao-chu-de-3.html`, viết lại `mo-ta-chi-tiet-de-tai.md` theo hướng nghiên cứu Long-CoT → Short-CoT, lưu mô tả DevFlow cũ thành `mo-ta-chi-tiet-de-tai-agent-cu.md`, và cập nhật roadmap sang bước xin thầy duyệt literature/model/dataset/protocol.
- 2026-08-15: Loại DAPB-VC khỏi active và lưu script/kết quả tại `test/archive-dapb-vc-rejected/`; chọn DC-Token làm ứng viên mới. Pilot 9 câu với trần 512 giữ 6/9 như Full-CoT và giảm 24.6% reasoning tokens, nhưng fixed budget vẫn mạnh hơn trên pilot; chưa gọi là reproduced result.
- 2026-08-15: Theo yêu cầu chuyển phương pháp, loại DC-Token khỏi active và lưu script/kết quả tại `test/archive-dc-token-rejected/`; chuyển sang DC-TokenSkip trên adapter TokenSkip 3B chính thức. Pilot 9 calibration + 9 evaluation, dùng toàn bộ train làm pool, trần 256: Full `4/9`, DC-TokenSkip `5/9`, giảm 24.4% tokens nhưng fixed ratio `0.5` vẫn ít token hơn và latency adapter tăng; chưa gọi là reproduced result.
- 2026-08-15: Loại DC-TokenSkip và TS-DC-TokenSkip khỏi active, lưu script/log/JSON tại `test/archive-dc-tokenskip-rejected/` và `test/archive-ts-dc-tokenskip-rejected/`. Chuyển sang SD-TokenSkip với policy khóa trước test `easy→0.5`, `medium/hard→Full-CoT`; pilot fresh 27 câu đạt cùng Full `26/27`, giảm 14.8% reasoning tokens, latency gần tương đương và fixed `0.9` có trade-off khác. Chỉ gọi đây là pilot/empirical policy, chưa gọi reproduced result.
- 2026-08-17: Hoàn tất fresh FP16 original-ratio baseline và extension gate trên cùng model `Qwen/Qwen2.5-3B-Instruct`, adapter TokenSkip 3B và GSM8K; mỗi run đủ 9 calibration/1.319 evaluation rows, batch 288, không 4-bit. Baseline ratio `0.5` đạt `74.8294%` accuracy và giảm `46.398241%` reasoning tokens so với Full-CoT nhưng mất `8.6429` điểm accuracy; extension gate đạt `81.5770%`, giảm `11.42989%` token và mất `1.8953` điểm. Lưu artifact ở hai thư mục FP16 và push commit `17b7eea`; không gọi đây là paper reported result.
- 2026-08-17: Hoàn tất multi-level SD-TokenSkip 3B và Full-CoT 7B trên Vast.ai RTX 3090. 3B đủ 1.319 rows, policy calibration chọn `easy→1.0`, `medium→0.5`, `hard→1.0`, giảm `16.272983%` token nhưng mất `3.1842` điểm accuracy so với Full-CoT. 7B Full-CoT đủ 1.319 rows, đạt `91.5845%` accuracy; đây chỉ là fresh baseline cùng GSM8K/model scale, chưa phải 7B SD và không phải paper reported result. Lưu artifacts local và summary `test/comparison_3b_7b.json`.

## Kết quả nghiên cứu kỹ thuật 2026-07-27 (hướng sản phẩm cũ, lưu lịch sử)

### Quyết định về cách gọi Agent

- Đề tài vẫn giữ nguyên: xây dựng agent tự động hoá trong lĩnh vực hỗ trợ quy trình phát triển phần mềm.
- Cách mô tả kỹ thuật chính xác là **bounded human-in-the-loop agentic workflow** hoặc **AI Agent hỗ trợ có kiểm soát**; không tuyên bố agent tự trị hoàn toàn.
- Để đủ cơ sở gọi là Agent, thiết kế phải có mục tiêu rõ, lấy context dự án qua công cụ đọc, state/run, điều phối nhiều bước, structured proposal, schema/semantic validation, điều kiện dừng, approval gate, history/trace và evaluation.
- Agent chỉ đề xuất tasks, priority, acceptance criteria và test cases. Không bổ sung sandbox chạy code, tự viết code hoàn chỉnh, tự commit, tự deploy hay chatbot chung chung.
- Proposal AI không được trở thành dữ liệu chính thức ngay sau khi model trả về. Chỉ bản đã được người dùng review và approve mới được backend lưu vào task/test canonical.

### Quyết định kiến trúc và MVP

- Kiến trúc ưu tiên là modular monolith cho nhóm hai người: NextJS/React, NodeJS/NestJS, PostgreSQL, REST API, Docker Compose và CI tối thiểu.
- Backend phải có lớp AI service/provider adapter độc lập với domain; lưu provider/model, prompt version, schema version, run state, validation result, latency/usage và approval history.
- MVP chỉ cần chọn **một provider/model primary** để hoàn thiện luồng end-to-end và evaluation. Không coi việc tích hợp đồng thời GPT, Gemini, Kimi, GLM và DeepSeek là điều kiện bắt buộc của MVP.
- Provider thứ hai chỉ nên thêm sau khi có baseline/eval, để so sánh chất lượng, coverage, edit burden, latency/cost hoặc làm fallback có kiểm soát. Việc này không thay đổi phạm vi sản phẩm.
- Research-only: chưa có chức năng nào được ghi là đã triển khai hoặc đã kiểm thử; `mo-ta-chi-tiet-de-tai.md` không bị chỉnh sửa.

### Thông tin API/model đã xác minh tại ngày 2026-07-27

- OpenAI: tài liệu model hiện nêu dòng GPT-5.6 (`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`) và Responses API; cần dùng Structured Outputs/Function Calling phù hợp.
- Gemini: tài liệu model hiện nêu Gemini 3.6 Flash, 3.5 Flash/Flash-Lite và các dòng liên quan; có Structured Output và Function Calling.
- Kimi: tài liệu platform hiện nêu Kimi K3 là flagship, dùng Chat Completions tương thích OpenAI và nên lấy model ID từ endpoint model list.
- GLM/Z.AI: tài liệu API hiện đặt `glm-5.2` làm model mặc định, có thinking/reasoning effort và function calling.
- DeepSeek: tài liệu hiện nêu `deepseek-v4-flash`/`deepseek-v4-pro`; changelog nói `deepseek-chat` và `deepseek-reasoner` đã ngừng từ 2026-07-24, không được hardcode tên legacy.
- Không gọi hosted API/model là “open source” nếu chưa kiểm tra đúng model card và license. Đã kiểm tra riêng: Kimi K2.5 có Modified MIT trong repository chính thức; GLM-4.5/4.5-Air ghi MIT; DeepSeek-V3 có code MIT nhưng model license riêng. Các thông tin này không được suy ra cho model hosted/current khác.

Nguồn chính thức của hướng sản phẩm cũ được lưu trong `nghien-cuu-de-tai-agent-cu.md`, gồm OpenAI, Anthropic, Gemini, Kimi, Z.AI, DeepSeek, NIST AI RMF, ISTQB/ISO 29119, OpenTelemetry, NextJS, NestJS, Docker, GitHub Actions và PostgreSQL.
