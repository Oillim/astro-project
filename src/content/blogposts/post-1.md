
---

title: Advanced Git
description: More advanced git subjects. That should be useful in a professional working environnement.
date: 2024-08-26 19:34
tags: 
- Code
heroImage:

---

Link bài báo: [Speculative RAG](https://arxiv.org/pdf/2407.08223)

Bài báo được nói đến được ra mắt bởi đội ngũ Google Research, mang đến sự đột phá trong việc áp dụng RAG cho các mô hình ngôn ngữ lớn (LLMs). 

Speculative RAG là 1 cấu trúc gồm 2 bộ phận chính:
- Một mô hình ngôn ngữ lớn hơn bao quát đảm nhiệm vai trò xác minh các bản nháp RAG (RAG drafts).
- Các bản nháp RAG được tạo ra đồng thời từ 1 mô hình ngôn ngữ nhỏ hơn chuyên biệt.
- Mỗi bản nháp được tạo sinh từ tập các tài liệu truy vấn, việc này giúp tạo ra nhiều ngữ cảnh đa dạng hơn đồng thời giảm số lượng token của mỗi bản nháp.

![[Pasted image 20240826185823.png]]

(a) RAG truyền thống: kết hợp tất cả các tài liệu vào prompt. Điều này làm tăng độ dài đầu vào và làm chậm quá trình suy luận.
(b) RAG tự phản chiếu: yêu cầu mô hình gắn nhãn trong quá trình suy luận và tự nhìn lại trong quá trình tuning có hướng dẫn.
(c) RAG kiểm tra tính đúng đắn: sử dụng mô hình NLI để phân loại chất lượng của tài liệu (đúng/ mơ hồ/ sai), tập trung vào thông tin mà bỏ qua khả năng lý luận.
(d) RAG quan sát: như trên.

## RAG quan sát qua chọn lọc

**Problem Formulation:** trong mỗi mục gồm 3 thành phần (*Q, D, A*). Trong đó *Q* là câu hỏi, *D* là tập các tài liệu trong *n* tài liệu truy vấn từ CSDL và *A* là câu trả lời. 

Bài báo này cho ta một cách tiếp cận mới sử dụng chia-để-trị thay vì phụ thuộc vào brute-force hay instruction-tuning như cũ. Khi này, tác giả dùng mô hình ngôn ngữ chuyên biệt nhỏ, gọi là bộ chọn lọc RAG, để liên tục tạo các câu trả lời nháp dựa trên kết quả truy vấn. Tiếp theo đó, một mô hình ngôn ngữ tổng quát lớn, gọi là bộ xác minh RAG, sẽ tiếp cận các bản nháp, lọc và chọn ra bản nháp tốt nhất dựa trên tính hợp lý của nó, tích hợp vào kết quả cuối cùng.

Dưới đây là phần mã giả mà nhóm tác giả cung cấp:
![[Pasted image 20240826192635.png]]

Giải thích sơ bộ:
- Tại dòng ~={red}2=~, ta sử dụng thuật toán k-means để phân nhóm các tài liệu vào k nhóm dựa vào sự liên quan giữa chúng. Mỗi nhóm thể hiện 1 khía cạnh riêng trong kết quả truy vấn.
- Từ dòng ~={red}5=~ - ~={red}8=~, ta "góp nhặt" 1 tài liệu trong mỗi cluster, tạo thành 1 tập con. Việc này giúp tối thiểu sự dư thừa và tối đa sự đa dạng của các tài liệu. 
- Dòng ~={red}12=~ sử dụng mô hình Drafter để tạo bản nháp (câu trả lời $\alpha$, lý luận $\beta$).
- Dòng ~={red}13=~ áp dụng mô hình Verifier để tính điểm dựa trên câu trả lời $\alpha$, câu hỏi *Q* và lý luận $\beta$.
- Toàn bộ quá trình từ dòng ~={red}11=~ - ~={red}14=~ được xử lý song song (in parallel).
- Cuối cùng, tại dòng ~={red}15=~, dựa trên bộ điểm đánh giá từ mô hình Verifier, ta sẽ chọn ra câu trả lời với điểm số cao nhất và tích hợp vào kết quả cuối cùng của mô hình.

Cần lưu ý rằng, tác giả có đề cập đến việc mô hình Verifier không cần phải được instruction-tuned vì khả năng mô hình ngôn ngữ hoá của bản thân nó đã được học trong quá trình pre-training. Trong khi đó, mô hình Verifier có thể xác minh các bản nháp dựa trên các suy luận cung cấp từ mô hình Drafter thay vì phải xử lý lại các tài liệu dư thừa và không có ích.

### Specialist RAG Drafter

Specialist RAG Drafter hay Chuyên viên chọn lọc RAG (nghe cứ như một con người :v), bao gồm các thành phần sau:
+ **Instruction Tuning**: Cho bộ ba (*Q, A, D*), ta tăng cường bằng cách thêm lý luận phân tách *E*. lý luận *E* trích xuất các thông tin tiềm năng và giải thích lý do phản hồi *A* phù hợp với truy vấn *Q*. Để fine-tune mô hình pre-trained, ta sử dụng maximizing the likelihood:
>[!note]+ Formula
> $$
> E_{(Q,A,D,E)} \log P_{M_{Drafter}}(A, E| Q, D)
> $$
> $(Q,A,D,E)$ là 1 mục được tăng cường trong bộ dữ liệu.
> $P_{M_{Drafter}}(A,E|Q,D)$ là xác suất tạo ra phản hồi và lý luận khi đã biết về truy vấn và tài liệu.


- **Multi-Perspective Sampling:** Quá trình lấy mẫu đa dạng ngữ cảnh diễn ra như sau: Với mỗi câu hỏi, ta chọn ra tập tài liệu từ CSDL. Các tài liệu này chứa đa dạng nội dung mặc cho sự *mơ hồ vốn có* (ambiguity inherent) trong truy vấn. Phương pháp này giúp giảm thiểu sự dư thừa và tăng cường sự đa dạng của tài liệu dùng để tạo câu trả lời nháp.
>[!Note]+ Formula
>$$
>\text{emb}(d_1),...,\text{emb}(d_n) = \mathcal{E}(d_1,...,d_n|Q)
>$$
>$$
>\{c_1,...,c_k\} = \text{K-Means}(\text{emb}(d_1),...,\text{emb}(d_n))
>$$
>$$
>\delta = \{\text{random.sample}(c) \text{ for } c \in \{c_i\}^k_1\}
>$$

Trong đó $\mathcal{E}$ tượng trưng cho mô hình nhúng nhận thức có hướng dẫn (instruction-aware embedding model), $\text{emb}(d_i)$ là tài liệu nhúng $d_i$, $c_j$ là 1 cụm giữa các tài liệu truy xuất với chủ đề và nội dung tương tự nhau, $k$ là siêu tham số kiểm soát số lượng cụm được tạo ra, $\delta$ là tập mẫu được tạo ra từ việc lấy ngẫu nhiên tài liệu trong mỗi cụm (mỗi tập mẫu sẽ chứa $k$ tài liệu). Kết quả ta sẽ có $m$ tập con.

- **RAG Drafting**: Sau quá trình lấy mẫu, ta đưa $m$ tập này để tạo ra câu trả lời nháp. Với tập dữ liệu $\delta_j$, mô hình Drafter tạo ra câu trả lời và lý luận bản nháp theo prompt được định sẵn. Quá trình này sẽ được thực hiện song song. Khi này, xác suất tạo sinh có điều kiện được tính bằng $\rho_{Draft,j} = P(\beta_j | Q, \theta_j) + P(\alpha_j|Q, \theta_j, \beta_j)$, được dùng để đo lường tính tin cậy của các lý luận và mức độ tự tin của câu trả lời.

### Generalist RAG Verifier
Sau khi đã tạo ra các bản nháp, ta tiến hành đánh giá chúng dựa vào một mô hình ngôn ngữ tổng quát $\mathcal{M}_{Verifier}$ để lọc ra ứng viên đáng tin cậy nhất.

- **Evaluation Scores:** Điểm đánh giá này được tính bằng xác suất sinh ra bản nháp $(\alpha, \beta)$  với câu hỏi cho trước $\rho_{Self-contain} = P(\alpha, \beta|Q)$ , hay còn gọi là **self-consistency score** (điểm tự đồng nhất).  Điểm này giúp xác định liệu có sự nhất quán giữa câu trả lời nháp với lý luận hay không. Bên cạnh đó, nhóm tác giả đã kết hợp thêm cho mô hình khả năng tự nhìn lại các lý luận để xem nếu lý luận đó hỗ trợ cho câu trả lời. Ta gọi nó là câu phát biểu $R$ được thêm vào câu lệnh cho mô hình. (Ví dụ: "Do you think rationale supports the answer, yes or no?"). Cách tính điểm cho khả năng tự phản chiếu là $\rho_{Self\_reflect} =  P('Yes'|Q,\alpha,\beta,R)$, tức xác suất mà mô hình trả lời "Yes" trong câu phát biểu đó.
- 
- **Computation Method:** Với việc đã giới thiệu các điểm đánh giá ở trên, ta sẽ có cách mà mô hình chọn lọc ứng viên như sau:
	- Với câu hỏi $Q$ và cặp $(\alpha, \beta)$, ta sẽ tạo câu prompt gồm các thành phần $[ Q, \alpha, \beta, R, 'Yes']$ và mã hoá đoạn prompt này. 
	- Tiếp theo, ta tính toán xác suất có điều kiện của mỗi token với các token trước nó $P(t_i|t_{<i})$ theo công thức như bên dưới:![[Pasted image 20240828174009.png]]
	- Cuối cùng, ta tính điểm $\rho_j=\rho_{Draft,j} \cdot \rho_{SC,j} \cdot \rho_{SR,j}$ và chọn ra câu trả lời tốt nhất $\hat{A} = \arg\max_{\alpha_j}\rho_j$.

### Thí nghiệm 
(xem thêm trong paper)


Tags: #ABC

























