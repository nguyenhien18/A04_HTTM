# Assignment 4 – So sánh CNN trên ba bài toán

Project thực hiện và so sánh ba cách triển khai mạng tích chập:

- NumPy **From Scratch**: tự cài đặt forward/backward và quá trình cập nhật trọng số.
- **PyTorch**: triển khai mô hình bằng `torch.nn`.
- **TensorFlow/Keras**: triển khai mô hình bằng `tf.keras`.

Ba bộ dữ liệu được sử dụng gồm:

1. **MNIST** – phân loại ảnh chữ số viết tay, 10 lớp.
2. **Diabetes Health Indicators** – phân loại nhị phân nguy cơ tiểu đường.
3. **Poland OLX House Price** – hồi quy giá nhà.

Mỗi notebook đều so sánh mô hình **Basic** và **Improved**, đồng thời lưu metric, biểu đồ loss theo epoch và trọng số mô hình.

## Cấu trúc thư mục

```text
ASignment4/
├── data/
│   ├── mnist_train.csv
│   ├── mnist_test.csv
│   ├── diabete.csv
│   └── house.csv
├── models/                 # Trọng số và bộ tiền xử lý đã lưu
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── B23DCCN290_01.ipynb  # MNIST
│   ├── B23DCCN290_02.ipynb  # Diabetes
│   └── B23DCCN290_03.ipynb  # House Price
├── results/                # Bảng so sánh metric dạng CSV
└── README.md
```

## Môi trường

Yêu cầu Python 3.x và các thư viện:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
torch
tensorflow
jupyter
```

Notebook `00_setup.ipynb` có thể kiểm tra và cài các thư viện còn thiếu. Có thể cài thủ công bằng lệnh:

```bash
pip install numpy pandas matplotlib scikit-learn joblib torch tensorflow jupyter
```

GPU không bắt buộc. Nếu PyTorch nhận diện được CUDA, notebook sẽ tự sử dụng GPU.

## Cách chạy

Từ PowerShell hoặc terminal:

```bash
cd D:\ASignment4\notebooks
jupyter lab
```

Thực hiện theo thứ tự:

1. Mở và chạy toàn bộ `00_setup.ipynb` một lần.
2. Chạy `B23DCCN290_01.ipynb` để huấn luyện và đánh giá trên MNIST.
3. Chạy `B23DCCN290_02.ipynb` để huấn luyện và đánh giá trên Diabetes.
4. Chạy `B23DCCN290_03.ipynb` để huấn luyện và đánh giá trên House Price.

Các notebook sử dụng seed `42` và mặc định huấn luyện trên toàn bộ tập train. Nếu máy chạy chậm, có thể đặt biến `TRAIN_LIMIT` trong notebook, nhưng cần dùng cùng giới hạn dữ liệu khi so sánh các implementation.

## Phương pháp và metric

### MNIST

- Input: ảnh xám kích thước `28 × 28`.
- Basic CNN: `Conv2D → ReLU → Conv2D → ReLU → MaxPool2D → Flatten → Dense`.
- Improved CNN: tăng số filter, thêm pooling, Batch Normalization và Dropout.
- Metric: Accuracy, Precision, Recall, F1 và loss theo epoch.

### Diabetes

- Input: dữ liệu dạng bảng, được chuẩn hóa trước khi đưa vào `Conv1D`.
- Bài toán: binary classification.
- Dữ liệu bị mất cân bằng lớp, vì vậy phiên bản cải tiến sử dụng weighted BCE/class weight.
- Metric: Accuracy, Precision, Recall, F1 và loss theo epoch.

Khi đọc kết quả Diabetes, không nên chỉ nhìn Accuracy; cần ưu tiên thêm Recall và F1 của lớp dương.

### House Price

- Input: dữ liệu thuộc tính nhà ở dạng số và categorical.
- Categorical feature được mã hóa, numerical feature được chuẩn hóa.
- Target dùng `log1p(price)` để giảm ảnh hưởng của phân phối lệch phải và outlier.
- Khi đánh giá, dự đoán được biến đổi ngược về đơn vị giá ban đầu.
- Metric: MAE, RMSE, R² và loss theo epoch.

## Kết quả đã lưu

Các file `*_fixed.csv` là kết quả của phiên bản notebook đã hiệu chỉnh và được ưu tiên sử dụng khi xem báo cáo:

| Dataset | File kết quả | Metric tiêu biểu trong kết quả hiện tại |
|---|---|---|
| MNIST | `results/mnist_comparison_fixed.csv` | PyTorch Improved: Accuracy `98.79%`, F1 `98.78%` |
| Diabetes | `results/diabetes_comparison_fixed.csv` | TensorFlow Basic: F1 `0.4450`, Recall `0.7633` |
| House Price | `results/house_comparison_fixed.csv` | TensorFlow Basic: RMSE khoảng `178,674`, R² `0.4320` |

Các file không có hậu tố `_fixed` được giữ lại để đối chiếu với lần chạy trước.

## Lưu ý

- Không đổi tên các file trong `data/` nếu chưa sửa lại đường dẫn trong notebook.
- Cần chạy notebook từ thư mục `notebooks/` vì code sử dụng `Path("..")` để truy cập `data/`, `models/` và `results/`.
- Khi chạy lại, các file model và result có thể bị ghi đè.
- Kết quả có thể thay đổi nhẹ tùy phiên bản Python, TensorFlow/PyTorch, CPU/GPU và giới hạn dữ liệu được chọn.

