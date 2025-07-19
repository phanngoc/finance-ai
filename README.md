# Finance AI - Phân tích Cổ phiếu Việt Nam

Ứng dụng phân tích và dự đoán giá cổ phiếu Việt Nam sử dụng Machine Learning (LSTM) và Streamlit.

## 🚀 Tính năng chính

- **Phân tích kỹ thuật**: Biểu đồ nến, khối lượng giao dịch
- **Dự đoán LSTM**: Dự đoán giá cổ phiếu sử dụng mạng neural LSTM
- **Tín hiệu trading**: Phân tích tín hiệu mua/bán tự động
- **Phân tích rủi ro**: Đánh giá độ biến động và rủi ro đầu tư
- **Giao diện thân thiện**: Streamlit web app dễ sử dụng

## 📁 Cấu trúc Project

```
finance-ai/
├── streamlit_app.py          # Main Streamlit application
├── main.py                   # Alternative entry point
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── data_processing.py    # Data loading and processing functions
│   ├── lstm_model.py         # LSTM model utilities
│   ├── trading_analysis.py   # Trading signal analysis functions
│   └── plotting.py           # Chart and visualization functions
└── README.md                 # Project documentation
```

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd finance-ai
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
streamlit run streamlit_app.py
```

## 📊 Sử dụng

1. **Nhập mã cổ phiếu** trong sidebar (VD: ACB, VCB, VHM, FPT)
2. **Chọn khoảng thời gian** phân tích
3. **Nhấn "Tải dữ liệu"** để xem biểu đồ và thống kê
4. **Kích hoạt dự đoán LSTM** để xem tín hiệu trading và khuyến nghị

## 🧩 Module Structure

### 📈 data_processing.py
- `load_stock_data()`: Tải dữ liệu cổ phiếu từ VCI
- `get_basic_stats()`: Tính toán thống kê cơ bản
- `prepare_prediction_dataframe()`: Chuẩn bị DataFrame cho dự đoán
- `format_prediction_table()`: Format bảng dự đoán để hiển thị

### 🤖 lstm_model.py
- `prepare_lstm_data()`: Chuẩn bị dữ liệu cho LSTM
- `create_lstm_model()`: Tạo và compile mô hình LSTM
- `train_lstm_model()`: Huấn luyện mô hình
- `make_predictions()`: Thực hiện dự đoán
- `calculate_model_accuracy()`: Tính toán độ chính xác

### 📊 trading_analysis.py
- `analyze_trading_signals()`: Phân tích tín hiệu trading tổng hợp
- `generate_trading_signals()`: Tạo tín hiệu mua/bán
- `find_optimal_buy_points()`: Tìm điểm mua tối ưu
- `find_optimal_sell_points()`: Tìm điểm bán tối ưu
- `calculate_potential_profit()`: Tính toán lợi nhuận tiềm năng

### 📉 plotting.py
- `create_combined_chart()`: Biểu đồ kết hợp nến + volume
- `create_comparison_chart()`: So sánh giá thực tế vs dự đoán
- `create_trading_signals_chart()`: Biểu đồ tín hiệu trading
- `create_accuracy_scatter_plot()`: Biểu đồ scatter độ chính xác

## 🔧 Dependencies

### Core
- `streamlit`: Web app framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `plotly`: Interactive charts

### Data Source
- `vnstock`: Vietnam stock data API

### Machine Learning
- `scikit-learn`: Data preprocessing
- `tensorflow/keras`: LSTM neural networks

## ⚠️ Lưu ý quan trọng

- **Không phải lời khuyên đầu tư**: Ứng dụng chỉ mang tính chất tham khảo
- **Có rủi ro**: Thị trường chứng khoán luôn có rủi ro cao
- **Tự nghiên cứu**: Luôn tự nghiên cứu kỹ trước khi đầu tư
- **Tham khảo chuyên gia**: Nên tham khảo ý kiến chuyên gia tài chính

## 🎯 Best Practices được áp dụng

### 1. **Separation of Concerns**
- Tách logic thành các module riêng biệt
- Mỗi module có trách nhiệm cụ thể

### 2. **Code Reusability**
- Các hàm có thể tái sử dụng trong nhiều context
- Interface rõ ràng với docstring đầy đủ

### 3. **Error Handling**
- Xử lý lỗi ở nhiều cấp độ
- Thông báo lỗi thân thiện với người dùng

### 4. **Performance Optimization**
- Sử dụng `@st.cache_data` cho các hàm tốn kém
- Tối ưu hóa việc tính toán và hiển thị

### 5. **Clean Code**
- Tên hàm và biến có ý nghĩa
- Code được organize theo logic rõ ràng
- Comments và documentation đầy đủ

## 🔄 Workflow

1. **Load Data** → `data_processing.load_stock_data()`
2. **Process Data** → `data_processing.get_basic_stats()`
3. **Visualize** → `plotting.create_combined_chart()`
4. **Prepare ML Data** → `lstm_model.prepare_lstm_data()`
5. **Train Model** → `lstm_model.create_lstm_model()` + `train_lstm_model()`
6. **Make Predictions** → `lstm_model.make_predictions()`
7. **Analyze Signals** → `trading_analysis.analyze_trading_signals()`
8. **Visualize Results** → `plotting.create_trading_signals_chart()`

## 📝 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request
