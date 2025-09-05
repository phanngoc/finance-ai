"""
LSTM Stock Prediction App - Standalone Version
Ứng dụng dự đoán giá cổ phiếu Việt Nam sử dụng LSTM
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom utilities
from utils.data_processing import (
    load_stock_data, get_basic_stats, prepare_prediction_dataframe, 
    format_prediction_table
)
from utils.lstm_model import (
    prepare_lstm_data, create_lstm_model, train_lstm_model, 
    make_predictions, calculate_model_accuracy, predict_future_prices, KERAS_AVAILABLE
)
from utils.plotting import (
    create_combined_chart, create_comparison_chart, create_accuracy_scatter_plot
)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="LSTM Stock Prediction - Việt Nam",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề chính
st.title("🤖 LSTM Stock Prediction - Việt Nam")
st.markdown("Dự đoán giá cổ phiếu sử dụng mạng neural LSTM")
st.markdown("---")

# Sidebar để cấu hình
st.sidebar.header("⚙️ Cấu hình")
symbol = st.sidebar.text_input(
    "Mã cổ phiếu", 
    value="ACB", 
    help="Nhập mã cổ phiếu (VD: ACB, VCB, VHM, FPT)"
)
start_date = st.sidebar.date_input(
    "Ngày bắt đầu", 
    value=pd.to_datetime("2024-01-01")
)
end_date = st.sidebar.date_input(
    "Ngày kết thúc", 
    value=pd.Timestamp.today()
)

# LSTM Configuration
st.sidebar.subheader("🧠 Cấu hình LSTM")
lookback_days = st.sidebar.slider(
    "Số ngày lookback", 
    min_value=30, 
    max_value=120, 
    value=60,
    help="Số ngày dữ liệu quá khứ để dự đoán"
)
epochs = st.sidebar.slider(
    "Số epochs", 
    min_value=10, 
    max_value=50, 
    value=20,
    help="Số lần huấn luyện mô hình"
)
enable_future_prediction = st.sidebar.checkbox(
    "Dự đoán 10 ngày tới", 
    value=True,
    help="Hiển thị dự đoán giá cho 10 ngày kinh doanh tiếp theo"
)

# Nút để tải dữ liệu
if st.sidebar.button("🚀 Bắt đầu phân tích", type="primary"):
    st.session_state.load_data = True

# Kiểm tra và tải dữ liệu
if 'load_data' not in st.session_state:
    st.session_state.load_data = True

if st.session_state.load_data:
    with st.spinner(f"Đang tải dữ liệu cho {symbol}..."):
        df, error = load_stock_data(symbol, start_date, end_date)
    
    if error:
        st.error(f"❌ Lỗi khi tải dữ liệu: {error}")
        st.stop()
    
    if df is None or df.empty:
        st.warning("⚠️ Không có dữ liệu cho mã cổ phiếu này trong khoảng thời gian đã chọn.")
        st.stop()
    
    # Hiển thị thông tin cơ bản
    col1, col2, col3, col4 = st.columns(4)
    
    # Get basic statistics
    stats = get_basic_stats(df)
    
    with col1:
        st.metric("📅 Tổng số ngày", stats['total_days'])
    
    with col2:
        st.metric("💰 Giá đóng cửa mới nhất", f"{stats['latest_price']:,.0f} VND")
    
    with col3:
        change_color = "normal"
        if stats['change_percent'] > 0:
            change_color = "normal"
        elif stats['change_percent'] < 0:
            change_color = "inverse"
        
        st.metric(
            "📊 Thay đổi hôm nay", 
            f"{stats['price_change']:,.0f} VND", 
            f"{stats['change_percent']:.2f}%",
            delta_color=change_color
        )
    
    with col4:
        st.metric("📈 Khối lượng TB", f"{stats['avg_volume']:,.0f}")
    
    st.markdown("---")

    # Biểu đồ kết hợp
    st.subheader(f"📈 Phân tích tổng hợp {symbol}")
    fig_combined = create_combined_chart(df, symbol)
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # LSTM Price Prediction Section
    st.markdown("---")
    st.subheader("🤖 Dự đoán giá sử dụng LSTM")
    
    if not KERAS_AVAILABLE:
        st.error("❌ TensorFlow/Keras chưa được cài đặt. Vui lòng cài đặt để sử dụng chức năng dự đoán LSTM.")
        st.code("pip install tensorflow", language="bash")
        st.stop()
    
    # Kiểm tra dữ liệu đủ để huấn luyện
    if len(df) < lookback_days + 30:
        st.warning(f"⚠️ Cần ít nhất {lookback_days + 30} ngày dữ liệu để huấn luyện mô hình LSTM. Hiện tại chỉ có {len(df)} ngày.")
        st.stop()
    
    # Initialize session state for LSTM prediction
    if 'run_lstm_prediction' not in st.session_state:
        st.session_state.run_lstm_prediction = False
    
    # Button to trigger LSTM prediction
    col_lstm1, col_lstm2 = st.columns([2, 1])
    
    with col_lstm1:
        if st.button("🚀 Chạy mô hình dự đoán LSTM", 
                    type="primary", 
                    help="Có thể mất vài phút để huấn luyện mô hình",
                    key="lstm_prediction_button"):
            st.session_state.run_lstm_prediction = True
    
    with col_lstm2:
        if st.session_state.run_lstm_prediction:
            if st.button("🔄 Reset", help="Xóa kết quả dự đoán", key="reset_lstm_button"):
                st.session_state.run_lstm_prediction = False
                st.rerun()
    
    # Run LSTM prediction if button was clicked
    if st.session_state.run_lstm_prediction:
        
        with st.spinner("Đang chuẩn bị dữ liệu và huấn luyện mô hình LSTM..."):
            try:
                # Chuẩn bị dữ liệu
                X, y, scaler = prepare_lstm_data(df, lookback_days)
                
                # Tạo và huấn luyện mô hình
                model = create_lstm_model(X, y)
                
                # Progress bar cho quá trình huấn luyện
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_size = 32
                
                # Huấn luyện với callback để cập nhật progress
                for epoch in range(epochs):
                    status_text.text(f'Epoch {epoch + 1}/{epochs}')
                    train_lstm_model(model, X, y, epochs=1, batch_size=batch_size, verbose=0)
                    progress_bar.progress((epoch + 1) / epochs)
                
                status_text.text('✅ Hoàn thành huấn luyện!')
                
                # Dự đoán
                predicted_prices = make_predictions(model, X, scaler)
                
                # Sử dụng dữ liệu thực tế từ DataFrame
                real_prices_original = df['close'].iloc[lookback_days:].values.reshape(-1, 1)
                
                # Đảm bảo kích thước khớp nhau
                min_length = min(len(predicted_prices), len(real_prices_original))
                predicted_prices = predicted_prices[:min_length]
                real_prices = real_prices_original[:min_length]
                
                # Tạo index cho dữ liệu dự đoán
                prediction_dates = df.index[lookback_days:lookback_days+min_length]
                
                # Tạo DataFrame cho dữ liệu dự đoán
                prediction_df = prepare_prediction_dataframe(prediction_dates, real_prices, predicted_prices)
                
                # === SECTION: FUTURE PREDICTIONS ===
                if enable_future_prediction:
                    st.markdown("---")
                    st.subheader("🔮 Dự đoán Giá 10 Ngày Tới")
                    
                    with st.spinner("Đang dự đoán giá cho 10 ngày tới..."):
                        try:
                            # Predict future prices
                            future_prices = predict_future_prices(model, df, scaler, lookback=lookback_days, days_ahead=10)
                            
                            # Generate future dates (business days only)
                            last_date = df.index[-1]
                            today = datetime.now().date()
                            
                            # Convert last_date to proper format
                            if hasattr(last_date, 'date'):
                                last_date_clean = last_date.date()
                            else:
                                try:
                                    last_date_clean = pd.to_datetime(str(last_date)).date()
                                except:
                                    last_date_clean = today
                            
                            # Generate future business dates
                            future_dates = []
                            current_date = last_date_clean + timedelta(days=1)
                            
                            while len(future_dates) < 10:
                                # Skip weekends (Monday=0, Sunday=6)
                                if current_date.weekday() < 5:  # Monday to Friday
                                    future_dates.append(current_date)
                                current_date += timedelta(days=1)

                            # Display future predictions in a table
                            future_df = pd.DataFrame({
                                'Ngày': [date.strftime('%d/%m/%Y') for date in future_dates],
                                'Dự đoán giá đóng cửa (VND)': [f"{price:,.0f}" for price in future_prices],
                                'Thay đổi từ hôm nay (%)': [f"{((price - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100):+.2f}%" for price in future_prices]
                            })

                            # Future prediction analysis
                            col_analysis1, col_analysis2 = st.columns(2)
                            
                            with col_analysis1:
                                current_price = df['close'].iloc[-1]
                                avg_future_price = np.mean(future_prices)
                                price_change_pct = ((avg_future_price - current_price) / current_price) * 100
                                
                                st.metric(
                                    "📊 Giá trung bình dự đoán (10 ngày)",
                                    f"{avg_future_price:,.0f} VND",
                                    f"{price_change_pct:+.2f}%"
                                )
                                
                                max_future_price = np.max(future_prices)
                                min_future_price = np.min(future_prices)
                                volatility_future = ((max_future_price - min_future_price) / min_future_price) * 100
                                
                                st.metric(
                                    "📈 Biên độ biến động dự đoán",
                                    f"{volatility_future:.2f}%",
                                    f"từ {min_future_price:,.0f} đến {max_future_price:,.0f} VND"
                                )
                            
                            with col_analysis2:
                                # Trend analysis for future predictions
                                if future_prices[-1] > future_prices[0]:
                                    trend_direction = "📈 Tăng"
                                    trend_color = "#28a745"
                                else:
                                    trend_direction = "📉 Giảm"
                                    trend_color = "#dc3545"
                                
                                trend_strength = abs(((future_prices[-1] - future_prices[0]) / future_prices[0]) * 100)
                                
                                st.markdown(f"""
                                <div style="background-color: {trend_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                                    <h4>Xu hướng 10 ngày: {trend_direction}</h4>
                                    <p><strong>Cường độ:</strong> {trend_strength:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Investment recommendation based on future trend
                                if price_change_pct > 5:
                                    st.success("💡 **Khuyến nghị:** Xu hướng tích cực, có thể cân nhắc mua")
                                elif price_change_pct < -5:
                                    st.error("⚠️ **Khuyến nghị:** Xu hướng tiêu cực, nên thận trọng")
                                else:
                                    st.info("📊 **Khuyến nghị:** Xu hướng ổn định, chờ tín hiệu rõ ràng hơn")
                            
                            # Display future predictions table
                            st.subheader("📋 Bảng dự đoán chi tiết")
                            st.dataframe(future_df, use_container_width=True)
                            
                            st.warning("""
                            **⚠️ Lưu ý quan trọng về dự đoán tương lai:**
                            - Dự đoán càng xa thì độ tin cậy càng giảm
                            - Các sự kiện bất ngờ có thể thay đổi hoàn toàn xu hướng giá
                            - Luôn kết hợp với phân tích cơ bản và tin tức thị trường
                            - Không nên dựa hoàn toàn vào dự đoán AI để đầu tư
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi khi dự đoán giá tương lai: {str(e)}")
                            st.info("Hãy thử lại hoặc kiểm tra dữ liệu đầu vào.")

                # So sánh Thực tế vs Dự đoán
                st.subheader("📊 So sánh Thực tế vs Dự đoán")
                fig_comparison = create_comparison_chart(prediction_df)
                st.plotly_chart(fig_comparison, use_container_width=True)

                # === SECTION: MODEL ACCURACY ===
                st.markdown("---")
                st.subheader("🎯 Độ chính xác Mô hình")
                
                # Tính toán độ chính xác
                accuracy_metrics = calculate_model_accuracy(real_prices, predicted_prices)
                
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    rmse_val = accuracy_metrics['rmse']
                    if rmse_val == float('inf') or np.isnan(rmse_val):
                        st.metric("RMSE", "N/A")
                    else:
                        st.metric("RMSE", f"{rmse_val:,.2f} VND")
                        
                with col_metric2:
                    mae_val = accuracy_metrics['mae']
                    if mae_val == float('inf') or np.isnan(mae_val):
                        st.metric("MAE", "N/A")
                    else:
                        st.metric("MAE", f"{mae_val:,.2f} VND")
                        
                with col_metric3:
                    mape_val = accuracy_metrics['mape']
                    if mape_val == float('inf') or np.isnan(mape_val):
                        st.metric("MAPE", "N/A")
                    else:
                        st.metric("MAPE", f"{mape_val:.2f}%")
                        
                with col_metric4:
                    accuracy_val = accuracy_metrics['accuracy']
                    if np.isnan(accuracy_val):
                        st.metric("Độ chính xác", "N/A")
                    else:
                        st.metric("Độ chính xác", f"{accuracy_val:.2f}%")
                
                # Biểu đồ scatter cho accuracy
                fig_accuracy = create_accuracy_scatter_plot(real_prices, predicted_prices)
                if fig_accuracy:
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                else:
                    st.error("Không thể tính toán độ chính xác do dữ liệu không hợp lệ")
                
                
            except Exception as e:
                st.error(f"❌ Lỗi khi huấn luyện mô hình LSTM: {str(e)}")
                st.info("Hãy thử lại với khoảng thời gian dữ liệu dài hơn.")

else:
    # Hiển thị khi chưa tải dữ liệu
    st.info("👈 Hãy cấu hình thông tin ở sidebar và nhấn 'Bắt đầu phân tích' để bắt đầu!")
    
    # Hiển thị hướng dẫn sử dụng
    st.markdown("""
    ## 🚀 Hướng dẫn sử dụng
    
    1. **Nhập mã cổ phiếu** trong sidebar (VD: ACB, VCB, VHM, FPT, v.v.)
    2. **Chọn khoảng thời gian** phân tích
    3. **Cấu hình tham số LSTM** (lookback days, epochs)
    4. **Nhấn nút "Bắt đầu phân tích"** để xem kết quả
    
    ## 📊 Các tính năng chính:
    - **Dự đoán giá LSTM**: Sử dụng mạng neural LSTM để dự đoán giá
    - **Dự đoán tương lai**: Dự đoán giá cho 10 ngày kinh doanh tiếp theo
    - **Độ chính xác mô hình**: Đánh giá hiệu suất dự đoán
    - **Phân tích xu hướng**: Xác định hướng và cường độ xu hướng giá
    """)
    
    st.markdown("---")
    st.markdown("""
    ## ⚠️ Lưu ý quan trọng
    
    - **Không phải lời khuyên đầu tư**: Ứng dụng chỉ mang tính chất tham khảo
    - **Có rủi ro**: Thị trường chứng khoán luôn có rủi ro cao
    - **Tự nghiên cứu**: Luôn tự nghiên cứu kỹ trước khi đầu tư
    - **Tham khảo chuyên gia**: Nên tham khảo ý kiến chuyên gia tài chính
    """)
