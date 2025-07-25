import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import custom utilities
from utils.data_processing import (
    load_stock_data, get_basic_stats, prepare_prediction_dataframe, 
    format_prediction_table, load_news_data, format_news_for_display
)
from utils.openai_summary import (
    get_openai_news_summary, format_articles_for_summary
)
from utils.lstm_model import (
    prepare_lstm_data, create_lstm_model, train_lstm_model, 
    make_predictions, calculate_model_accuracy, predict_future_prices, KERAS_AVAILABLE
)
from utils.trading_analysis import (
    analyze_trading_signals, generate_trading_signals, 
    calculate_trend_strength_over_time, calculate_signal_distribution,
    calculate_potential_profit
)
from utils.plotting import (
    create_combined_chart, create_comparison_chart, create_trading_signals_chart,
    create_signals_pie_chart, create_trend_strength_chart, create_accuracy_scatter_plot,
    create_future_prediction_chart
)
# Get today's date as base reference
from datetime import datetime, timedelta

# Check if Keras is available
if not KERAS_AVAILABLE:
    st.warning("Keras/TensorFlow chưa được cài đặt. Chức năng dự đoán LSTM sẽ không khả dụng.")

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân tích Cổ phiếu Việt Nam",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề chính
st.title("📈 Phân tích Cổ phiếu Việt Nam")
st.markdown("---")

# Sidebar để cấu hình
st.sidebar.header("Cấu hình")
symbol = st.sidebar.text_input("Mã cổ phiếu", value="ACB", help="Nhập mã cổ phiếu (VD: ACB, VCB, VHM)")
start_date = st.sidebar.date_input("Ngày bắt đầu", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("Ngày kết thúc", value=pd.Timestamp.today())

# OpenAI API Configuration
st.sidebar.subheader("🤖 Cấu hình OpenAI")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    help="Nhập API key để sử dụng tính năng tóm tắt tin tức bằng AI",
    placeholder="sk-..."
)

# News summary configuration
enable_ai_summary = st.sidebar.checkbox(
    "Tạo tóm tắt tin tức bằng AI", 
    value=False,
    help="Sử dụng OpenAI để phân tích và tóm tắt tin tức (cần API key)"
)

# Nút để tải dữ liệu
if st.sidebar.button("Tải dữ liệu", type="primary"):
    st.session_state.load_data = True

# Kiểm tra và tải dữ liệu
if 'load_data' not in st.session_state:
    st.session_state.load_data = True

if st.session_state.load_data:
    with st.spinner(f"Đang tải dữ liệu cho {symbol}..."):
        df, error = load_stock_data(symbol, start_date, end_date)
    
    if error:
        st.error(f"Lỗi khi tải dữ liệu: {error}")
        st.stop()
    
    if df is None or df.empty:
        st.warning("Không có dữ liệu cho mã cổ phiếu này trong khoảng thời gian đã chọn.")
        st.stop()
    
    # Hiển thị thông tin cơ bản
    col1, col2, col3, col4 = st.columns(4)
    
    # Get basic statistics
    stats = get_basic_stats(df)
    
    with col1:
        st.metric("Tổng số ngày", stats['total_days'])
    
    with col2:
        st.metric("Giá đóng cửa mới nhất", f"{stats['latest_price']:,.0f} VND")
    
    with col3:
        st.metric("Thay đổi hôm nay", f"{stats['price_change']:,.0f} VND", f"{stats['change_percent']:.2f}%")
    
    with col4:
        st.metric("Khối lượng TB", f"{stats['avg_volume']:,.0f}")
    
    st.markdown("---")

    # Biểu đồ kết hợp (toàn bộ chiều rộng)
    st.subheader(f"📈 Phân tích tổng hợp {symbol}")
    fig_combined = create_combined_chart(df, symbol)
    st.plotly_chart(fig_combined, use_container_width=True)
    st.subheader(f"📰 Tin tức về {symbol}")
    
    # Load news data for the selected symbol
    with st.spinner(f"Đang tải tin tức cho {symbol}..."):
        news_df = load_news_data(symbol)
    
    if not news_df.empty:
        # AI Summary Section (before showing individual news)
        if enable_ai_summary and openai_api_key:
            st.markdown("---")
            st.subheader("🤖 Phân tích AI - Tóm tắt 10 tin tức mới nhất")
            
            if st.button("🔄 Tạo phân tích AI", type="primary", help="Sử dụng OpenAI để phân tích tác động tin tức lên giá cổ phiếu"):
                with st.spinner("🤖 Đang phân tích tin tức bằng AI..."):
                    try:
                        # Get 10 latest articles for AI analysis
                        articles_for_ai = format_articles_for_summary(news_df, max_articles=10)
                        
                        if articles_for_ai:
                            # Get AI summary
                            ai_summary = get_openai_news_summary(articles_for_ai, symbol, openai_api_key)
                            
                            if ai_summary:
                                # Display AI analysis in an attractive format
                                st.markdown("### 📊 Báo cáo Phân tích AI")
                                
                                # Create a styled container for AI analysis
                                with st.container():
                                    st.markdown(
                                        """
                                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Display the AI analysis
                                    st.markdown(ai_summary)
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Add disclaimer
                                st.warning("""
                                ⚠️ **Lưu ý quan trọng:** 
                                - Đây là phân tích được tạo bởi AI dựa trên dữ liệu tin tức có sẵn
                                - Không được coi là lời khuyên đầu tư tài chính
                                - Luôn thực hiện nghiên cứu độc lập trước khi đưa ra quyết định đầu tư
                                - Kết quả có thể thay đổi tùy theo diễn biến thị trường
                                """)
                                
                            else:
                                st.error("Không thể tạo phân tích AI. Vui lòng kiểm tra API key hoặc thử lại.")
                        else:
                            st.warning("Không có đủ tin tức để phân tích.")
                            
                    except Exception as e:
                        st.error(f"Lỗi khi tạo phân tích AI: {str(e)}")
            
            st.markdown("---")
        
        elif enable_ai_summary and not openai_api_key:
            st.warning("🔑 Vui lòng nhập OpenAI API Key trong sidebar để sử dụng tính năng phân tích AI.")
            st.markdown("---")
        
        # Display news statistics
        col_news1, col_news2, col_news3 = st.columns(3)
        
        with col_news1:
            st.metric("Tổng số tin tức", len(news_df))
        
        # Filter data
        filtered_news = news_df.copy()
        formatted_news = format_news_for_display(filtered_news, 15)
        
        if not formatted_news.empty:
            for i, (idx, row) in enumerate(formatted_news.iterrows()):
                with st.expander(f"📄 {row['Tiêu đề'][:80]}..." if len(row['Tiêu đề']) > 80 else f"📄 {row['Tiêu đề']}"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Ngày đăng:** {row['Ngày đăng']}")
                        st.write(f"**Danh mục:** {row['Danh mục']}")
                    
                    with col_info2:
                        st.write(f"**Chuyên mục:** {row['Chuyên mục']}")
                        st.write(f"**Độ tin cậy:** {row['Độ tin cậy']}")
                    
                    # Get description if available from the original filtered news
                    if i < len(filtered_news) and 'description' in filtered_news.columns:
                        original_row = filtered_news.iloc[i]
                        if pd.notna(original_row['description']):
                            st.write(f"**Mô tả:** {original_row['description']}")
                    
                    # Link to full article
                    if row['Link'] and row['Link'] != '#':
                        st.markdown(f"🔗 [Đọc bài viết đầy đủ]({row['Link']})")

        else:
            st.info("Không có tin tức nào để hiển thị sau khi lọc.")
    else:
        st.info(f"Không tìm thấy tin tức cho mã chứng khoán {symbol}.")
        st.markdown("""
        **Lưu ý:** Tin tức chỉ khả dụng cho các mã chứng khoán có trong hệ thống dữ liệu. 
        Hãy thử với các mã phổ biến như: ACB, VCB, FPT, VHM, HPG, VIC, v.v.
        """)
    
    # LSTM Price Prediction Section
    st.markdown("---")
    st.subheader("🤖 Dự đoán giá sử dụng LSTM")
    
    if KERAS_AVAILABLE:
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
                    lookback = 60
                    if len(df) < lookback + 30:
                        st.warning(f"Cần ít nhất {lookback + 30} ngày dữ liệu để huấn luyện mô hình LSTM. Hiện tại chỉ có {len(df)} ngày.")
                    else:
                        X, y, scaler = prepare_lstm_data(df, lookback)
                        
                        # Tạo và huấn luyện mô hình
                        model = create_lstm_model(X, y)
                        
                        # Progress bar cho quá trình huấn luyện
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        epochs = 20
                        batch_size = 32
                        
                        # Huấn luyện với callback để cập nhật progress
                        for epoch in range(epochs):
                            status_text.text(f'Epoch {epoch + 1}/{epochs}')
                            train_lstm_model(model, X, y, epochs=1, batch_size=batch_size, verbose=0)
                            progress_bar.progress((epoch + 1) / epochs)
                        
                        status_text.text('Hoàn thành huấn luyện!')
                        
                        # Dự đoán
                        predicted_prices = make_predictions(model, X, scaler)
                        
                        # Sử dụng dữ liệu thực tế từ DataFrame thay vì từ y (scaled)
                        # y tương ứng với dữ liệu từ ngày lookback trở đi
                        real_prices_original = df['close'].iloc[lookback:].values.reshape(-1, 1)
                        
                        # Đảm bảo kích thước khớp nhau
                        min_length = min(len(predicted_prices), len(real_prices_original))
                        predicted_prices = predicted_prices[:min_length]
                        real_prices = real_prices_original[:min_length]
                        
                        # Debug prints to check data
                        print(f"Predicted prices shape: {predicted_prices.shape}")
                        print(f"Real prices shape: {real_prices.shape}")
                        print(f"Sample predicted: {predicted_prices[:5].flatten()}")
                        print(f"Sample real: {real_prices[:5].flatten()}")
                        print(f"Data length check - X: {len(X)}, predicted: {len(predicted_prices)}, real: {len(real_prices)}")
                        
                        # Tạo index cho dữ liệu dự đoán (bỏ qua lookback ngày đầu)
                        prediction_dates = df.index[lookback:lookback+min_length]
                        
                        # Tạo DataFrame cho dữ liệu dự đoán
                        prediction_df = prepare_prediction_dataframe(prediction_dates, real_prices, predicted_prices)
                        
                        # === SECTION: FUTURE PREDICTIONS ===
                        st.markdown("---")
                        st.subheader("🔮 Dự đoán Giá 10 Ngày Tới")
                        
                        # Add checkbox to enable future prediction
                        if st.checkbox("Hiển thị dự đoán 10 ngày tới", value=True, help="Dự đoán giá cho 10 ngày kinh doanh tiếp theo"):
                            with st.spinner("Đang dự đoán giá cho 10 ngày tới..."):
                                try:
                                    # Predict future prices
                                    future_prices = predict_future_prices(model, df, scaler, lookback=lookback, days_ahead=10)
                                    
                                    # Generate future dates (business days only)
                                    last_date = df.index[-1]
                                    today = datetime.now().date()
                                    
                                    # Convert last_date to proper format
                                    if hasattr(last_date, 'date'):
                                        last_date_clean = last_date.date()
                                    else:
                                        # Parse string date if needed
                                        try:
                                            last_date_clean = pd.to_datetime(str(last_date)).date()
                                        except:
                                            # Fallback to today if parsing fails
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
                                            "Giá trung bình dự đoán (10 ngày)",
                                            f"{avg_future_price:,.0f} VND",
                                            f"{price_change_pct:+.2f}%"
                                        )
                                        
                                        max_future_price = np.max(future_prices)
                                        min_future_price = np.min(future_prices)
                                        volatility_future = ((max_future_price - min_future_price) / min_future_price) * 100
                                        
                                        st.metric(
                                            "Biên độ biến động dự đoán",
                                            f"{volatility_future:.2f}%",
                                            f"từ {min_future_price:,.0f} đến {max_future_price:,.0f} VND"
                                        )
                                    
                                    with col_analysis2:
                                        # Trend analysis for future predictions
                                        if future_prices[-1] > future_prices[0]:
                                            trend_direction = "📈 Tăng"
                                            trend_color = "green"
                                        else:
                                            trend_direction = "📉 Giảm"
                                            trend_color = "red"
                                        
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
                                    
                                    st.warning("""
                                    **⚠️ Lưu ý quan trọng về dự đoán tương lai:**
                                    - Dự đoán càng xa thì độ tin cậy càng giảm
                                    - Các sự kiện bất ngờ có thể thay đổi hoàn toàn xu hướng giá
                                    - Luôn kết hợp với phân tích cơ bản và tin tức thị trường
                                    - Không nên dựa hoàn toàn vào dự đoán AI để đầu tư
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"Lỗi khi dự đoán giá tương lai: {str(e)}")
                                    st.info("Hãy thử lại hoặc kiểm tra dữ liệu đầu vào.")

                        st.subheader("📊 So sánh Thực tế vs Dự đoán")
                        fig_comparison = create_comparison_chart(prediction_df)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        signals = generate_trading_signals(predicted_prices)
                        # Hiển thị tín hiệu cuối cùng và phân tích trading
                        latest_signal = signals[-1] if signals else "Không xác định"
                        signal_color = "green" if latest_signal == "Mua" else "red"
                        
                        # Thực hiện phân tích trading
                        trading_info = analyze_trading_signals(
                            predicted_prices.flatten(), 
                            real_prices.flatten(), 
                            prediction_dates
                        )
                        print(f"Trading info:", trading_info)

                        # === SECTION: MODEL ACCURACY ===
                        st.markdown("---")
                        st.subheader("🎯 Độ chính xác Mô hình")
                        
                        # Tính toán độ chính xác
                        accuracy_metrics = calculate_model_accuracy(real_prices, predicted_prices)
                        
                        # Debug print accuracy metrics
                        print(f"Accuracy metrics: {accuracy_metrics}")
                        
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        
                        with col_metric1:
                            rmse_val = accuracy_metrics['rmse']
                            print(f"RMSE value: ", rmse_val)
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
                    st.error(f"Lỗi khi huấn luyện mô hình LSTM: {str(e)}")
                    st.info("Hãy thử lại với khoảng thời gian dữ liệu dài hơn.")
    else:
        st.info("Để sử dụng chức năng dự đoán LSTM, vui lòng cài đặt TensorFlow/Keras:")
        st.code("pip install tensorflow", language="bash")

else:
    # Hiển thị khi chưa tải dữ liệu
    st.info("👈 Hãy cấu hình thông tin ở sidebar và nhấn 'Tải dữ liệu' để bắt đầu phân tích!")
    
    # Hiển thị hướng dẫn sử dụng
    st.markdown("""
    ## 🚀 Hướng dẫn sử dụng
    
    1. **Nhập mã cổ phiếu** trong sidebar (VD: ACB, VCB, VHM, FPT, v.v.)
    2. **Chọn khoảng thời gian** phân tích
    3. **Nhấn nút "Tải dữ liệu"** để xem biểu đồ
    
    ## 📊 Các biểu đồ sẽ hiển thị:
    - **Biểu đồ nến (Candlestick)**: Hiển thị giá mở, đóng, cao, thấp
    - **Biểu đồ khối lượng**: Hiển thị khối lượng giao dịch
    - **Biểu đồ tổng hợp**: Kết hợp giá và khối lượng
    - **Biểu đồ xu hướng**: Đường giá đóng cửa
    - **Thống kê mô tả**: Các chỉ số thống kê chi tiết
    """)
