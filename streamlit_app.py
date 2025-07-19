import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import custom utilities
from utils.data_processing import load_stock_data, get_basic_stats, prepare_prediction_dataframe, format_prediction_table
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
    
    # Hiển thị dữ liệu mẫu
    with st.expander("📊 Xem dữ liệu chi tiết"):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Kích thước dữ liệu:** {df.shape[0]} hàng, {df.shape[1]} cột")
    
    # Biểu đồ kết hợp (toàn bộ chiều rộng)
    st.subheader(f"📈 Phân tích tổng hợp {symbol}")
    fig_combined = create_combined_chart(df, symbol)
    st.plotly_chart(fig_combined, use_container_width=True)

    # Phân tích thống kê
    st.markdown("---")
    st.subheader("📋 Thống kê mô tả")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Thống kê giá:**")
        price_stats = df[['open', 'high', 'low', 'close']].describe()
        st.dataframe(price_stats, use_container_width=True)
    
    with col2:
        st.write("**Thống kê khối lượng:**")
        volume_stats = df[['volume']].describe()
        st.dataframe(volume_stats, use_container_width=True)
    
    # LSTM Price Prediction Section
    st.markdown("---")
    st.subheader("🤖 Dự đoán giá sử dụng LSTM")
    
    if KERAS_AVAILABLE:
        # Thêm checkbox để cho phép người dùng chọn có chạy dự đoán hay không
        if st.checkbox("Chạy mô hình dự đoán LSTM", value=False, help="Có thể mất vài phút để huấn luyện mô hình"):
            
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
                        real_prices = scaler.inverse_transform(y.reshape(-1, 1))
                        
                        # Debug: Kiểm tra dữ liệu
                        st.write(f"**Debug info:**")
                        st.write(f"- Shape predicted_prices: {predicted_prices.shape}")
                        st.write(f"- Shape real_prices: {real_prices.shape}")
                        st.write(f"- Min predicted: {np.min(predicted_prices):,.2f}")
                        st.write(f"- Max predicted: {np.max(predicted_prices):,.2f}")
                        st.write(f"- Min real: {np.min(real_prices):,.2f}")
                        st.write(f"- Max real: {np.max(real_prices):,.2f}")
                        
                        # Tạo index cho dữ liệu dự đoán (bỏ qua lookback ngày đầu)
                        prediction_dates = df.index[lookback:]
                        
                        # Tạo DataFrame cho dữ liệu dự đoán
                        prediction_df = prepare_prediction_dataframe(prediction_dates, real_prices, predicted_prices)
                        
                        # Hiển thị biểu đồ so sánh
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 So sánh Thực tế vs Dự đoán")
                            fig_comparison = create_comparison_chart(prediction_df)
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        with col2:
                            st.subheader("📈 Tín hiệu Mua/Bán")
                            
                            # Tạo tín hiệu mua/bán
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
                            
                            st.markdown(f"""
                            <div style="background-color: {signal_color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h3>Tín hiệu mới nhất: {latest_signal}</h3>
                                <p><strong>Xu hướng:</strong> {trading_info['trend_direction']} ({trading_info['trend_strength']:.1f}%)</p>
                                <p><strong>Độ biến động:</strong> {trading_info['volatility']:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # === SECTION: TRADING SIGNALS ANALYSIS ===
                        st.markdown("---")
                        st.subheader("📊 Phân tích Tín hiệu Trading")
                        
                        # Tạo biểu đồ tín hiệu mua/bán trên giá
                        fig_trading = create_trading_signals_chart(prediction_df, signals, trading_info)
                        st.plotly_chart(fig_trading, use_container_width=True)
                        
                        # Hiển thị distribution của tín hiệu
                        col_pie, col_bar = st.columns(2)
                        
                        with col_pie:
                            fig_signals = create_signals_pie_chart(signals)
                            if fig_signals:
                                st.plotly_chart(fig_signals, use_container_width=True)
                            else:
                                st.info("Không có tín hiệu để hiển thị")
                        
                        with col_bar:
                            # Tạo biểu đồ trend strength theo thời gian
                            trend_data = calculate_trend_strength_over_time(predicted_prices)
                            fig_trend = create_trend_strength_chart(trend_data, prediction_dates)
                            if fig_trend:
                                st.plotly_chart(fig_trend, use_container_width=True)
                            else:
                                st.info("Không đủ dữ liệu để phân tích xu hướng")
                        
                        # === SECTION: TRADING RECOMMENDATIONS ===
                        st.markdown("---")
                        st.subheader("🎯 Khuyến nghị Trading")
                        
                        col_buy, col_sell = st.columns(2)
                        
                        with col_buy:
                            st.markdown("#### 💰 Điểm mua tối ưu")
                            if trading_info['best_buy']:
                                buy_info = trading_info['best_buy']
                                # Format ngày an toàn
                                date_str = 'N/A'
                                if buy_info['date']:
                                    try:
                                        if hasattr(buy_info['date'], 'strftime'):
                                            date_str = buy_info['date'].strftime('%d/%m/%Y')
                                        else:
                                            date_str = str(buy_info['date'])[:10]
                                    except:
                                        date_str = 'N/A'
                                
                                st.success(f"""
                                **Giá mua đề xuất:** {buy_info['price']:,.0f} VND
                                **Tiềm năng lãi:** {buy_info['profit_potential']:.1f}%
                                **Ngày dự kiến:** {date_str}
                                """)
                            else:
                                st.info("Không tìm thấy điểm mua tối ưu trong dự đoán")
                            
                            # Hiển thị giá thấp nhất dự đoán
                            min_date_str = 'N/A'
                            if trading_info['min_price_date']:
                                try:
                                    if hasattr(trading_info['min_price_date'], 'strftime'):
                                        min_date_str = trading_info['min_price_date'].strftime('%d/%m/%Y')
                                    else:
                                        min_date_str = str(trading_info['min_price_date'])[:10]
                                except:
                                    min_date_str = 'N/A'
                            
                            st.info(f"""
                            **Giá thấp nhất dự đoán:** {trading_info['min_price']:,.0f} VND
                            **Ngày dự kiến:** {min_date_str}
                            """)
                        
                        with col_sell:
                            st.markdown("#### 🎯 Điểm chốt lời tối ưu")
                            if trading_info['best_sell']:
                                sell_info = trading_info['best_sell']
                                # Format ngày an toàn
                                date_str = 'N/A'
                                if sell_info['date']:
                                    try:
                                        if hasattr(sell_info['date'], 'strftime'):
                                            date_str = sell_info['date'].strftime('%d/%m/%Y')
                                        else:
                                            date_str = str(sell_info['date'])[:10]
                                    except:
                                        date_str = 'N/A'
                                
                                st.warning(f"""
                                **Giá bán đề xuất:** {sell_info['price']:,.0f} VND
                                **Mức rủi ro:** {sell_info['risk_level']:.1f}%
                                **Ngày dự kiến:** {date_str}
                                """)
                            else:
                                st.info("Không tìm thấy điểm bán tối ưu trong dự đoán")
                            
                            # Hiển thị giá cao nhất dự đoán
                            max_date_str = 'N/A'
                            if trading_info['max_price_date']:
                                try:
                                    if hasattr(trading_info['max_price_date'], 'strftime'):
                                        max_date_str = trading_info['max_price_date'].strftime('%d/%m/%Y')
                                    else:
                                        max_date_str = str(trading_info['max_price_date'])[:10]
                                except:
                                    max_date_str = 'N/A'
                            
                            st.info(f"""
                            **Giá cao nhất dự đoán:** {trading_info['max_price']:,.0f} VND
                            **Ngày dự kiến:** {max_date_str}
                            """)
                        
                        # Tính toán và hiển thị lợi nhuận tiềm năng
                        profit_analysis = calculate_potential_profit(trading_info.get('best_buy'), trading_info.get('best_sell'))
                        
                        if profit_analysis['is_profitable']:
                            st.success(f"""
                            ### 📈 Tiềm năng lợi nhuận: {profit_analysis['profit_percentage']:.1f}%
                            **Chiến lược:** Mua ở {trading_info['best_buy']['price']:,.0f} VND, bán ở {trading_info['best_sell']['price']:,.0f} VND
                            **Lãi dự kiến:** {profit_analysis['profit_per_share']:,.0f} VND/cổ phiếu
                            """)
                        else:
                            # Hiển thị thông tin dự đoán chung
                            try:
                                min_price = float(trading_info['min_price'])
                                max_price = float(trading_info['max_price'])
                                if min_price > 0:
                                    general_profit = ((max_price - min_price) / min_price * 100)
                                    st.info(f"""
                                    ### 📊 Biên độ giá dự đoán: {general_profit:.1f}%
                                    **Từ:** {min_price:,.0f} VND **đến** {max_price:,.0f} VND
                                    """)
                            except:
                                st.info("📊 Đang phân tích dữ liệu để đưa ra khuyến nghị...")
                        
                        # Thêm thông tin về thời gian dự đoán
                        st.markdown("---")
                        current_date = pd.Timestamp.today().strftime('%d/%m/%Y')
                        st.info(f"""
                        📅 **Thông tin dự đoán:**
                        - Ngày hiện tại: {current_date}
                        - Khung thời gian dự đoán: 1-30 ngày tới
                        - Dựa trên mô hình LSTM và dữ liệu lịch sử
                        """)
                        
                        # === SECTION: RISK ANALYSIS ===
                        st.markdown("---")
                        st.subheader("⚠️ Phân tích Rủi ro")
                        
                        col_risk1, col_risk2 = st.columns(2)
                        
                        with col_risk1:
                            st.markdown("#### 📊 Thống kê Rủi ro")
                            
                            # Tính toán signal_counts an toàn
                            signal_distribution = calculate_signal_distribution(signals)
                            
                            st.warning(f"""
                            - **Độ biến động:** {trading_info['volatility']:.1f}% {'(Cao)' if trading_info['volatility'] > 5 else '(Thấp)'}
                            - **Xu hướng ngắn hạn:** {trading_info['trend_direction']} ({trading_info['trend_strength']:.1f}%)
                            - **Tín hiệu Mua:** {signal_distribution['Mua']} lần
                            - **Tín hiệu Bán:** {signal_distribution['Bán']} lần
                            """)
                        
                        with col_risk2:
                            st.markdown("#### 🛡️ Khuyến nghị An toàn")
                            st.info("""
                            - Đặt stop-loss ở -5% từ giá mua
                            - Đặt take-profit ở +10% từ giá mua
                            - Chỉ đầu tư 5-10% tổng tài sản
                            - Theo dõi tin tức thị trường hàng ngày
                            """)
                        
                        st.warning("""
                        ⚠️ **LỜI CẢNH BÁO QUAN TRỌNG**
                        - Đây chỉ là dự đoán dựa trên AI, không phải lời khuyên đầu tư
                        - Thị trường chứng khoán có rủi ro cao, có thể mất toàn bộ số tiền đầu tư
                        - Luôn tự nghiên cứu kỹ lưỡng trước khi đưa ra quyết định đầu tư
                        - Hãy tham khảo ý kiến của chuyên gia tài chính trước khi đầu tư
                        """)
                        
                        # === SECTION: MODEL ACCURACY ===
                        st.markdown("---")
                        st.subheader("🎯 Độ chính xác Mô hình")
                        
                        # Tính toán độ chính xác
                        accuracy_metrics = calculate_model_accuracy(real_prices, predicted_prices)
                        
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        
                        with col_metric1:
                            st.metric("RMSE", f"{accuracy_metrics['rmse']:,.0f} VND")
                        with col_metric2:
                            st.metric("MAE", f"{accuracy_metrics['mae']:,.0f} VND")
                        with col_metric3:
                            st.metric("MAPE", f"{accuracy_metrics['mape']:.2f}%")
                        with col_metric4:
                            st.metric("Độ chính xác", f"{accuracy_metrics['accuracy']:.2f}%")
                        
                        # Biểu đồ scatter cho accuracy
                        fig_accuracy = create_accuracy_scatter_plot(real_prices, predicted_prices)
                        if fig_accuracy:
                            st.plotly_chart(fig_accuracy, use_container_width=True)
                        else:
                            st.error("Không thể tính toán độ chính xác do dữ liệu không hợp lệ")
                        
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
                                    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')
                                    
                                    # Create future prediction chart
                                    # Show last 30 days of historical data for context
                                    recent_data = df.tail(30)
                                    fig_future = create_future_prediction_chart(recent_data, future_prices, future_dates, symbol)
                                    st.plotly_chart(fig_future, use_container_width=True)
                                    
                                    # Display future predictions in a table
                                    future_df = pd.DataFrame({
                                        'Ngày': future_dates.strftime('%d/%m/%Y'),
                                        'Dự đoán giá đóng cửa (VND)': [f"{price:,.0f}" for price in future_prices],
                                        'Thay đổi từ hôm nay (%)': [f"{((price - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100):+.2f}%" for price in future_prices]
                                    })
                                    
                                    st.markdown("##### 📊 Bảng Dự đoán Chi tiết")
                                    st.dataframe(future_df, use_container_width=True, hide_index=True)
                                    
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
                        
                        # Hiển thị bảng dự đoán mới nhất
                        st.subheader("📋 Dự đoán 10 ngày gần nhất")
                        recent_predictions = format_prediction_table(prediction_df)
                        st.dataframe(recent_predictions, use_container_width=True)
                        
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
