import streamlit as st
from vnstock import Vnstock
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
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

# Hàm để tải và xử lý dữ liệu
@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(start=start_date.strftime('%Y-%m-%d'), 
                                end=end_date.strftime('%Y-%m-%d'), 
                                interval='1D')
        return df, None
    except Exception as e:
        return None, str(e)

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
    
    with col1:
        st.metric("Tổng số ngày", len(df))
    
    with col2:
        latest_price = df['close'].iloc[-1]
        st.metric("Giá đóng cửa mới nhất", f"{latest_price:,.0f} VND")
    
    with col3:
        price_change = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
        change_percent = (price_change / df['close'].iloc[-2] * 100) if len(df) > 1 else 0
        st.metric("Thay đổi hôm nay", f"{price_change:,.0f} VND", f"{change_percent:.2f}%")
    
    with col4:
        avg_volume = df['volume'].mean()
        st.metric("Khối lượng TB", f"{avg_volume:,.0f}")
    
    st.markdown("---")
    
    # Hiển thị dữ liệu mẫu
    with st.expander("📊 Xem dữ liệu chi tiết"):
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Kích thước dữ liệu:** {df.shape[0]} hàng, {df.shape[1]} cột")
    
    # Biểu đồ kết hợp (toàn bộ chiều rộng)
    st.subheader(f"📈 Phân tích tổng hợp {symbol}")
    fig_combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'Giá cổ phiếu {symbol}', 'Khối lượng giao dịch'),
        row_heights=[0.7, 0.3]
    )
    
    # Thêm biểu đồ nến
    fig_combined.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Thêm biểu đồ volume
    fig_combined.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig_combined.update_layout(
        title=f'Phân tích cổ phiếu {symbol}',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=False
    )
    
    fig_combined.update_xaxes(title_text="Ngày", row=2, col=1)
    fig_combined.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig_combined.update_yaxes(title_text="Khối lượng", row=2, col=1)
    
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
            
            # Hàm tạo dữ liệu cho LSTM
            @st.cache_data
            def prepare_lstm_data(data, lookback=60):
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data[['close']].values)
                
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i - lookback:i, 0])
                    y.append(scaled_data[i, 0])
                
                X, y = np.array(X), np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                
                return X, y, scaler
            
            # Hàm tạo và huấn luyện mô hình LSTM
            def create_lstm_model(X, y):
                model = Sequential()
                model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(64))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                return model
            
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
                            model.fit(X, y, epochs=1, batch_size=batch_size)
                            progress_bar.progress((epoch + 1) / epochs)
                        
                        status_text.text('Hoàn thành huấn luyện!')
                        
                        # Dự đoán
                        predicted = model.predict(X)
                        predicted_prices = scaler.inverse_transform(predicted)
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
                        prediction_df = pd.DataFrame({
                            'date': prediction_dates,
                            'actual': real_prices.flatten(),
                            'predicted': predicted_prices.flatten()
                        })
                        
                        # Hiển thị biểu đồ so sánh
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 So sánh Thực tế vs Dự đoán")
                            fig_comparison = go.Figure()
                            
                            fig_comparison.add_trace(go.Scatter(
                                x=prediction_df['date'],
                                y=prediction_df['actual'],
                                mode='lines',
                                name='Giá thực tế',
                                line=dict(color='blue')
                            ))
                            
                            fig_comparison.add_trace(go.Scatter(
                                x=prediction_df['date'],
                                y=prediction_df['predicted'],
                                mode='lines',
                                name='Giá dự đoán',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig_comparison.update_layout(
                                title='So sánh Giá thực tế vs Dự đoán',
                                xaxis_title='Ngày',
                                yaxis_title='Giá (VND)',
                                height=400
                            )
                            
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        with col2:
                            st.subheader("📈 Tín hiệu Mua/Bán")
                            
                            # Tạo tín hiệu mua/bán
                            signals = []
                            for i in range(1, len(predicted_prices)):
                                if predicted_prices[i] > predicted_prices[i - 1]:
                                    signals.append('Mua')
                                else:
                                    signals.append('Bán')
                            
                            # Hiển thị tín hiệu cuối cùng và phân tích trading
                            latest_signal = signals[-1] if signals else "Không xác định"
                            signal_color = "green" if latest_signal == "Mua" else "red"
                            
                            # Phân tích xu hướng và dự đoán điểm mua/bán
                            def analyze_trading_signals(predicted_prices, real_prices, dates):
                                trading_analysis = {}
                                
                                # Tính toán độ biến động giá
                                price_volatility = np.std(predicted_prices) / np.mean(predicted_prices) * 100
                                
                                # Tìm điểm thấp nhất và cao nhất trong dự đoán
                                min_price_idx = np.argmin(predicted_prices)
                                max_price_idx = np.argmax(predicted_prices)
                                
                                # Tính toán đà tăng/giảm
                                recent_trend = predicted_prices[-5:] if len(predicted_prices) >= 5 else predicted_prices
                                trend_direction = "Tăng" if recent_trend[-1] > recent_trend[0] else "Giảm"
                                trend_strength = abs((recent_trend[-1] - recent_trend[0]) / recent_trend[0] * 100)
                                
                                # Dự đoán điểm mua tối ưu (giá thấp + xu hướng tăng)
                                buy_opportunities = []
                                for i in range(1, len(predicted_prices) - 1):
                                    if (predicted_prices[i] < predicted_prices[i-1] and 
                                        predicted_prices[i] < predicted_prices[i+1]):
                                        profit_potential = (np.max(predicted_prices[i:]) - predicted_prices[i]) / predicted_prices[i] * 100
                                        if profit_potential > 2:  # Chỉ consider nếu có tiềm năng lãi > 2%
                                            # Xử lý an toàn cho date
                                            date_value = None
                                            if i < len(dates):
                                                try:
                                                    date_value = pd.to_datetime(dates[i])
                                                except:
                                                    date_value = None
                                            
                                            buy_opportunities.append({
                                                'index': i,
                                                'price': predicted_prices[i],
                                                'date': date_value,
                                                'profit_potential': profit_potential
                                            })
                                
                                # Sắp xếp theo tiềm năng lãi
                                buy_opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)
                                
                                # Dự đoán điểm bán tối ưu (giá cao + xu hướng giảm)
                                sell_opportunities = []
                                for i in range(1, len(predicted_prices) - 1):
                                    if (predicted_prices[i] > predicted_prices[i-1] and 
                                        predicted_prices[i] > predicted_prices[i+1]):
                                        price_drop = (predicted_prices[i] - np.min(predicted_prices[i:])) / predicted_prices[i] * 100
                                        if price_drop > 2:  # Chỉ consider nếu có khả năng giảm > 2%
                                            # Xử lý an toàn cho date
                                            date_value = None
                                            if i < len(dates):
                                                try:
                                                    date_value = pd.to_datetime(dates[i])
                                                except:
                                                    date_value = None
                                            
                                            sell_opportunities.append({
                                                'index': i,
                                                'price': predicted_prices[i],
                                                'date': date_value,
                                                'risk_level': price_drop
                                            })
                                
                                sell_opportunities.sort(key=lambda x: x['risk_level'], reverse=True)
                                
                                trading_analysis = {
                                    'volatility': price_volatility,
                                    'trend_direction': trend_direction,
                                    'trend_strength': trend_strength,
                                    'best_buy': buy_opportunities[0] if buy_opportunities else None,
                                    'best_sell': sell_opportunities[0] if sell_opportunities else None,
                                    'min_price_date': pd.to_datetime(dates[min_price_idx]) if min_price_idx < len(dates) else None,
                                    'max_price_date': pd.to_datetime(dates[max_price_idx]) if max_price_idx < len(dates) else None,
                                    'min_price': predicted_prices[min_price_idx],
                                    'max_price': predicted_prices[max_price_idx]
                                }
                                
                                return trading_analysis
                            
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
                            
                            # Hiển thị khuyến nghị mua/bán cụ thể
                            st.markdown("### 🎯 Khuyến nghị Trading")
                            
                            col_buy, col_sell = st.columns(2)
                            
                            with col_buy:
                                st.markdown("#### 💰 Điểm mua tối ưu")
                                if trading_info['best_buy']:
                                    buy_info = trading_info['best_buy']
                                    date_str = buy_info['date'].strftime('%d/%m/%Y') if buy_info['date'] and hasattr(buy_info['date'], 'strftime') else 'N/A'
                                    st.success(f"""
                                    **Giá mua đề xuất:** {buy_info['price']:,.0f} VND
                                    **Tiềm năng lãi:** {buy_info['profit_potential']:.1f}%
                                    **Ngày dự kiến:** {date_str}
                                    """)
                                else:
                                    st.info("Không tìm thấy điểm mua tối ưu trong dự đoán")
                                
                                # Hiển thị giá thấp nhất dự đoán
                                min_date_str = trading_info['min_price_date'].strftime('%d/%m/%Y') if trading_info['min_price_date'] and hasattr(trading_info['min_price_date'], 'strftime') else 'N/A'
                                st.info(f"""
                                **Giá thấp nhất dự đoán:** {trading_info['min_price']:,.0f} VND
                                **Ngày:** {min_date_str}
                                """)
                            
                            with col_sell:
                                st.markdown("#### 🎯 Điểm chốt lời tối ưu")
                                if trading_info['best_sell']:
                                    sell_info = trading_info['best_sell']
                                    date_str = sell_info['date'].strftime('%d/%m/%Y') if sell_info['date'] and hasattr(sell_info['date'], 'strftime') else 'N/A'
                                    st.warning(f"""
                                    **Giá bán đề xuất:** {sell_info['price']:,.0f} VND
                                    **Mức rủi ro:** {sell_info['risk_level']:.1f}%
                                    **Ngày dự kiến:** {date_str}
                                    """)
                                else:
                                    st.info("Không tìm thấy điểm bán tối ưu trong dự đoán")
                                
                                # Hiển thị giá cao nhất dự đoán
                                max_date_str = trading_info['max_price_date'].strftime('%d/%m/%Y') if trading_info['max_price_date'] and hasattr(trading_info['max_price_date'], 'strftime') else 'N/A'
                                st.info(f"""
                                **Giá cao nhất dự đoán:** {trading_info['max_price']:,.0f} VND
                                **Ngày:** {max_date_str}
                                """)
                            
                            # Tính toán và hiển thị lợi nhuận tiềm năng
                            if trading_info['best_buy'] and trading_info['best_sell']:
                                potential_profit = ((trading_info['best_sell']['price'] - trading_info['best_buy']['price']) / 
                                                 trading_info['best_buy']['price'] * 100)
                                
                                if potential_profit > 0:
                                    st.success(f"""
                                    ### 📈 Tiềm năng lợi nhuận: {potential_profit:.1f}%
                                    **Chiến lược:** Mua ở {trading_info['best_buy']['price']:,.0f} VND, bán ở {trading_info['best_sell']['price']:,.0f} VND
                                    """)
                                else:
                                    st.warning("⚠️ Không có cơ hội lợi nhuận rõ ràng trong khoảng thời gian dự đoán")
                            
                            # Thêm cảnh báo rủi ro
                            st.markdown("### ⚠️ Lưu ý quan trọng")
                            st.warning(f"""
                            - Độ biến động hiện tại: **{trading_info['volatility']:.1f}%** {'(Cao)' if trading_info['volatility'] > 5 else '(Thấp)'}
                            - Xu hướng ngắn hạn: **{trading_info['trend_direction']}** với cường độ {trading_info['trend_strength']:.1f}%
                            - Đây chỉ là dự đoán dựa trên AI, không phải lời khuyên đầu tư
                            - Luôn đặt stop-loss và take-profit phù hợp
                            - Không đầu tư quá 5-10% tổng tài sản vào một cổ phiếu
                            """)
                            
                            # Hiển thị thống kê độ chính xác
                            # Đảm bảo dữ liệu có cùng shape và loại bỏ giá trị NaN
                            real_flat = real_prices.flatten()
                            pred_flat = predicted_prices.flatten()
                            
                            # Loại bỏ các giá trị NaN hoặc inf
                            valid_indices = ~(np.isnan(real_flat) | np.isnan(pred_flat) | 
                                            np.isinf(real_flat) | np.isinf(pred_flat))
                            real_clean = real_flat[valid_indices]
                            pred_clean = pred_flat[valid_indices]
                            
                            if len(real_clean) > 0:
                                # Tính toán metrics
                                mse = np.mean((real_clean - pred_clean) ** 2)
                                rmse = np.sqrt(mse)
                                mae = np.mean(np.abs(real_clean - pred_clean))
                                
                                # Tính percentage accuracy (MAPE - Mean Absolute Percentage Error)
                                mape = np.mean(np.abs((real_clean - pred_clean) / real_clean)) * 100
                                
                                st.write("**Thống kê độ chính xác:**")
                                st.write(f"- RMSE: {rmse:,.2f} VND")
                                st.write(f"- MAE: {mae:,.2f} VND")
                                st.write(f"- MAPE: {mape:.2f}%")
                                st.write(f"- Độ chính xác: {100 - mape:.2f}%")
                            else:
                                st.write("**Thống kê độ chính xác:**")
                                st.write("- Không thể tính toán do dữ liệu không hợp lệ")
                            
                            # Hiển thị distribution của tín hiệu
                            signal_counts = pd.Series(signals).value_counts()
                            fig_signals = px.pie(
                                values=signal_counts.values,
                                names=signal_counts.index,
                                title="Phân bổ tín hiệu Mua/Bán"
                            )
                            st.plotly_chart(fig_signals, use_container_width=True)
                        
                        # Hiển thị bảng dự đoán mới nhất
                        st.subheader("📋 Dự đoán 10 ngày gần nhất")
                        recent_predictions = prediction_df.tail(10).copy()
                        recent_predictions['difference'] = recent_predictions['predicted'] - recent_predictions['actual']
                        
                        # Tính độ chính xác với xử lý trường hợp chia cho 0
                        def calculate_accuracy(actual, predicted):
                            if actual == 0:
                                return 0
                            return max(0, (1 - abs(predicted - actual) / abs(actual)) * 100)
                        
                        recent_predictions['accuracy'] = recent_predictions.apply(
                            lambda row: calculate_accuracy(row['actual'], row['predicted']), axis=1
                        )
                        
                        # Sao lưu dữ liệu số cho tính toán
                        actual_backup = recent_predictions['actual'].copy()
                        predicted_backup = recent_predictions['predicted'].copy()
                        difference_backup = recent_predictions['difference'].copy()
                        accuracy_backup = recent_predictions['accuracy'].copy()
                        
                        # Format hiển thị
                        recent_predictions['actual'] = actual_backup.apply(lambda x: f"{x:,.0f}")
                        recent_predictions['predicted'] = predicted_backup.apply(lambda x: f"{x:,.0f}")
                        recent_predictions['difference'] = difference_backup.apply(lambda x: f"{x:+,.0f}")
                        recent_predictions['accuracy'] = accuracy_backup.apply(lambda x: f"{x:.1f}%")
                        
                        recent_predictions.columns = ['Ngày', 'Giá thực tế (VND)', 'Giá dự đoán (VND)', 'Chênh lệch (VND)', 'Độ chính xác']
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
