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
                                
                                # Flatten predicted_prices để đảm bảo là 1D array
                                pred_flat = predicted_prices.flatten()
                                
                                # Tính toán độ biến động giá
                                price_volatility = np.std(pred_flat) / np.mean(pred_flat) * 100
                                
                                # Tìm điểm thấp nhất và cao nhất trong dự đoán
                                min_price_idx = np.argmin(pred_flat)
                                max_price_idx = np.argmax(pred_flat)
                                
                                # Tính toán đà tăng/giảm
                                recent_trend = pred_flat[-5:] if len(pred_flat) >= 5 else pred_flat
                                trend_direction = "Tăng" if recent_trend[-1] > recent_trend[0] else "Giảm"
                                trend_strength = abs((recent_trend[-1] - recent_trend[0]) / recent_trend[0] * 100)
                                
                                # Tạo future dates tính từ ngày hiện tại
                                today = pd.Timestamp.today()
                                
                                # Dự đoán điểm mua tối ưu trong tương lai gần (30 ngày tới)
                                buy_opportunities = []
                                for i in range(1, min(len(pred_flat) - 1, 20)):  # Chỉ xem xét 20 điểm gần nhất
                                    if (pred_flat[i] < pred_flat[i-1] and 
                                        pred_flat[i] < pred_flat[i+1]):
                                        profit_potential = (np.max(pred_flat[i:]) - pred_flat[i]) / pred_flat[i] * 100
                                        if profit_potential > 2:
                                            # Tính ngày dự kiến trong tương lai - sử dụng datetime.timedelta
                                            days_ahead = int(i)
                                            future_date = today + pd.DateOffset(days=days_ahead)
                                            
                                            buy_opportunities.append({
                                                'index': i,
                                                'price': float(pred_flat[i]),
                                                'date': future_date,
                                                'profit_potential': profit_potential
                                            })
                                
                                # Nếu không có cơ hội mua từ dự đoán, tạo dựa trên xu hướng
                                if not buy_opportunities and len(pred_flat) > 0:
                                    # Lấy giá thấp nhất gần đây làm điểm mua
                                    recent_prices = pred_flat[-10:] if len(pred_flat) >= 10 else pred_flat
                                    min_recent_idx = np.argmin(recent_prices)
                                    min_price = recent_prices[min_recent_idx]
                                    days_ahead = int(min_recent_idx + 1)
                                    future_date = today + pd.DateOffset(days=days_ahead)
                                    
                                    buy_opportunities.append({
                                        'index': min_recent_idx,
                                        'price': float(min_price),
                                        'date': future_date,
                                        'profit_potential': 15.0  # Giả định 15% tiềm năng lãi
                                    })
                                
                                # Sắp xếp theo tiềm năng lãi
                                buy_opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)
                                
                                # Dự đoán điểm bán tối ưu trong tương lai
                                sell_opportunities = []
                                for i in range(1, min(len(pred_flat) - 1, 20)):
                                    if (pred_flat[i] > pred_flat[i-1] and 
                                        pred_flat[i] > pred_flat[i+1]):
                                        price_drop = (pred_flat[i] - np.min(pred_flat[i:])) / pred_flat[i] * 100
                                        if price_drop > 2:
                                            days_ahead = int(i)
                                            future_date = today + pd.DateOffset(days=days_ahead)
                                            
                                            sell_opportunities.append({
                                                'index': i,
                                                'price': float(pred_flat[i]),
                                                'date': future_date,
                                                'risk_level': price_drop
                                            })
                                
                                # Nếu không có cơ hội bán từ dự đoán, tạo dựa trên xu hướng
                                if not sell_opportunities and len(pred_flat) > 0:
                                    # Lấy giá cao nhất gần đây làm điểm bán
                                    recent_prices = pred_flat[-10:] if len(pred_flat) >= 10 else pred_flat
                                    max_recent_idx = np.argmax(recent_prices)
                                    max_price = recent_prices[max_recent_idx]
                                    days_ahead = int(max_recent_idx + 1)
                                    future_date = today + pd.DateOffset(days=days_ahead)
                                    
                                    sell_opportunities.append({
                                        'index': max_recent_idx,
                                        'price': float(max_price),
                                        'date': future_date,
                                        'risk_level': 10.0  # Giả định 10% rủi ro
                                    })
                                
                                sell_opportunities.sort(key=lambda x: x['risk_level'], reverse=True)
                                
                                # Tính toán ngày cho min/max price trong tương lai
                                min_days = int(min_price_idx % 30 + 1)
                                max_days = int(max_price_idx % 30 + 1)
                                min_price_future_date = today + pd.DateOffset(days=min_days)
                                max_price_future_date = today + pd.DateOffset(days=max_days)
                                
                                trading_analysis = {
                                    'volatility': price_volatility,
                                    'trend_direction': trend_direction,
                                    'trend_strength': trend_strength,
                                    'best_buy': buy_opportunities[0] if buy_opportunities else None,
                                    'best_sell': sell_opportunities[0] if sell_opportunities else None,
                                    'min_price_date': min_price_future_date,
                                    'max_price_date': max_price_future_date,
                                    'min_price': float(pred_flat[min_price_idx]),
                                    'max_price': float(pred_flat[max_price_idx])
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
                            
                        # === SECTION: TRADING SIGNALS ANALYSIS ===
                        st.markdown("---")
                        st.subheader("📊 Phân tích Tín hiệu Trading")
                        
                        # Tạo biểu đồ tín hiệu mua/bán trên giá
                        fig_trading = go.Figure()
                        
                        # Thêm giá dự đoán
                        fig_trading.add_trace(go.Scatter(
                            x=prediction_df['date'],
                            y=prediction_df['predicted'],
                            mode='lines',
                            name='Giá dự đoán',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Thêm điểm mua và bán
                        buy_dates, buy_prices = [], []
                        sell_dates, sell_prices = [], []
                        
                        # Đảm bảo signals và prediction_dates có cùng độ dài
                        min_length = min(len(signals), len(prediction_dates) - 1)  # -1 vì signals bắt đầu từ index 1
                        
                        for i in range(min_length):
                            signal_idx = i + 1  # signals bắt đầu từ ngày thứ 2
                            if signal_idx < len(prediction_dates) and signal_idx < len(predicted_prices):
                                if signals[i] == 'Mua':
                                    buy_dates.append(prediction_dates[signal_idx])
                                    buy_prices.append(predicted_prices[signal_idx][0] if len(predicted_prices[signal_idx].shape) > 0 else predicted_prices[signal_idx])
                                else:
                                    sell_dates.append(prediction_dates[signal_idx])
                                    sell_prices.append(predicted_prices[signal_idx][0] if len(predicted_prices[signal_idx].shape) > 0 else predicted_prices[signal_idx])
                        
                        # Thêm scatter points cho tín hiệu mua
                        if buy_dates:
                            fig_trading.add_trace(go.Scatter(
                                x=buy_dates,
                                y=buy_prices,
                                mode='markers',
                                name='Tín hiệu MUA',
                                marker=dict(color='green', size=10, symbol='triangle-up'),
                                showlegend=True
                            ))
                        
                        # Thêm scatter points cho tín hiệu bán
                        if sell_dates:
                            fig_trading.add_trace(go.Scatter(
                                x=sell_dates,
                                y=sell_prices,
                                mode='markers',
                                name='Tín hiệu BÁN',
                                marker=dict(color='red', size=10, symbol='triangle-down'),
                                showlegend=True
                            ))
                        
                        # Highlight điểm mua/bán tối ưu nếu có
                        if trading_info['best_buy'] and trading_info['best_buy']['date'] is not None:
                            try:
                                buy_date = trading_info['best_buy']['date']
                                buy_price = float(trading_info['best_buy']['price'])
                                
                                # Chỉ hiển thị nếu ngày trong phạm vi hợp lý
                                today = pd.Timestamp.today()
                                one_year_ago = today - pd.DateOffset(days=365)
                                if buy_date >= one_year_ago:
                                    fig_trading.add_trace(go.Scatter(
                                        x=[buy_date],
                                        y=[buy_price],
                                        mode='markers',
                                        name='Điểm MUA tối ưu',
                                        marker=dict(color='darkgreen', size=15, symbol='star'),
                                        showlegend=True
                                    ))
                            except Exception as e:
                                st.write(f"Debug: Không thể hiển thị điểm mua tối ưu - {str(e)}")
                        
                        if trading_info['best_sell'] and trading_info['best_sell']['date'] is not None:
                            try:
                                sell_date = trading_info['best_sell']['date']
                                sell_price = float(trading_info['best_sell']['price'])
                                
                                # Chỉ hiển thị nếu ngày trong phạm vi hợp lý
                                today = pd.Timestamp.today()
                                one_year_ago = today - pd.DateOffset(days=365)
                                if sell_date >= one_year_ago:
                                    fig_trading.add_trace(go.Scatter(
                                        x=[sell_date],
                                        y=[sell_price],
                                        mode='markers',
                                        name='Điểm BÁN tối ưu',
                                        marker=dict(color='darkred', size=15, symbol='star'),
                                        showlegend=True
                                    ))
                            except Exception as e:
                                st.write(f"Debug: Không thể hiển thị điểm bán tối ưu - {str(e)}")
                        
                        fig_trading.update_layout(
                            title='Tín hiệu Mua/Bán trên Biểu đồ Giá',
                            xaxis_title='Ngày',
                            yaxis_title='Giá (VND)',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_trading, use_container_width=True)
                        
                        # Hiển thị distribution của tín hiệu
                        col_pie, col_bar = st.columns(2)
                        
                        with col_pie:
                            if len(signals) > 0:
                                signal_counts = pd.Series(signals).value_counts()
                                fig_signals = px.pie(
                                    values=signal_counts.values,
                                    names=signal_counts.index,
                                    title="Phân bổ tín hiệu Mua/Bán",
                                    color_discrete_map={'Mua': 'green', 'Bán': 'red'}
                                )
                                st.plotly_chart(fig_signals, use_container_width=True)
                            else:
                                st.info("Không có tín hiệu để hiển thị")
                        
                        with col_bar:
                            # Tạo biểu đồ trend strength theo thời gian
                            trend_data = []
                            window_size = 10
                            
                            # Flatten predicted_prices để đảm bảo là 1D array
                            pred_flat = predicted_prices.flatten()
                            
                            if len(pred_flat) > window_size:
                                for i in range(window_size, len(pred_flat)):
                                    window_data = pred_flat[i-window_size:i]
                                    if len(window_data) > 0 and window_data[0] != 0:
                                        trend_strength = (window_data[-1] - window_data[0]) / window_data[0] * 100
                                        trend_data.append(trend_strength)
                                
                                if trend_data and len(trend_data) > 0:
                                    # Đảm bảo số lượng dates khớp với trend_data
                                    available_dates = len(prediction_dates) - window_size
                                    if available_dates > 0:
                                        trend_dates = prediction_dates[window_size:window_size + len(trend_data)]
                                        
                                        fig_trend = px.bar(
                                            x=trend_dates,
                                            y=trend_data,
                                            title="Cường độ Xu hướng (%)",
                                            color=trend_data,
                                            color_continuous_scale=['red', 'yellow', 'green']
                                        )
                                        fig_trend.update_layout(height=400)
                                        st.plotly_chart(fig_trend, use_container_width=True)
                                    else:
                                        st.info("Không đủ dữ liệu để hiển thị biểu đồ xu hướng")
                                else:
                                    st.info("Không có dữ liệu xu hướng để hiển thị")
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
                        if trading_info['best_buy'] and trading_info['best_sell']:
                            try:
                                buy_price = float(trading_info['best_buy']['price'])
                                sell_price = float(trading_info['best_sell']['price'])
                                
                                if buy_price > 0:
                                    potential_profit = ((sell_price - buy_price) / buy_price * 100)
                                    
                                    if potential_profit > 0:
                                        st.success(f"""
                                        ### 📈 Tiềm năng lợi nhuận: {potential_profit:.1f}%
                                        **Chiến lược:** Mua ở {buy_price:,.0f} VND, bán ở {sell_price:,.0f} VND
                                        **Lãi dự kiến:** {sell_price - buy_price:,.0f} VND/cổ phiếu
                                        """)
                                    else:
                                        st.warning("⚠️ Không có cơ hội lợi nhuận rõ ràng trong khoảng thời gian dự đoán")
                                else:
                                    st.warning("⚠️ Dữ liệu giá không hợp lệ để tính toán lợi nhuận")
                            except Exception as e:
                                st.warning("⚠️ Không thể tính toán lợi nhuận tiềm năng do dữ liệu không đầy đủ")
                        else:
                            # Hiển thị thông tin dự đoán chung nếu không có điểm mua/bán cụ thể
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
                            signal_counts = pd.Series(signals).value_counts() if len(signals) > 0 else pd.Series()
                            buy_count = signal_counts.get('Mua', 0)
                            sell_count = signal_counts.get('Bán', 0)
                            
                            st.warning(f"""
                            - **Độ biến động:** {trading_info['volatility']:.1f}% {'(Cao)' if trading_info['volatility'] > 5 else '(Thấp)'}
                            - **Xu hướng ngắn hạn:** {trading_info['trend_direction']} ({trading_info['trend_strength']:.1f}%)
                            - **Tín hiệu Mua:** {buy_count} lần
                            - **Tín hiệu Bán:** {sell_count} lần
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
                            accuracy = max(0.0, 100.0 - float(mape))
                            
                            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                            
                            with col_metric1:
                                st.metric("RMSE", f"{rmse:,.0f} VND")
                            with col_metric2:
                                st.metric("MAE", f"{mae:,.0f} VND")
                            with col_metric3:
                                st.metric("MAPE", f"{mape:.2f}%")
                            with col_metric4:
                                st.metric("Độ chính xác", f"{accuracy:.2f}%")
                            
                            # Biểu đồ scatter cho accuracy
                            fig_accuracy = px.scatter(
                                x=real_clean, 
                                y=pred_clean,
                                title="Tương quan Giá thực tế vs Dự đoán",
                                labels={'x': 'Giá thực tế (VND)', 'y': 'Giá dự đoán (VND)'}
                            )
                            # Thêm đường y=x để show perfect prediction
                            min_val = min(np.min(real_clean), np.min(pred_clean))
                            max_val = max(np.max(real_clean), np.max(pred_clean))
                            fig_accuracy.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Dự đoán hoàn hảo',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            st.plotly_chart(fig_accuracy, use_container_width=True)
                        else:
                            st.error("Không thể tính toán độ chính xác do dữ liệu không hợp lệ")
                        
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
