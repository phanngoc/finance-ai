import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from vnstock import Vnstock
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Phân tích định giá & sức khỏe tài chính ngân hàng",
    page_icon="🏦",
    layout="wide"
)

class BankAnalyzer:
    def __init__(self):
        self.banks = {
            'ACB': 'Ngân hàng TMCP Á Châu',
            'VCB': 'Ngân hàng TMCP Ngoại thương Việt Nam', 
            'TCB': 'Ngân hàng TMCP Kỹ thương Việt Nam',
            'STB': 'Ngân hàng TMCP Sài Gòn Thương Tín',
            'BID': 'Ngân hàng TMCP Đầu tư và Phát triển Việt Nam'
        }
        self.financial_data = {}
        
    def get_bank_data(self, symbol):
        """Lấy dữ liệu tài chính cho một ngân hàng"""
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            
            # Lấy các báo cáo tài chính
            balance_sheet = stock.finance.balance_sheet(period='year', lang='vi', dropna=True)
            income_statement = stock.finance.income_statement(period='year', lang='vi', dropna=True)
            ratio = stock.finance.ratio(period='year', lang='vi', dropna=True)
            
            # Lấy giá cổ phiếu hiện tại (có thể cần API khác)
            try:
                price_data = stock.quote.history(period='1Y')
                current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
            except:
                current_price = 0
            
            return {
                'balance_sheet': balance_sheet,
                'income_statement': income_statement,
                'ratio': ratio,
                'current_price': current_price
            }
        except Exception as e:
            st.error(f"Lỗi khi lấy dữ liệu cho {symbol}: {e}")
            return None
    
    def calculate_financial_health_score(self, data):
        """Tính điểm sức khỏe tài chính (0-100)"""
        if not data or data['ratio'].empty:
            return 0
        
        ratio_df = data['ratio']
        latest_data = ratio_df.iloc[-1]  # Dữ liệu năm gần nhất
        
        score = 0
        
        try:
            # ROE > 15% (+20), 10-15% (+10), <10% (0)
            roe = latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'), 0)
            if roe > 15:
                score += 20
            elif roe > 10:
                score += 10
                
            # ROA > 1.2% (+15), 0.8-1.2% (+10), <0.8% (0)
            roa = latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'ROA (%)'), 0)
            if roa > 1.2:
                score += 15
            elif roa > 0.8:
                score += 10
            
            # P/E < 10 (+15), 10-15 (+10), >15 (0) - thấp hơn tốt hơn
            pe = latest_data.get(('Chỉ tiêu định giá', 'P/E'), 0)
            if 0 < pe < 10:
                score += 15
            elif 10 <= pe <= 15:
                score += 10
            
            # P/B < 1.5 (+10), 1.5-2.0 (+5), >2.0 (0)
            pb = latest_data.get(('Chỉ tiêu định giá', 'P/B'), 0)
            if 0 < pb < 1.5:
                score += 10
            elif 1.5 <= pb <= 2.0:
                score += 5
            
            # Đòn bẩy tài chính < 10 (+10), 10-15 (+5), >15 (0)
            leverage = latest_data.get(('Chỉ tiêu thanh khoản', 'Đòn bẩy tài chính'), 0)
            if leverage < 10:
                score += 10
            elif leverage <= 15:
                score += 5
            
            # Thanh khoản hiện thời > 1.2 (+10), 1.0-1.2 (+5), <1.0 (0)
            current_ratio = latest_data.get(('Chỉ tiêu thanh khoản', 'Chỉ số thanh toán hiện thời'), 0)
            if current_ratio > 1.2:
                score += 10
            elif current_ratio >= 1.0:
                score += 5
            
            # Biên lợi nhuận ròng > 20% (+15), 15-20% (+10), <15% (0)
            net_margin = latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'), 0)
            if net_margin > 20:
                score += 15
            elif net_margin > 15:
                score += 10
            
            # Tỷ suất cổ tức > 5% (+5), 3-5% (+3), <3% (0)
            dividend_yield = latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'Tỷ suất cổ tức (%)'), 0)
            if dividend_yield > 5:
                score += 5
            elif dividend_yield >= 3:
                score += 3
                
        except Exception as e:
            st.warning(f"Lỗi khi tính điểm sức khỏe: {e}")
        
        return min(score, 100)  # Giới hạn tối đa 100 điểm
    
    def calculate_intrinsic_value_pb_roe(self, data, risk_free_rate=0.05):
        """Tính giá trị hợp lý dựa trên mô hình P/B và ROE"""
        if not data or data['ratio'].empty:
            return 0, 0
        
        ratio_df = data['ratio']
        latest_data = ratio_df.iloc[-1]
        
        try:
            roe = latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'), 0) / 100
            pb_current = latest_data.get(('Chỉ tiêu định giá', 'P/B'), 0)
            bvps = latest_data.get(('Chỉ tiêu định giá', 'BVPS (VND)'), 0)
            
            # Ước tính chi phí vốn chủ sở hữu (thường cao hơn lãi suất phi rủi ro 3-5%)
            cost_of_equity = risk_free_rate + 0.04  # 4% risk premium cho ngân hàng
            
            # Fair P/B = ROE / Cost of Equity
            fair_pb = roe / cost_of_equity if cost_of_equity > 0 else 0
            
            # Intrinsic Value = BVPS * Fair P/B
            intrinsic_value = bvps * fair_pb
            
            return intrinsic_value, fair_pb
            
        except Exception as e:
            st.warning(f"Lỗi khi tính giá trị hợp lý: {e}")
            return 0, 0
    
    def get_key_metrics(self, data):
        """Lấy các chỉ số quan trọng"""
        if not data or data['ratio'].empty:
            return {}
        
        ratio_df = data['ratio']
        latest_data = ratio_df.iloc[-1]

        return {
            'ROE': latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'), 0),
            'ROA': latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'ROA (%)'), 0),
            'P/E': latest_data.get(('Chỉ tiêu định giá', 'P/E'), 0),
            'P/B': latest_data.get(('Chỉ tiêu định giá', 'P/B'), 0),
            'EPS': latest_data.get(('Chỉ tiêu định giá', 'EPS (VND)'), 0),
            'BVPS': latest_data.get(('Chỉ tiêu định giá', 'BVPS (VND)'), 0),
            'Net_Margin': latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'Biên lợi nhuận ròng (%)'), 0),
            'Dividend_Yield': latest_data.get(('Chỉ tiêu khả năng sinh lợi', 'Tỷ suất cổ tức (%)'), 0),
            'Leverage': latest_data.get(('Chỉ tiêu thanh khoản', 'Đòn bẩy tài chính'), 0),
            'Current_Ratio': latest_data.get(('Chỉ tiêu thanh khoản', 'Chỉ số thanh toán hiện thời'), 0)
        }

def main():
    st.title("🏦 Phân tích định giá & sức khỏe tài chính ngân hàng")
    st.markdown("### Xác định ngân hàng nào đang bị định giá thấp trong top 5 ngân hàng Việt Nam")
    
    analyzer = BankAnalyzer()
    
    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình phân tích")
        risk_free_rate = st.slider("Lãi suất phi rủi ro (%)", 3.0, 8.0, 5.0, 0.1) / 100
        selected_banks = st.multiselect(
            "Chọn ngân hàng để phân tích",
            list(analyzer.banks.keys()),
            default=list(analyzer.banks.keys())
        )
        
        if st.button("🔄 Cập nhật dữ liệu"):
            st.session_state.update_data = True
    
    # Load data
    if 'bank_analysis' not in st.session_state or st.session_state.get('update_data', False):
        with st.spinner("Đang tải dữ liệu tài chính..."):
            bank_analysis = {}
            progress_bar = st.progress(0)
            
            for i, bank in enumerate(selected_banks):
                progress_bar.progress((i + 1) / len(selected_banks))
                data = analyzer.get_bank_data(bank)
                
                if data:
                    health_score = analyzer.calculate_financial_health_score(data)
                    intrinsic_value, fair_pb = analyzer.calculate_intrinsic_value_pb_roe(data, risk_free_rate)
                    key_metrics = analyzer.get_key_metrics(data)
                    bank_analysis[bank] = {
                        'name': analyzer.banks[bank],
                        'data': data,
                        'health_score': health_score,
                        'intrinsic_value': intrinsic_value,
                        'fair_pb': fair_pb,
                        'current_price': data['current_price'],
                        'metrics': key_metrics
                    }
            
            st.session_state.bank_analysis = bank_analysis
            st.session_state.update_data = False
            progress_bar.empty()
    
    bank_analysis = st.session_state.get('bank_analysis', {})
    
    if not bank_analysis:
        st.warning("Không có dữ liệu để hiển thị. Vui lòng chọn ngân hàng và cập nhật dữ liệu.")
        return
    
    # Tạo DataFrame tổng hợp
    summary_data = []
    for bank, analysis in bank_analysis.items():
        metrics = analysis['metrics']
        current_price = analysis['current_price']
        intrinsic_value = analysis['intrinsic_value']
        print("summary_data:metrics", metrics)
        # Tính % định giá
        if current_price > 0 and intrinsic_value > 0:
            valuation_gap = ((current_price - intrinsic_value) / intrinsic_value) * 100
        else:
            valuation_gap = 0
        
        summary_data.append({
            'Ngân hàng': bank,
            'Tên đầy đủ': analysis['name'],
            'Giá hiện tại (VND)': f"{current_price:,.0f}" if current_price > 0 else "N/A",
            'Giá hợp lý (VND)': f"{intrinsic_value:,.0f}" if intrinsic_value > 0 else "N/A",
            'Chênh lệch (%)': f"{valuation_gap:+.1f}%" if valuation_gap != 0 else "N/A",
            'Sức khỏe TÀI (0-100)': analysis['health_score'],
            'ROE (%)': f"{metrics['ROE']:.1f}",
            'ROA (%)': f"{metrics['ROA']:.2f}",
            'P/E': f"{metrics['P/E']:.1f}" if metrics['P/E'] > 0 else "N/A",
            'P/B': f"{metrics['P/B']:.2f}" if metrics['P/B'] > 0 else "N/A",
            'Fair P/B': f"{analysis['fair_pb']:.2f}" if analysis['fair_pb'] > 0 else "N/A",
            'Biên LN ròng (%)': f"{metrics['Net_Margin']:.1f}",
            'Tỷ suất cổ tức (%)': f"{metrics['Dividend_Yield']:.1f}",
            # Thêm columns số để vẽ biểu đồ
            'ROE_numeric': metrics['ROE'],
            'ROA_numeric': metrics['ROA'],
            'P/E_numeric': metrics['P/E'] if metrics['P/E'] > 0 else None,
            'P/B_numeric': metrics['P/B'] if metrics['P/B'] > 0 else None,
            'Net_Margin_numeric': metrics['Net_Margin'],
            'Dividend_Yield_numeric': metrics['Dividend_Yield'],
            'Valuation_Gap_numeric': valuation_gap
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Hiển thị bảng tổng hợp
    st.subheader("📊 Bảng tổng hợp phân tích")
    st.dataframe(df_summary, use_container_width=True)
    
    # Phân tích và đánh giá
    st.subheader("💡 Phân tích định giá")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔻 Ngân hàng bị định giá thấp (Undervalued)")
        undervalued = []
        for bank, analysis in bank_analysis.items():
            current_price = analysis['current_price']
            intrinsic_value = analysis['intrinsic_value']
            if current_price > 0 and intrinsic_value > 0:
                gap = ((current_price - intrinsic_value) / intrinsic_value) * 100
                if gap < -10:  # Giá thấp hơn giá trị hợp lý 10%
                    undervalued.append((bank, gap, analysis['health_score']))
        
        undervalued.sort(key=lambda x: x[1])  # Sắp xếp theo mức độ undervalued
        
        if undervalued:
            for bank, gap, health in undervalued:
                st.success(f"**{bank}**: {gap:+.1f}% (Sức khỏe: {health}/100)")
        else:
            st.info("Không có ngân hàng nào bị định giá thấp đáng kể")
    
    with col2:
        st.markdown("#### 🔺 Ngân hàng bị định giá cao (Overvalued)")
        overvalued = []
        for bank, analysis in bank_analysis.items():
            current_price = analysis['current_price']
            intrinsic_value = analysis['intrinsic_value']
            if current_price > 0 and intrinsic_value > 0:
                gap = ((current_price - intrinsic_value) / intrinsic_value) * 100
                if gap > 10:  # Giá cao hơn giá trị hợp lý 10%
                    overvalued.append((bank, gap, analysis['health_score']))
        
        overvalued.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo mức độ overvalued
        
        if overvalued:
            for bank, gap, health in overvalued:
                st.warning(f"**{bank}**: {gap:+.1f}% (Sức khỏe: {health}/100)")
        else:
            st.info("Không có ngân hàng nào bị định giá cao đáng kể")
    
    # Biểu đồ trực quan
    st.subheader("📈 Biểu đồ trực quan")
    
    # Lọc dữ liệu có giá trị hợp lệ cho biểu đồ
    df_chart = df_summary.dropna(subset=['P/B_numeric', 'ROE_numeric'])
    
    if not df_chart.empty:
        # Biểu đồ sức khỏe tài chính vs P/B
        fig1 = px.scatter(
            df_chart,
            x='Sức khỏe TÀI (0-100)',
            y='P/B_numeric',
            size='ROE_numeric',
            color='Ngân hàng',
            title="Sức khỏe tài chính vs P/B (Size = ROE)",
            hover_data=['ROE_numeric', 'ROA_numeric'],
            labels={'P/B_numeric': 'P/B', 'ROE_numeric': 'ROE (%)', 'ROA_numeric': 'ROA (%)'}
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Không đủ dữ liệu để vẽ biểu đồ scatter")
    
    # Biểu đồ so sánh P/B hiện tại vs Fair P/B
    banks_with_data = []
    current_pb = []
    fair_pb = []
    
    for bank, analysis in bank_analysis.items():
        if analysis['fair_pb'] > 0 and analysis['metrics']['P/B'] > 0:
            banks_with_data.append(bank)
            current_pb.append(analysis['metrics']['P/B'])
            fair_pb.append(analysis['fair_pb'])
    
    if banks_with_data:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='P/B hiện tại',
            x=banks_with_data,
            y=current_pb,
            marker_color='lightblue'
        ))
        fig2.add_trace(go.Bar(
            name='P/B hợp lý',
            x=banks_with_data,
            y=fair_pb,
            marker_color='orange'
        ))
        
        fig2.update_layout(
            title="So sánh P/B hiện tại vs P/B hợp lý",
            xaxis_title="Ngân hàng",
            yaxis_title="P/B Ratio",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Không đủ dữ liệu để vẽ biểu đồ so sánh P/B")
    
    # Biểu đồ chênh lệch định giá
    df_valuation = df_summary[df_summary['Valuation_Gap_numeric'] != 0].copy()
    if not df_valuation.empty:
        fig2_1 = px.bar(
            df_valuation,
            x='Ngân hàng',
            y='Valuation_Gap_numeric',
            color='Valuation_Gap_numeric',
            title="Chênh lệch định giá so với giá trị hợp lý (%)",
            color_continuous_scale='RdYlGn_r',
            labels={'Valuation_Gap_numeric': 'Chênh lệch (%)'}
        )
        fig2_1.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Giá trị hợp lý")
        fig2_1.update_layout(height=400)
        st.plotly_chart(fig2_1, use_container_width=True)
    
    # Radar chart cho các chỉ số
    st.subheader("🎯 So sánh đa chiều các ngân hàng")
    
    if len(bank_analysis) >= 2:
        selected_for_radar = st.multiselect(
            "Chọn ngân hàng để so sánh (tối đa 3)",
            list(bank_analysis.keys()),
            default=list(bank_analysis.keys())[:3]
        )
        
        if selected_for_radar:
            fig3 = go.Figure()
            
            categories = ['Sức khỏe TÀI', 'ROE', 'ROA', 'Biên LN ròng', 'Tỷ suất cổ tức']
            
            def normalize_to_100_scale(value, max_val):
                """Chuẩn hóa giá trị về thang 0-100 dựa trên max_val"""
                if value <= 0 or max_val <= 0:
                    return 0
                return min(100, (value / max_val) * 100)
            
            # Tìm max cho từng chỉ số trong các ngân hàng được chọn
            max_roe = max(bank_analysis[bank]['metrics']['ROE'] for bank in selected_for_radar)
            max_roa = max(bank_analysis[bank]['metrics']['ROA'] for bank in selected_for_radar)
            max_net_margin = max(bank_analysis[bank]['metrics']['Net_Margin'] for bank in selected_for_radar)
            max_dividend_yield = max(bank_analysis[bank]['metrics']['Dividend_Yield'] for bank in selected_for_radar)
            
            for bank in selected_for_radar:
                analysis = bank_analysis[bank]
                metrics = analysis['metrics']
                
                # Chuẩn hóa từng chỉ số về thang 0-100 với max là giá trị lớn nhất trong nhóm
                values = [
                    # Sức khỏe tài chính (đã ở thang 0-100)
                    analysis['health_score'],
                    
                    # ROE: max trong nhóm
                    normalize_to_100_scale(metrics['ROE'], max_roe),
                    
                    # ROA: max trong nhóm
                    normalize_to_100_scale(metrics['ROA'], max_roa),
                    
                    # Biên lợi nhuận ròng: max trong nhóm
                    normalize_to_100_scale(metrics['Net_Margin'], max_net_margin),
                    
                    # Tỷ suất cổ tức: max trong nhóm
                    normalize_to_100_scale(metrics['Dividend_Yield'], max_dividend_yield)
                ]
                print("Normalized values for", bank, ":", values)
                fig3.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=bank,
                    hovertemplate=f'<b>{bank}</b><br>' +
                                 'Sức khỏe TÀI: %{r[0]:.1f}/100<br>' +
                                 f'ROE: {metrics["ROE"]:.1f}% (Score: %{{r[1]:.1f}})<br>' +
                                 f'ROA: {metrics["ROA"]:.2f}% (Score: %{{r[2]:.1f}})<br>' +
                                 f'Biên LN ròng: {metrics["Net_Margin"]:.1f}% (Score: %{{r[3]:.1f}})<br>' +
                                 f'Tỷ suất cổ tức: {metrics["Dividend_Yield"]:.1f}% (Score: %{{r[4]:.1f}})<br>' +
                                 '<extra></extra>'
                ))
            
            fig3.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickmode='linear',
                        tick0=0,
                        dtick=20
                    )),
                showlegend=True,
                title="Radar Chart - So sánh đa chiều (Thang điểm 0-100)",
                height=600
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Thêm giải thích về cách chuẩn hóa
            with st.expander("📖 Giải thích thang điểm radar chart"):
                st.markdown("""
                **Cách chuẩn hóa các chỉ số về thang 0-100 (đã điều chỉnh cân đối):**
                
                - **Sức khỏe TÀI**: Đã ở thang 0-100
                - **ROE**: 100 điểm khi ROE ≥ 15%, tỷ lệ thuận với ROE thực tế (max 25%)
                - **ROA**: 100 điểm khi ROA ≥ 1.0%, tỷ lệ thuận với ROA thực tế (max 2.0%)
                - **Biên LN ròng**: 100 điểm khi ≥ 15%, tỷ lệ thuận với biên LN thực tế (max 30%)
                - **Tỷ suất cổ tức**: 100 điểm khi ≥ 4%, tỷ lệ thuận với tỷ suất thực tế (max 8%)
                
                *Các ngưỡng được điều chỉnh phù hợp với thực tế ngành ngân hàng Việt Nam*
                
                *Hover vào các điểm trên biểu đồ để xem giá trị thực tế và điểm số chuẩn hóa*
                """)
    
    # Kết luận và khuyến nghị
    st.subheader("🎯 Kết luận và khuyến nghị đầu tư")
    
    # Tìm ngân hàng tốt nhất để đầu tư
    investment_scores = []
    for bank, analysis in bank_analysis.items():
        current_price = analysis['current_price']
        intrinsic_value = analysis['intrinsic_value']
        health_score = analysis['health_score']
        
        if current_price > 0 and intrinsic_value > 0:
            valuation_gap = ((current_price - intrinsic_value) / intrinsic_value) * 100
            # Điểm đầu tư = Sức khỏe + bonus nếu undervalued
            investment_score = health_score + max(-valuation_gap * 0.5, 0)
            investment_scores.append((bank, investment_score, valuation_gap, health_score))
    
    investment_scores.sort(key=lambda x: x[1], reverse=True)
    
    if investment_scores:
        st.markdown("#### 🏆 Xếp hạng khuyến nghị đầu tư")
        for i, (bank, score, gap, health) in enumerate(investment_scores[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            status = "Undervalued" if gap < -5 else "Overvalued" if gap > 5 else "Fair Value"
            st.info(f"{medal} **{bank}**: Điểm đầu tư {score:.1f} | Định giá: {status} ({gap:+.1f}%) | Sức khỏe: {health}/100")
    
    # Thông tin bổ sung
    with st.expander("ℹ️ Thông tin về phương pháp phân tích"):
        st.markdown("""
        **Cách tính điểm sức khỏe tài chính (0-100 điểm):**
        - ROE > 15%: +20 điểm
        - ROA > 1.2%: +15 điểm  
        - P/E < 10: +15 điểm
        - P/B < 1.5: +10 điểm
        - Đòn bẩy < 10: +10 điểm
        - Thanh khoản > 1.2: +10 điểm
        - Biên LN ròng > 20%: +15 điểm
        - Tỷ suất cổ tức > 5%: +5 điểm
        
        **Mô hình định giá P/B-ROE:**
        - Fair P/B = ROE / Chi phí vốn chủ sở hữu
        - Giá trị hợp lý = BVPS × Fair P/B
        - Chi phí vốn = Lãi suất phi rủi ro + Risk Premium (4%)
        """)

if __name__ == "__main__":
    main()