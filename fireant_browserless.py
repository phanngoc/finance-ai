import asyncio
import streamlit as st
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import openai
import base64
import io
from PIL import Image
import os
import uuid
import tempfile
from dotenv import load_dotenv
from utils import DoclingProcessor
import os
# Load environment variables from .env file
load_dotenv()

class FireantBrowserlessScraper:
    def __init__(self, browserless_url=None, base_url=None):
        # Load configuration from environment variables
        self.browserless_url = browserless_url or os.getenv("BROWSERLESS_URL", "ws://localhost:3000")
        self.base_url = base_url or os.getenv("FIREANT_BASE_URL", "https://fireant.vn")
        
        # Data storage directories
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.screenshots_dir = os.getenv("SCREENSHOTS_DIR", "data/screenshots")
        self.analysis_dir = os.getenv("ANALYSIS_DIR", "data/analysis")
        self.complete_analysis_dir = os.getenv("COMPLETE_ANALYSIS_DIR", "data/complete_analysis")
        
        # Create directories if they don't exist
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.complete_analysis_dir, exist_ok=True)
    
    def create_env_file(self):
        """
        Tạo file .env từ template nếu chưa tồn tại
        """
        import streamlit as st
        
        env_file = ".env"
        env_example = "env.example"
        
        if not os.path.exists(env_file) and os.path.exists(env_example):
            try:
                # Copy từ env.example sang .env
                import shutil
                shutil.copy(env_example, env_file)
                st.success(f"✅ Đã tạo file {env_file} từ {env_example}")
                st.info("💡 Hãy chỉnh sửa file .env với API key của bạn")
                return True
            except Exception as e:
                st.error(f"❌ Lỗi khi tạo file .env: {str(e)}")
                return False
        elif os.path.exists(env_file):
            st.info(f"✅ File {env_file} đã tồn tại")
            return True
        else:
            st.warning(f"⚠️ Không tìm thấy file {env_example}")
            return False

    async def _handle_popups(self, page):
        """
        Xử lý và đóng các popup trước khi chụp screenshot
        
        Args:
            page: Playwright page object
        """
        try:
            # Đợi một chút để popup có thể xuất hiện
            await page.wait_for_timeout(1000)
            
            # Xử lý đặc biệt cho FireAnt notification popup
            await page.evaluate("""
                () => {
                    // Tìm và xử lý FireAnt notification dialog
                    const fireantDialogs = document.querySelectorAll('div[role="dialog"][data-state="open"]');
                    fireantDialogs.forEach(dialog => {
                        const header = dialog.querySelector('h2');
                        if (header && header.textContent.includes('Nhận thông báo')) {
                            // Tìm nút "Để sau"
                            const buttons = dialog.querySelectorAll('button');
                            let clicked = false;
                            
                            buttons.forEach(btn => {
                                const text = btn.textContent.toLowerCase();
                                if (text.includes('để sau') || text.includes('sau')) {
                                    btn.click();
                                    clicked = true;
                                }
                            });
                            
                            // Nếu không tìm thấy nút "Để sau", thử click nút X
                            if (!clicked) {
                                const closeBtn = dialog.querySelector('button[type="button"] svg');
                                if (closeBtn) {
                                    closeBtn.closest('button').click();
                                    clicked = true;
                                }
                            }
                            
                            // Nếu vẫn không click được, ẩn dialog
                            if (!clicked) {
                                dialog.style.display = 'none';
                                dialog.style.visibility = 'hidden';
                                dialog.style.opacity = '0';
                                dialog.style.zIndex = '-9999';
                                dialog.style.pointerEvents = 'none';
                            }
                        }
                    });
                }
            """)

            # Đợi thêm một chút để các thay đổi có hiệu lực
            await page.wait_for_timeout(500)
            
        except Exception as e:
            print(f"Lỗi khi xử lý popup: {str(e)}")

    async def _smart_scroll_and_capture(self, page, symbol, page_type="general"):
        """
        Scroll thêm 1 bước bằng độ cao màn hình hiện tại, chụp ảnh và lưu thành file temp
        
        Args:
            page: Playwright page object
            symbol: Mã chứng khoán
            page_type: Loại trang ("general", "financial", "chart")
            
        Returns:
            str: Tên file temp đã lưu
        """
        try:
            # Lấy kích thước viewport hiện tại
            viewport = await page.viewport_size()
            screen_height = viewport['height']
            
            # Scroll thêm 1 bước bằng độ cao màn hình
            await page.evaluate(f"""
                () => {{
                    window.scrollBy(0, {screen_height});
                }}
            """)
            
            # Đợi một chút để trang load sau khi scroll
            await page.wait_for_timeout(1000)
            
            # Chụp ảnh toàn trang sau khi scroll
            screenshot = await page.screenshot(full_page=True)
            
            # Tạo UUID để định danh file
            file_uuid = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Tạo tên file temp với UUID
            temp_filename = f"temp_{symbol}_{page_type}_{timestamp}_{file_uuid}.png"
            temp_filepath = os.path.join(tempfile.gettempdir(), temp_filename)
            
            # Lưu screenshot vào file temp
            with open(temp_filepath, "wb") as f:
                f.write(screenshot)
            
            return temp_filepath
            
        except Exception as e:
            st.error(f"❌ Lỗi khi scroll và chụp ảnh: {str(e)}")
            return None
    
    async def _click_financial_tab_and_capture(self, page, symbol):
        """
        Click vào tab "Tài chính" và chụp ảnh chart
        
        Args:
            page: Playwright page object
            symbol: Mã chứng khoán
            
        Returns:
            tuple: (screenshot, success_status)
        """
        try:
            # Đợi trang load hoàn toàn
            await page.wait_for_timeout(2000)
            
            # Tìm và click vào button "Tài chính"
            financial_button = await page.query_selector('button:has-text("Tài chính")')
            if financial_button:
                await financial_button.click()
                st.success("✅ Đã click vào tab 'Tài chính'")
                
                # Đợi trang load sau khi click
                await page.wait_for_timeout(3000)
                
                # Scroll để load tất cả chart
                await page.evaluate("""
                    () => {
                        window.scrollTo(0, document.body.scrollHeight);
                        setTimeout(() => {
                            window.scrollTo(0, 0);
                        }, 1000);
                    }
                """)
                
                # Chụp ảnh toàn trang
                screenshot = await page.screenshot()
                return screenshot, True
            else:
                st.warning("⚠️ Không tìm thấy button 'Tài chính'")
                return None, False
                
        except Exception as e:
            st.error(f"❌ Lỗi khi click tab Tài chính: {str(e)}")
            return None, False

    def save_financial_analysis(self, analysis_text, symbol):
        """
        Lưu phân tích tài chính vào file markdown
        
        Args:
            analysis_text: Nội dung phân tích từ OpenAI
            symbol: Mã chứng khoán
        """
        try:
            # Tạo thư mục lưu trữ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"{self.analysis_dir}/{symbol}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Lưu file markdown
            filename = f"{save_dir}/financial_analysis.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(analysis_text)
            
            return filename
            
        except Exception as e:
            st.error(f"❌ Lỗi khi lưu phân tích: {str(e)}")
            return None

    async def get_stock_info_browserless(self, symbol):
        """Sử dụng Browserless để lấy thông tin chứng khoán"""
        try:
            async with async_playwright() as p:
                # Kết nối đến Browserless
                browser = await p.chromium.connect_over_cdp(self.browserless_url)
                page = await browser.new_page()
                
                # Cấu hình viewport
                await page.set_viewport_size({"width": 1920, "height": 1080})
                
                # Truy cập trang Fireant
                url = f"{self.base_url}/ma-chung-khoan/{symbol}"
                await page.goto(url, wait_until="networkidle")
                
                # Xử lý popup ngay sau khi trang load xong
                await self._handle_popups(page)

                # Lấy thông tin cơ bản từ bảng tài chính sau khi đã scroll
                stock_data = await page.evaluate("""
                    () => {
                        const data = {};
                        
                        // Lấy tên công ty từ header mới
                        const companyNameHeader = document.querySelector('div.sticky.z-50.mb-1.bg-white');
                        if (companyNameHeader) {
                            data.company_name = companyNameHeader.textContent.trim();
                        } else {
                            // Fallback cho các selector cũ
                            const companyName = document.querySelector('h1, .company-name, .stock-name, .symbol-name');
                            data.company_name = companyName ? companyName.textContent.trim() : 'Không tìm thấy';
                        }
                        
                        // Lấy mã chứng khoán và sàn từ header mới
                        const symbolElement = document.querySelector('div.sticky.z-50.mb-1.bg-white');
                        if (symbolElement) {
                            const symbolText = symbolElement.textContent.trim();
                            // Tách mã và sàn (ví dụ: "TCB:HSX")
                            const parts = symbolText.split(':');
                            if (parts.length >= 2) {
                                data.symbol_code = parts[0].trim();
                                data.exchange = parts[1].trim();
                            } else {
                                data.symbol_code = symbolText;
                                data.exchange = 'N/A';
                            }
                        }
                        
                        // Lấy giá hiện tại từ header mới
                        const currentPriceElement = document.querySelector('div.sticky.z-50.mb-1.bg-white span');
                        if (currentPriceElement) {
                            data.current_price = currentPriceElement.textContent.trim();
                        } else {
                            // Fallback cho các selector cũ
                            const currentPrice = document.querySelector('.current-price, .price, [data-price], .stock-price, .text-lg.font-bold');
                            data.current_price = currentPrice ? currentPrice.textContent.trim() : 'N/A';
                        }
                        
                        // Lấy thay đổi giá từ header mới
                        const priceChangeElement = document.querySelector('div.sticky.z-50.mb-1.bg-white span');
                        if (priceChangeElement && priceChangeElement.style.color) {
                            data.price_change = priceChangeElement.textContent.trim();
                        } else {
                            // Fallback cho các selector cũ
                            const priceChange = document.querySelector('.price-change, .change, [data-change], .stock-change');
                            data.price_change = priceChange ? priceChange.textContent.trim() : 'N/A';
                        }
                        
                        // Lấy dữ liệu từ bảng tài chính
                        const table = document.querySelector('table tbody');
                        if (table) {
                            const rows = table.querySelectorAll('tr');
                            rows.forEach(row => {
                                const cells = row.querySelectorAll('td');
                                if (cells.length >= 2) {
                                    const label = cells[0].textContent.trim();
                                    const value = cells[1].textContent.trim();
                                    
                                    switch(label) {
                                        case 'Tham chiếu':
                                            data.reference_price = value;
                                            break;
                                        case 'Mở cửa':
                                            data.open_price = value;
                                            break;
                                        case 'Thấp - Cao':
                                            data.low_high = value;
                                            break;
                                        case 'Khối lượng':
                                            data.volume = value;
                                            break;
                                        case 'Giá trị':
                                            data.value = value;
                                            break;
                                        case 'KLTB 10 ngày':
                                            data.avg_volume_10d = value;
                                            break;
                                        case 'Beta':
                                            data.beta = value;
                                            break;
                                        case 'Thị giá vốn':
                                            data.market_cap = value;
                                            break;
                                        case 'Số lượng CPLH':
                                            data.outstanding_shares = value;
                                            break;
                                        case 'P/E':
                                            data.pe_ratio = value;
                                            break;
                                        case 'EPS':
                                            data.eps = value;
                                            break;
                                    }
                                }
                            });
                        }
                        
                        return data;
                    }
                """)
                print('stock_data', stock_data)
                                
                # Sử dụng cơ chế scroll thông minh cho trang chính
                screenshot_filepath = await self._smart_scroll_and_capture(page, symbol, "general")
                
                # Bước 2: Click vào tab "Tài chính" và chụp ảnh chart
                financial_screenshot, financial_success = await self._click_financial_tab_and_capture(page, symbol)
                
                # Bước 3: Phân tích chart với OpenAI nếu chụp ảnh thành công
                financial_analysis = None
                if financial_success and financial_screenshot:
                    with st.spinner("🤖 Đang phân tích chart..."):
                        # Sử dụng DoclingProcessor để trích xuất thông tin và tạo báo cáo markdown
                        docling = DoclingProcessor()
                        result = docling.process_screenshot(financial_screenshot, symbol)
                        financial_analysis = result.get('markdown_structure')
    
                
                await browser.close()
                
                return {
                    'symbol': symbol,
                    'symbol_code': stock_data.get('symbol_code', symbol),
                    'exchange': stock_data.get('exchange', 'N/A'),
                    'company_name': stock_data.get('company_name', 'Không tìm thấy'),
                    'current_price': stock_data.get('current_price', 'N/A'),
                    'price_change': stock_data.get('price_change', 'N/A'),
                    'reference_price': stock_data.get('reference_price', 'N/A'),
                    'open_price': stock_data.get('open_price', 'N/A'),
                    'low_high': stock_data.get('low_high', 'N/A'),
                    'volume': stock_data.get('volume', 'N/A'),
                    'value': stock_data.get('value', 'N/A'),
                    'avg_volume_10d': stock_data.get('avg_volume_10d', 'N/A'),
                    'beta': stock_data.get('beta', 'N/A'),
                    'market_cap': stock_data.get('market_cap', 'N/A'),
                    'outstanding_shares': stock_data.get('outstanding_shares', 'N/A'),
                    'pe_ratio': stock_data.get('pe_ratio', 'N/A'),
                    'eps': stock_data.get('eps', 'N/A'),
                    'url': url,
                    'screenshot_filepath': screenshot_filepath,
                    'financial_screenshot': financial_screenshot,
                    'financial_analysis': financial_analysis,
                    'financial_success': financial_success,
                }
                
        except Exception as e:
            st.error(f"Lỗi khi sử dụng Browserless: {str(e)}")
            return None
    
    def save_screenshots(self, screenshot, symbol, page_type="general"):
        """
        Lưu screenshot vào thư mục và trích xuất báo cáo markdown bằng Docling
        
        Args:
            screenshot: Screenshot data
            symbol: Mã chứng khoán
            page_type: Loại trang
        """
        import os
        from datetime import datetime
        import streamlit as st
        
        # Tạo thư mục lưu trữ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{self.screenshots_dir}/{symbol}_{page_type}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{save_dir}/screenshot.png"
        with open(filename, "wb") as f:
            f.write(screenshot)

        return filename
    
    def display_screenshots(self, screenshot, symbol, page_type="general"):
        """
        Hiển thị screenshot trong Streamlit
        
        Args:
            screenshot: Screenshot data
            symbol: Mã chứng khoán
            page_type: Loại trang
        """
        import streamlit as st
        
        st.subheader(f"📸 Screenshot - {symbol} ({page_type})")
        st.image(screenshot, caption=f"Screenshot - {symbol}", use_column_width=True)
        st.info("🖼️ Screenshot toàn trang - hiển thị tất cả nội dung đã được scroll và load đầy đủ")


async def demo_scroll_capture():
    """Demo cơ chế scroll, chụp ảnh và phân tích chart với OpenAI"""
    import streamlit as st
    
    st.title("🚀 Demo Scraping và Phân tích Chart với OpenAI")
    st.write("Test cơ chế scroll, chụp ảnh và phân tích chart tự động với AI")
    
    # Tạo scraper
    scraper = FireantBrowserlessScraper()
    
    # Kiểm tra cấu hình
    # config_status = scraper.check_configuration() # This line is removed
    
    # Nút tạo file .env
    if st.button("📝 Tạo file .env"):
        scraper.create_env_file()
    
    # Input mã chứng khoán
    symbol = st.text_input("Nhập mã chứng khoán:", value="TCB")
    
    if st.button("🔍 Test Scroll và Chụp Ảnh"):
        if symbol:
            with st.spinner("Đang test cơ chế scroll và chụp ảnh..."):
                try:
                    # Test trang thông tin cơ bản
                    st.subheader("📊 Test Trang Thông Tin Cơ Bản")
                    result = await scraper.get_stock_info_browserless(symbol)
                    
                    if result and 'screenshot_filepath' in result:
                        st.success(f"✅ Đã chụp được screenshot và phân tích chart")
                        
                        # Hiển thị thông tin cơ bản
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Mã CK:** {result['symbol_code']}")
                            st.write(f"**Sàn:** {result['exchange']}")
                            st.write(f"**Tên công ty:** {result['company_name']}")
                            st.write(f"**Giá hiện tại:** {result['current_price']}")
                        with col2:
                            st.write(f"**Thay đổi:** {result['price_change']}")
                            st.write(f"**Khối lượng:** {result['volume']}")
                            st.write(f"**Tham chiếu:** {result['reference_price']}")
                            st.write(f"**URL:** {result['url']}")
                        
                        # Hiển thị thông tin file screenshot
                        st.info(f"📁 Screenshot đã lưu tại: {result['screenshot_filepath']}")
                        
                        # Hiển thị screenshot trang chính nếu file tồn tại
                        if result['screenshot_filepath'] and os.path.exists(result['screenshot_filepath']):
                            with open(result['screenshot_filepath'], "rb") as f:
                                screenshot_data = f.read()
                            scraper.display_screenshots(screenshot_data, symbol, "general")
                        
                        # Hiển thị kết quả phân tích chart
                        if result.get('financial_success') and result.get('financial_screenshot'):
                            st.subheader("📊 Screenshot Tab Tài chính")
                            scraper.display_screenshots(result['financial_screenshot'], symbol, "financial")
                        
                        # Lưu screenshot và hiển thị báo cáo markdown từ Docling
                        if st.button("💾 Lưu Screenshot"):
                            if result['screenshot_filepath'] and os.path.exists(result['screenshot_filepath']):
                                with open(result['screenshot_filepath'], "rb") as f:
                                    screenshot_data = f.read()
                                saved_file = scraper.save_screenshots(screenshot_data, symbol, "general")
                                st.success(f"✅ Đã lưu screenshot: {saved_file}")
                                markdown_path = os.path.join(os.path.dirname(saved_file), "report.md")
                                if os.path.exists(markdown_path):
                                    with open(markdown_path, "r", encoding="utf-8") as f:
                                        markdown_content = f.read()
                                    st.subheader("📄 Báo cáo tài chính (Docling)")
                                    st.markdown(markdown_content)
                                else:
                                    st.info("Không tìm thấy file báo cáo markdown từ Docling.")
                            else:
                                st.error("❌ Không tìm thấy file screenshot để lưu")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    asyncio.run(demo_scroll_capture()) 