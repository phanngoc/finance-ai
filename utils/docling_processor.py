"""
Module xử lý screenshot với Docling để trích xuất thông tin tài chính
Tích hợp với Fireant Browserless Scraper
"""

import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from PIL import Image
import io
import base64

from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument


class DoclingProcessor:
    """Xử lý screenshot với Docling để trích xuất thông tin tài chính"""
    
    def __init__(self):
        self.converter = DocumentConverter()
        self.financial_keywords = {
            'price': ['giá', 'price', 'thị giá', 'giá hiện tại', 'current price'],
            'change': ['thay đổi', 'change', 'tăng', 'giảm', 'delta'],
            'volume': ['khối lượng', 'volume', 'vol', 'kltb'],
            'market_cap': ['thị giá vốn', 'market cap', 'vốn hóa'],
            'pe_ratio': ['p/e', 'pe', 'price to earnings'],
            'eps': ['eps', 'earnings per share', 'lợi nhuận trên cổ phiếu'],
            'beta': ['beta', 'hệ số beta'],
            'reference': ['tham chiếu', 'reference', 'ref'],
            'open': ['mở cửa', 'open', 'giá mở cửa'],
            'high_low': ['thấp - cao', 'high - low', 'cao nhất', 'thấp nhất'],
            'value': ['giá trị', 'value', 'tổng giá trị'],
            'outstanding': ['cplh', 'outstanding', 'số lượng cổ phiếu']
        }
    
    def process_screenshot(self, screenshot_bytes: bytes, symbol: str) -> Dict[str, Any]:
        """
        Xử lý screenshot với Docling để trích xuất thông tin
        
        Args:
            screenshot_bytes: Dữ liệu screenshot dưới dạng bytes
            symbol: Mã chứng khoán
            
        Returns:
            Dict chứa thông tin được trích xuất
        """
        
        try:
            # Chuyển bytes thành PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            # Lưu tạm thời để Docling xử lý
            temp_path = f"temp_screenshot_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image.save(temp_path)
            
            # Sử dụng Docling để xử lý
            result = self.converter.convert(temp_path)
            
            # Trích xuất text từ DoclingDocument
            extracted_text = result.document.export_to_markdown()
            print("extracted_text", extracted_text)
            # Xử lý và phân tích text
            financial_data = self._extract_financial_data(extracted_text, symbol)
            print('financial_data', financial_data)
            # Tạo cấu trúc markdown cho chatbot
            markdown_structure = self._create_markdown_structure(financial_data, symbol)
            print("markdown_structure", markdown_structure)
            # Dọn dẹp file tạm
            
            return {
                'extracted_text': extracted_text,
                'financial_data': financial_data,
                'markdown_structure': markdown_structure,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý screenshot với Docling: {str(e)}")
            return self._fallback_extraction(symbol)
    
    def _extract_financial_data(self, text: str, symbol: str) -> Dict[str, Any]:
        """
        Trích xuất thông tin tài chính từ text được Docling xử lý
        
        Args:
            text: Text được trích xuất từ Docling
            symbol: Mã chứng khoán
            
        Returns:
            Dict chứa thông tin tài chính
        """
        financial_data = {
            'symbol': symbol,
            'company_name': '',
            'current_price': '',
            'price_change': '',
            'reference_price': '',
            'open_price': '',
            'high_low': '',
            'volume': '',
            'value': '',
            'market_cap': '',
            'pe_ratio': '',
            'eps': '',
            'beta': '',
            'outstanding_shares': '',
            'avg_volume_10d': ''
        }
        
        # Tìm tên công ty
        company_patterns = [
            r'([A-Z]{3,})\s*[-–]\s*(.+)',
            r'Công ty\s+(.+)',
            r'(.+)\s+\([A-Z]{3,}\)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['company_name'] = match.group(1) if 'Công ty' not in pattern else match.group(1)
                break
        
        # Tìm giá hiện tại
        price_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:VND|₫|đồng)',
            r'Giá[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\(hiện tại\)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['current_price'] = match.group(1)
                break
        
        # Tìm thay đổi giá
        change_patterns = [
            r'([+-]\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:VND|₫|%)',
            r'Thay đổi[:\s]*([+-]\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'([+-]\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\([+-]\d+\.?\d*%\)'
        ]
        
        for pattern in change_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['price_change'] = match.group(1)
                break
        
        # Tìm khối lượng
        volume_patterns = [
            r'Khối lượng[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Volume[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:cổ phiếu|shares)'
        ]
        
        for pattern in volume_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['volume'] = match.group(1)
                break
        
        # Tìm P/E ratio
        pe_patterns = [
            r'P/E[:\s]*(\d+\.?\d*)',
            r'Price/Earnings[:\s]*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*\(P/E\)'
        ]
        
        for pattern in pe_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['pe_ratio'] = match.group(1)
                break
        
        # Tìm EPS
        eps_patterns = [
            r'EPS[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Earnings per share[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\(EPS\)'
        ]
        
        for pattern in eps_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['eps'] = match.group(1)
                break
        
        # Tìm Beta
        beta_patterns = [
            r'Beta[:\s]*(\d+\.?\d*)',
            r'Hệ số Beta[:\s]*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*\(Beta\)'
        ]
        
        for pattern in beta_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['beta'] = match.group(1)
                break
        
        # Tìm thị giá vốn
        market_cap_patterns = [
            r'Thị giá vốn[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Market cap[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Vốn hóa[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in market_cap_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data['market_cap'] = match.group(1)
                break
        
        return financial_data
    
    def _create_markdown_structure(self, financial_data: Dict[str, Any], symbol: str) -> str:
        """
        Tạo cấu trúc markdown phù hợp cho chatbot AI phân tích
        
        Args:
            financial_data: Dữ liệu tài chính đã trích xuất
            symbol: Mã chứng khoán
            
        Returns:
            String markdown có cấu trúc
        """
        markdown = f"""# Phân Tích Tài Chính - {symbol}

## 📊 Thông Tin Cơ Bản

| Chỉ Số | Giá Trị |
|--------|---------|
| **Mã Chứng Khoán** | {financial_data['symbol']} |
| **Tên Công Ty** | {financial_data['company_name'] or 'Chưa xác định'} |
| **Giá Hiện Tại** | {financial_data['current_price'] or 'N/A'} |
| **Thay Đổi Giá** | {financial_data['price_change'] or 'N/A'} |

## 📈 Chỉ Số Định Giá

| Chỉ Số | Giá Trị | Đánh Giá |
|--------|---------|----------|
| **P/E Ratio** | {financial_data['pe_ratio'] or 'N/A'} | {self._evaluate_pe_ratio(financial_data['pe_ratio'])} |
| **EPS** | {financial_data['eps'] or 'N/A'} | {self._evaluate_eps(financial_data['eps'])} |
| **Beta** | {financial_data['beta'] or 'N/A'} | {self._evaluate_beta(financial_data['beta'])} |

## 💰 Thông Tin Giao Dịch

| Chỉ Số | Giá Trị |
|--------|---------|
| **Khối Lượng** | {financial_data['volume'] or 'N/A'} |
| **Thị Giá Vốn** | {financial_data['market_cap'] or 'N/A'} |
| **Số Lượng CPLH** | {financial_data['outstanding_shares'] or 'N/A'} |

## 🔍 Phân Tích Nhanh

### Đánh Giá Định Giá
- **P/E Ratio {financial_data['pe_ratio'] or 'N/A'}**: {self._get_pe_analysis(financial_data['pe_ratio'])}
- **EPS {financial_data['eps'] or 'N/A'}**: {self._get_eps_analysis(financial_data['eps'])}

### Đánh Giá Rủi Ro
- **Beta {financial_data['beta'] or 'N/A'}**: {self._get_beta_analysis(financial_data['beta'])}

### Đánh Giá Thanh Khoản
- **Khối Lượng {financial_data['volume'] or 'N/A'}**: {self._get_volume_analysis(financial_data['volume'])}

## 📋 Khuyến Nghị Cho Chatbot AI

### Cấu Trúc Dữ Liệu JSON
```json
{{
  "symbol": "{financial_data['symbol']}",
  "company_name": "{financial_data['company_name'] or 'N/A'}",
  "current_price": "{financial_data['current_price'] or 'N/A'}",
  "price_change": "{financial_data['price_change'] or 'N/A'}",
  "pe_ratio": "{financial_data['pe_ratio'] or 'N/A'}",
  "eps": "{financial_data['eps'] or 'N/A'}",
  "beta": "{financial_data['beta'] or 'N/A'}",
  "volume": "{financial_data['volume'] or 'N/A'}",
  "market_cap": "{financial_data['market_cap'] or 'N/A'}",
  "analysis": {{
    "valuation": "{self._get_pe_analysis(financial_data['pe_ratio'])}",
    "risk": "{self._get_beta_analysis(financial_data['beta'])}",
    "liquidity": "{self._get_volume_analysis(financial_data['volume'])}"
  }}
}}
```

### Câu Hỏi Gợi Ý Cho Chatbot
1. **Phân tích định giá**: {symbol} có đang được định giá hợp lý không?
2. **Đánh giá rủi ro**: Mức độ biến động của {symbol} so với thị trường?
3. **Khuyến nghị đầu tư**: Có nên đầu tư vào {symbol} không?
4. **So sánh ngành**: {symbol} so sánh với các công ty cùng ngành?
5. **Xu hướng giá**: Dự báo xu hướng giá của {symbol}?

---
*Dữ liệu được trích xuất bằng Docling từ screenshot Fireant - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return markdown
    
    def _evaluate_pe_ratio(self, pe_ratio: str) -> str:
        """Đánh giá P/E ratio"""
        if not pe_ratio or pe_ratio == 'N/A':
            return 'Không đủ dữ liệu'
        
        try:
            pe = float(pe_ratio.replace(',', ''))
            if pe < 10:
                return 'Thấp - Có thể undervalued'
            elif pe < 20:
                return 'Trung bình - Định giá hợp lý'
            else:
                return 'Cao - Có thể overvalued'
        except:
            return 'Không thể đánh giá'
    
    def _evaluate_eps(self, eps: str) -> str:
        """Đánh giá EPS"""
        if not eps or eps == 'N/A':
            return 'Không đủ dữ liệu'
        
        try:
            eps_val = float(eps.replace(',', ''))
            if eps_val > 0:
                return 'Tích cực - Có lợi nhuận'
            else:
                return 'Tiêu cực - Đang lỗ'
        except:
            return 'Không thể đánh giá'
    
    def _evaluate_beta(self, beta: str) -> str:
        """Đánh giá Beta"""
        if not beta or beta == 'N/A':
            return 'Không đủ dữ liệu'
        
        try:
            beta_val = float(beta.replace(',', ''))
            if beta_val < 0.8:
                return 'Thấp - Ít biến động'
            elif beta_val < 1.2:
                return 'Trung bình - Biến động bình thường'
            else:
                return 'Cao - Biến động mạnh'
        except:
            return 'Không thể đánh giá'
    
    def _get_pe_analysis(self, pe_ratio: str) -> str:
        """Phân tích P/E ratio"""
        if not pe_ratio or pe_ratio == 'N/A':
            return 'Không đủ dữ liệu để phân tích'
        
        try:
            pe = float(pe_ratio.replace(',', ''))
            if pe < 10:
                return 'P/E thấp cho thấy cổ phiếu có thể đang bị định giá thấp'
            elif pe < 20:
                return 'P/E ở mức trung bình, định giá tương đối hợp lý'
            else:
                return 'P/E cao có thể cho thấy cổ phiếu đang được định giá cao'
        except:
            return 'Không thể phân tích P/E ratio'
    
    def _get_eps_analysis(self, eps: str) -> str:
        """Phân tích EPS"""
        if not eps or eps == 'N/A':
            return 'Không đủ dữ liệu để phân tích'
        
        try:
            eps_val = float(eps.replace(',', ''))
            if eps_val > 0:
                return 'EPS dương cho thấy công ty đang có lợi nhuận'
            else:
                return 'EPS âm cho thấy công ty đang thua lỗ'
        except:
            return 'Không thể phân tích EPS'
    
    def _get_beta_analysis(self, beta: str) -> str:
        """Phân tích Beta"""
        if not beta or beta == 'N/A':
            return 'Không đủ dữ liệu để phân tích'
        
        try:
            beta_val = float(beta.replace(',', ''))
            if beta_val < 0.8:
                return 'Beta thấp cho thấy cổ phiếu ít biến động hơn thị trường'
            elif beta_val < 1.2:
                return 'Beta trung bình, biến động tương đương thị trường'
            else:
                return 'Beta cao cho thấy cổ phiếu biến động mạnh hơn thị trường'
        except:
            return 'Không thể phân tích Beta'
    
    def _get_volume_analysis(self, volume: str) -> str:
        """Phân tích khối lượng"""
        if not volume or volume == 'N/A':
            return 'Không đủ dữ liệu để phân tích'
        
        return 'Khối lượng giao dịch cho thấy mức độ thanh khoản của cổ phiếu'
    
    def _fallback_extraction(self, symbol: str) -> Dict[str, Any]:
        """Fallback khi không có Docling"""
        return {
            'extracted_text': f'Không thể trích xuất thông tin từ screenshot cho {symbol}',
            'financial_data': {
                'symbol': symbol,
                'company_name': 'Không xác định',
                'current_price': 'N/A',
                'price_change': 'N/A',
                'reference_price': 'N/A',
                'open_price': 'N/A',
                'high_low': 'N/A',
                'volume': 'N/A',
                'value': 'N/A',
                'market_cap': 'N/A',
                'pe_ratio': 'N/A',
                'eps': 'N/A',
                'beta': 'N/A',
                'outstanding_shares': 'N/A',
                'avg_volume_10d': 'N/A'
            },
            'markdown_structure': f'# Không thể trích xuất thông tin cho {symbol}\n\nDocling chưa được cài đặt hoặc có lỗi xảy ra.',
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        } 