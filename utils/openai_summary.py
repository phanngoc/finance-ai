"""
OpenAI utilities for news summarization and analysis
"""
import openai
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional


def create_news_summary_prompt(articles: List[Dict], symbol: str) -> str:
    """
    Create a detailed prompt for OpenAI to summarize news articles and analyze their impact on stock price
    
    Args:
        articles (List[Dict]): List of news articles with title, description, date, category
        symbol (str): Stock symbol
    
    Returns:
        str: Formatted prompt for OpenAI
    """
    
    # Format articles for the prompt
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"""
Tin tức {i}:
- Tiêu đề: {article.get('title', 'N/A')}
- Mô tả: {article.get('description', 'N/A')}
- Ngày đăng: {article.get('pub_date', 'N/A')}
- Danh mục: {article.get('category', 'N/A')}
- Chuyên mục: {article.get('section', 'N/A')}
- Điểm tin cậy: {article.get('confidence_score', 'N/A')}/5
"""
    
    prompt = f"""
Bạn là một chuyên gia phân tích tài chính và chứng khoán Việt Nam. Hãy phân tích 10 tin tức mới nhất về mã cổ phiếu {symbol} và tạo một báo cáo tổng quan chi tiết.

THÔNG TIN TIN TỨC:
{articles_text}

HÃY TẠO BÁO CÁO THEO CẤU TRÚC SAU:

## 📋 TÓM TẮT TỔNG QUAN
[Tóm tắt ngắn gọn 2-3 câu về tình hình chung của công ty/mã cổ phiếu dựa trên tin tức]

## 📈 CÁC YẾU TỐ TÍCH CỰC
[Liệt kê các tin tức/thông tin tích cực có thể làm tăng giá cổ phiếu]
- [Yếu tố 1]: [Giải thích tác động]
- [Yếu tố 2]: [Giải thích tác động]
- ...

## 📉 CÁC YẾU TỐ TIÊU CỰC  
[Liệt kê các tin tức/thông tin tiêu cực có thể làm giảm giá cổ phiếu]
- [Yếu tố 1]: [Giải thích tác động]
- [Yếu tố 2]: [Giải thích tác động]
- ...

## ⚖️ CÁC YẾU TỐ TRUNG TÍNH
[Liệt kê các thông tin có tính chất trung tính hoặc chưa rõ tác động]
- [Yếu tố 1]: [Giải thích]
- [Yếu tố 2]: [Giải thích]
- ...

## 🎯 ĐIỂM QUAN TRỌNG CẦN LÚU Ý
[3-5 điểm quan trọng nhất từ các tin tức, tập trung vào:]
1. Kết quả kinh doanh/tài chính
2. Hoạt động đầu tư/mở rộng
3. Chính sách/quy định mới
4. Thay đổi nhân sự cấp cao
5. Tình hình thị trường/ngành

## 📊 ĐÁNH GIÁ TÁC ĐỘNG TỔNG THỂ
- **Xu hướng ngắn hạn (1-2 tuần)**: [Tích cực/Tiêu cực/Trung tính] - [Lý do]
- **Xu hướng trung hạn (1-3 tháng)**: [Tích cực/Tiêu cực/Trung tính] - [Lý do]
- **Mức độ rủi ro**: [Thấp/Trung bình/Cao] - [Giải thích]
- **Khuyến nghị**: [Mua/Giữ/Bán/Chờ đợi] - [Lý do chi tiết]

## ⚠️ LƯU Ý QUAN TRỌNG
[Các yếu tố bên ngoài hoặc rủi ro cần theo dõi]

QUAN TRỌNG: 
- Chỉ phân tích dựa trên thông tin có trong tin tức được cung cấp
- Không đưa ra lời khuyên đầu tư cụ thể mà chỉ phân tích xu hướng
- Đề cập đến độ tin cậy của từng tin tức khi phân tích
- Sử dụng tiếng Việt tự nhiên và dễ hiểu
- Tập trung vào các yếu tố có thể định lượng được tác động đến giá cổ phiếu
"""
    
    return prompt


def get_openai_news_summary(articles: List[Dict], symbol: str, api_key: str) -> Optional[str]:
    """
    Get news summary from OpenAI API
    
    Args:
        articles (List[Dict]): List of news articles
        symbol (str): Stock symbol
        api_key (str): OpenAI API key
    
    Returns:
        Optional[str]: Summary text or None if error
    """
    try:
        # Set up OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Create prompt
        prompt = create_news_summary_prompt(articles, symbol)
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
            messages=[
                {
                    "role": "system", 
                    "content": "Bạn là một chuyên gia phân tích tài chính và chứng khoán Việt Nam có nhiều năm kinh nghiệm. Bạn có khả năng phân tích tin tức và đánh giá tác động đến giá cổ phiếu một cách chính xác và khách quan."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=3000,
            temperature=0.3,  # Lower temperature for more consistent analysis
            top_p=0.9
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Lỗi khi gọi OpenAI API: {str(e)}")
        return None


def format_articles_for_summary(news_df: pd.DataFrame, max_articles: int = 10) -> List[Dict]:
    """
    Format news DataFrame for OpenAI analysis
    
    Args:
        news_df (pd.DataFrame): News DataFrame
        max_articles (int): Maximum number of articles to include
    
    Returns:
        List[Dict]: Formatted articles for OpenAI
    """
    if news_df.empty:
        return []
    
    # Get the latest articles
    latest_news = news_df.head(max_articles)
    
    articles = []
    for _, row in latest_news.iterrows():
        articles.append({
            'title': row.get('title', 'N/A'),
            'description': row.get('description', 'N/A'),
            'pub_date': row.get('pub_date', 'N/A'),
            'category': row.get('category', 'N/A'),
            'section': row.get('section', 'N/A'),
            'confidence_score': row.get('confidence_score', 'N/A'),
            'link': row.get('link', '#')
        })
    
    return articles
