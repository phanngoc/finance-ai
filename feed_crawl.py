import requests
import feedparser
import pandas as pd
from datetime import datetime
import time
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

class VietStockRSSCrawler:
    def __init__(self):
        self.base_url = "https://vietstock.vn"
        self.rss_page_url = "https://vietstock.vn/rss"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.rss_feeds: List[Dict[str, str]] = []
        self.articles: List[Dict[str, Any]] = []
        
    def find_rss_links(self) -> List[Dict[str, str]]:
        """Tìm tất cả các link RSS từ trang RSS của VietStock"""
        try:
            response = requests.get(self.rss_page_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm tất cả các link có chứa '.rss'
            rss_links = []
            for link in soup.find_all('a', href=True):
                try:
                    # Sử dụng dictionary-style access
                    href = link['href'] if link.has_attr('href') else ''
                    if href and '.rss' in href:
                        # Chuyển đổi link tương đối thành link tuyệt đối
                        full_url = urljoin(self.base_url, href)
                        category = link.get_text(strip=True)
                        rss_links.append({
                            'url': full_url,
                            'category': category,
                            'section': self.get_section_from_url(href)
                        })
                except Exception as e:
                    print(f"Lỗi khi xử lý link: {e}")
                    continue
            
            self.rss_feeds = rss_links
            print(f"Đã tìm thấy {len(rss_links)} RSS feeds:")
            for feed in rss_links:
                print(f"- {feed['category']}: {feed['url']}")
            
            return rss_links
            
        except Exception as e:
            print(f"Lỗi khi tìm RSS links: {e}")
            return []
    
    def get_section_from_url(self, url: str) -> str:
        """Lấy section từ URL RSS"""
        sections = {
            'chung-khoan': 'Chứng khoán',
            'doanh-nghiep': 'Doanh nghiệp',
            'bat-dong-san': 'Bất động sản',
            'hang-hoa': 'Hàng hóa',
            'tai-chinh': 'Tài chính',
            'kinh-te': 'Kinh tế',
            'the-gioi': 'Thế giới',
            'dong-duong': 'Đông Dương',
            'tai-chinh-ca-nhan': 'Tài chính cá nhân',
            'nhan-dinh-phan-tich': 'Phân tích'
        }
        
        for key, value in sections.items():
            if key in url:
                return value
        return 'Khác'
    
    def parse_rss_feed(self, rss_url: str, category: str, section: str) -> List[Dict[str, Any]]:
        """Parse một RSS feed và trích xuất thông tin bài viết"""
        try:
            print(f"Đang crawl RSS: {category}")
            
            # Sử dụng feedparser để parse RSS
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                print(f"Không tìm thấy bài viết nào trong RSS: {category}")
                return []
            
            feed_articles = []
            for entry in feed.entries:
                try:
                    title = getattr(entry, 'title', '')
                    link = getattr(entry, 'link', '')
                    description = getattr(entry, 'description', '') or getattr(entry, 'summary', '')
                    pub_date = getattr(entry, 'published', '') or getattr(entry, 'updated', '')
                    
                    # Làm sạch description (bỏ HTML tags)
                    if description:
                        description = re.sub(r'<[^>]+>', '', description).strip()
                    
                    article = {
                        'title': title.strip() if title else '',
                        'link': link.strip() if link else '',
                        'description': description,
                        'pub_date': pub_date,
                        'category': category,
                        'section': section,
                        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    feed_articles.append(article)
                    
                except Exception as e:
                    print(f"Lỗi khi parse entry trong RSS {category}: {e}")
                    continue
            
            print(f"Đã crawl {len(feed_articles)} bài viết từ {category}")
            return feed_articles
            
        except Exception as e:
            print(f"Lỗi khi crawl RSS {category}: {e}")
            return []
    
    def crawl_all_feeds(self) -> List[Dict[str, Any]]:
        """Crawl tất cả RSS feeds"""
        if not self.rss_feeds:
            print("Chưa tìm thấy RSS feeds. Đang tìm kiếm...")
            self.find_rss_links()
        
        all_articles = []
        
        for i, feed in enumerate(self.rss_feeds):
            try:
                print(f"[{i+1}/{len(self.rss_feeds)}] Crawling: {feed['category']}")
                articles = self.parse_rss_feed(feed['url'], feed['category'], feed['section'])
                all_articles.extend(articles)
                
                # Nghỉ một chút để tránh spam server
                time.sleep(1)
                
            except Exception as e:
                print(f"Lỗi khi crawl feed {feed['category']}: {e}")
                continue
        
        self.articles = all_articles
        print(f"\nTổng cộng đã crawl {len(all_articles)} bài viết từ tất cả RSS feeds")
        return all_articles
    
    def save_to_csv(self, filename: str = 'vietstock_rss_data.csv') -> None:
        """Lưu dữ liệu vào file CSV"""
        if not self.articles:
            print("Không có dữ liệu để lưu!")
            return
        
        try:
            df = pd.DataFrame(self.articles)
            
            # Sắp xếp theo thời gian publish (nếu có)
            if 'pub_date' in df.columns and not df['pub_date'].isna().all():
                df = df.sort_values('pub_date', ascending=False)
            
            # Lưu vào CSV
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Đã lưu {len(self.articles)} bài viết vào file {filename}")
            
            # Hiển thị thống kê
            print("\nThống kê theo danh mục:")
            category_stats = df['category'].value_counts()
            for category, count in category_stats.items():
                print(f"- {category}: {count} bài viết")
            
            print("\nThống kê theo chuyên mục:")
            section_stats = df['section'].value_counts()
            for section, count in section_stats.items():
                print(f"- {section}: {count} bài viết")
                
        except Exception as e:
            print(f"Lỗi khi lưu file CSV: {e}")
    
    def save_rss_feeds_info(self, filename: str = 'vietstock_rss_feeds.csv') -> None:
        """Lưu thông tin các RSS feeds vào file CSV"""
        if not self.rss_feeds:
            print("Không có thông tin RSS feeds để lưu!")
            return
        
        try:
            df = pd.DataFrame(self.rss_feeds)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"Đã lưu thông tin {len(self.rss_feeds)} RSS feeds vào file {filename}")
            
        except Exception as e:
            print(f"Lỗi khi lưu file RSS feeds: {e}")

def main():
    """Hàm main để chạy crawler"""
    print("🚀 Bắt đầu crawl RSS feeds từ VietStock...")
    
    crawler = VietStockRSSCrawler()
    
    # Bước 1: Tìm tất cả RSS links
    print("\n📡 Đang tìm kiếm RSS feeds...")
    rss_links = crawler.find_rss_links()
    
    if not rss_links:
        print("❌ Không tìm thấy RSS feeds nào!")
        return
    
    # Bước 2: Lưu thông tin RSS feeds
    print("\n💾 Lưu thông tin RSS feeds...")
    crawler.save_rss_feeds_info()
    
    # Bước 3: Crawl tất cả RSS feeds
    print("\n📰 Đang crawl tất cả RSS feeds...")
    articles = crawler.crawl_all_feeds()
    
    if not articles:
        print("❌ Không crawl được bài viết nào!")
        return
    
    # Bước 4: Lưu dữ liệu vào CSV
    print("\n💾 Lưu dữ liệu vào CSV...")
    crawler.save_to_csv()
    
    print("\n✅ Hoàn thành crawl RSS feeds từ VietStock!")

if __name__ == "__main__":
    main()