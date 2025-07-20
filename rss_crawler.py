"""
RSS Crawler đã được tối ưu và sửa lỗi
Hỗ trợ VietStock và StockBiz
"""

import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RSSConfig:
    """Configuration cho RSS Crawler"""
    base_url: str
    rss_page_url: str
    name: str
    headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 1.0

@dataclass
class Article:
    """Data class cho bài viết"""
    title: str
    link: str
    description: str
    pub_date: str
    category: str
    section: str
    source: str
    crawl_time: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'link': self.link,
            'description': self.description,
            'pub_date': self.pub_date,
            'category': self.category,
            'section': self.section,
            'source': self.source,
            'crawl_time': self.crawl_time
        }

class BaseRSSCrawler(ABC):
    """Base class cho RSS Crawler"""
    
    def __init__(self, config: RSSConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        self.rss_feeds: List[Dict[str, str]] = []
        self.articles: List[Article] = []
    
    @abstractmethod
    def find_rss_links(self) -> List[Dict[str, str]]:
        """Tìm tất cả RSS links từ trang web"""
        pass
    
    @abstractmethod
    def get_section_from_url(self, url: str) -> str:
        """Lấy section từ URL"""
        pass
    
    def clean_description(self, description: str) -> str:
        """Làm sạch mô tả bài viết"""
        if not description:
            return ""
        
        # Bỏ HTML tags
        description = re.sub(r'<[^>]+>', '', description)
        # Bỏ khoảng trắng thừa
        description = ' '.join(description.split())
        return description.strip()
    
    def parse_date(self, date_str: str) -> str:
        """Chuẩn hóa định dạng ngày"""
        if not date_str:
            return ""
        
        try:
            # Thử parse với feedparser
            import email.utils
            parsed_time = email.utils.parsedate_tz(date_str)
            if parsed_time:
                dt = datetime(*parsed_time[:6])
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
        
        return date_str
    
    def parse_rss_feed(self, rss_url: str, category: str, section: str) -> List[Article]:
        """Parse một RSS feed với retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Crawling RSS: {category} (attempt {attempt + 1})")
                
                # Sử dụng requests để get RSS với timeout
                response = self.session.get(rss_url, timeout=self.config.timeout)
                response.raise_for_status()
                
                # Kiểm tra content type
                content_type = response.headers.get('content-type', '').lower()
                if 'xml' not in content_type and 'rss' not in content_type and 'text' not in content_type:
                    logger.warning(f"Unexpected content type for {category}: {content_type}")
                
                # Parse với feedparser
                feed = feedparser.parse(response.content)
                
                if not feed.entries:
                    logger.warning(f"No articles found in RSS: {category}")
                    return []
                
                articles = []
                for entry in feed.entries:
                    try:
                        title = getattr(entry, 'title', '').strip()
                        link = getattr(entry, 'link', '').strip()
                        description = getattr(entry, 'description', '') or getattr(entry, 'summary', '')
                        pub_date = getattr(entry, 'published', '') or getattr(entry, 'updated', '')
                        
                        # Validate required fields
                        if not title or not link:
                            continue
                        
                        article = Article(
                            title=title,
                            link=link,
                            description=self.clean_description(description),
                            pub_date=self.parse_date(pub_date),
                            category=category,
                            section=section,
                            source=self.config.name,
                            crawl_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                        
                        articles.append(article)
                        
                    except Exception as e:
                        logger.error(f"Error parsing entry in RSS {category}: {e}")
                        continue
                
                logger.info(f"Successfully crawled {len(articles)} articles from {category}")
                return articles
                
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    logger.error(f"RSS feed not found (404): {category} - {rss_url}")
                    return []  # Không retry cho 404
                else:
                    logger.error(f"HTTP error for RSS {category} (attempt {attempt + 1}): {e}")
            except requests.exceptions.Timeout:
                logger.error(f"Timeout for RSS {category} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Error crawling RSS {category} (attempt {attempt + 1}): {e}")
            
            # Retry logic
            if attempt < self.config.max_retries - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)  # Exponential backoff
            else:
                logger.error(f"Failed to crawl RSS {category} after {self.config.max_retries} attempts")
                
        return []
    
    def crawl_all_feeds(self, max_workers: int = 3) -> List[Article]:
        """Crawl tất cả RSS feeds với threading"""
        if not self.rss_feeds:
            logger.info("Finding RSS feeds...")
            self.find_rss_links()
        
        if not self.rss_feeds:
            logger.error("No RSS feeds found!")
            return []
        
        all_articles = []
        
        # Crawl tuần tự để tránh spam server
        for i, feed in enumerate(self.rss_feeds):
            try:
                logger.info(f"[{i+1}/{len(self.rss_feeds)}] Processing: {feed['category']}")
                articles = self.parse_rss_feed(feed['url'], feed['category'], feed['section'])
                all_articles.extend(articles)
                
                # Nghỉ một chút để tránh spam server
                time.sleep(self.config.delay_between_requests)
                
            except Exception as e:
                logger.error(f"Error in feed {feed['category']}: {e}")
        
        self.articles = all_articles
        logger.info(f"Total crawled {len(all_articles)} articles from all RSS feeds")
        return all_articles

class VietStockRSSCrawler(BaseRSSCrawler):
    """RSS Crawler cho VietStock"""
    
    def find_rss_links(self) -> List[Dict[str, str]]:
        """Tìm tất cả các link RSS từ trang RSS của VietStock"""
        try:
            response = self.session.get(self.config.rss_page_url, timeout=self.config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            rss_links = []
            for link in soup.find_all('a', href=True):
                try:
                    href = link['href'] if link.has_attr('href') else ''
                    href = str(href) if href else ''
                    if href and ('.rss' in href or '.ashx' in href):
                        full_url = urljoin(self.config.base_url, href)
                        category = link.get_text(strip=True)
                        
                        if category and full_url:  # Validate
                            rss_links.append({
                                'url': full_url,
                                'category': category,
                                'section': self.get_section_from_url(href)
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing link: {e}")
                    continue
            
            self.rss_feeds = rss_links
            logger.info(f"Found {len(rss_links)} RSS feeds from VietStock")
            return rss_links
            
        except Exception as e:
            logger.error(f"Error finding RSS links: {e}")
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
        
        url_lower = url.lower()
        for key, value in sections.items():
            if key in url_lower:
                return value
        return 'Khác'

class StockBizRSSCrawler(BaseRSSCrawler):
    """RSS Crawler cho StockBiz"""
    
    def find_rss_links(self) -> List[Dict[str, str]]:
        """Tìm tất cả các link RSS từ trang RSS của VietStock"""
        try:
            response = self.session.get(self.config.rss_page_url, timeout=self.config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            rss_links = []
            for link in soup.find_all('a', href=True):
                try:
                    href = link['href'] if link.has_attr('href') else ''
                    href = str(href) if href else ''
                    if href and ('.rss' in href or '.ashx' in href):
                        full_url = urljoin(self.config.base_url, href)
                        category = link.get_text(strip=True)
                        
                        if category and full_url:  # Validate
                            rss_links.append({
                                'url': full_url,
                                'category': category,
                                'section': self.get_section_from_url(href)
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing link: {e}")
                    continue
            
            self.rss_feeds = rss_links
            logger.info(f"Found {len(rss_links)} RSS feeds from VietStock")
            return rss_links
            
        except Exception as e:
            logger.error(f"Error finding RSS links: {e}")
            return []
    
    def get_section_from_url(self, url: str) -> str:
        """Lấy section từ URL RSS"""
        sections = {
            'market': 'Thị trường',
            'stock': 'Cổ phiếu', 
            'finance': 'Tài chính',
            'economy': 'Kinh tế',
            'business': 'Kinh doanh',
            'investment': 'Đầu tư',
            'news': 'Tin tức',
            'analysis': 'Phân tích'
        }
        
        url_lower = url.lower()
        for key, value in sections.items():
            if key in url_lower:
                return value
        return 'Tin tức'

class RSSCrawlerFactory:
    """Factory để tạo các RSS Crawler"""
    
    @staticmethod
    def create_crawler(source: str) -> BaseRSSCrawler:
        """Tạo crawler based on source"""
        if source.lower() == 'vietstock':
            config = RSSConfig(
                base_url="https://vietstock.vn",
                rss_page_url="https://vietstock.vn/rss",
                name="VietStock"
            )
            return VietStockRSSCrawler(config)
        
        elif source.lower() == 'stockbiz':
            config = RSSConfig(
                base_url="http://en.stockbiz.vn",
                rss_page_url="http://en.stockbiz.vn/Rss.aspx",
                name="StockBiz",
                timeout=15,
                max_retries=2
            )
            return StockBizRSSCrawler(config)
        
        else:
            raise ValueError(f"Unsupported source: {source}")

class MultiSourceRSSCrawler:
    """Crawler tổng hợp từ nhiều nguồn"""
    
    def __init__(self, sources: List[str]):
        self.sources = sources
        self.crawlers = [RSSCrawlerFactory.create_crawler(source) for source in sources]
        self.all_articles: List[Article] = []
    
    def crawl_all_sources(self) -> List[Article]:
        """Crawl từ tất cả các nguồn"""
        all_articles = []
        
        for crawler in self.crawlers:
            try:
                logger.info(f"Starting crawl for {crawler.config.name}")
                articles = crawler.crawl_all_feeds()
                all_articles.extend(articles)
                logger.info(f"Completed {crawler.config.name}: {len(articles)} articles")
            except Exception as e:
                logger.error(f"Error crawling {crawler.config.name}: {e}")
        
        self.all_articles = all_articles
        return all_articles
    
    def save_to_csv(self, filename: str = 'merged_rss_data.csv') -> None:
        """Lưu tất cả dữ liệu vào file CSV"""
        if not self.all_articles:
            logger.warning("No articles to save!")
            return
        
        try:
            # Convert articles to dict
            articles_data = [article.to_dict() for article in self.all_articles]
            df = pd.DataFrame(articles_data)
            
            # Remove duplicates based on link
            initial_count = len(df)
            df = df.drop_duplicates(subset=['link'], keep='first')
            duplicate_count = initial_count - len(df)
            
            if duplicate_count > 0:
                logger.info(f"Removed {duplicate_count} duplicate articles")
            
            # Sort by crawl_time
            df = df.sort_values('crawl_time', ascending=False)
            
            # Save to CSV
            df.to_csv('./data/rss_news/' + filename, index=False, encoding='utf-8')
            logger.info(f"Saved {len(df)} articles to {filename}")
            
            # Display statistics
            self._display_statistics(df)
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def _display_statistics(self, df: pd.DataFrame) -> None:
        """Hiển thị thống kê"""
        print(f"\n📊 THỐNG KÊ TỔNG HỢP:")
        print(f"Tổng số bài viết: {len(df)}")
        
        print(f"\n📈 Theo nguồn:")
        source_stats = df['source'].value_counts()
        for source, count in source_stats.items():
            print(f"- {source}: {count} bài viết")
        
        print(f"\n📂 Theo chuyên mục:")
        section_stats = df['section'].value_counts()
        for section, count in section_stats.head(10).items():
            print(f"- {section}: {count} bài viết")
        
        print(f"\n🗂️ Theo danh mục:")
        category_stats = df['category'].value_counts()
        for category, count in category_stats.head(10).items():
            print(f"- {category}: {count} bài viết")
    
    def save_feeds_info(self, filename: str = 'all_rss_feeds.csv') -> None:
        """Lưu thông tin tất cả RSS feeds"""
        all_feeds = []
        
        for crawler in self.crawlers:
            for feed in crawler.rss_feeds:
                feed_info = feed.copy()
                feed_info['source'] = crawler.config.name
                all_feeds.append(feed_info)
        
        if all_feeds:
            df = pd.DataFrame(all_feeds)
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Saved {len(all_feeds)} RSS feeds info to {filename}")

def main():
    """Hàm main để chạy crawler tổng hợp"""
    print("🚀 Bắt đầu crawl RSS feeds từ nhiều nguồn...")
    
    # Khởi tạo crawler với nhiều nguồn
    sources = ['stockbiz', 'vietstock']
    crawler = MultiSourceRSSCrawler(sources)
    
    try:
        # Crawl từ tất cả nguồn
        print("\n📰 Đang crawl từ tất cả nguồn...")
        articles = crawler.crawl_all_sources()
        
        if not articles:
            print("❌ Không crawl được bài viết nào!")
            return
        
        # Lưu thông tin RSS feeds
        print("\n💾 Lưu thông tin RSS feeds...")
        crawler.save_feeds_info()
        
        # Lưu dữ liệu merge vào CSV
        print("\n💾 Lưu dữ liệu merge vào CSV...")
        crawler.save_to_csv()
        
        print("\n✅ Hoàn thành crawl RSS feeds từ tất cả nguồn!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Đã dừng crawling...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
