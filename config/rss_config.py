"""
Configuration file cho RSS Crawler
"""

RSS_SOURCES_CONFIG = {
    'vietstock': {
        'base_url': 'https://vietstock.vn',
        'rss_page_url': 'https://vietstock.vn/rss',
        'name': 'VietStock',
        'timeout': 30,
        'max_retries': 3,
        'delay_between_requests': 1.0,
        'section_mapping': {
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
    },
    
    'stockbiz': {
        'base_url': 'http://en.stockbiz.vn',
        'rss_page_url': 'http://en.stockbiz.vn/Rss.aspx',
        'name': 'StockBiz',
        'timeout': 30,
        'max_retries': 3,
        'delay_between_requests': 1.5,
        'section_mapping': {
            'market': 'Thị trường',
            'stock': 'Cổ phiếu',
            'finance': 'Tài chính',
            'economy': 'Kinh tế',
            'business': 'Kinh doanh',
            'investment': 'Đầu tư',
            'news': 'Tin tức',
            'analysis': 'Phân tích'
        }
    }
}

CRAWLER_SETTINGS = {
    'max_workers': 5,
    'output_dir': 'data',
    'merged_filename': 'merged_rss_data.csv',
    'feeds_info_filename': 'all_rss_feeds.csv',
    'log_level': 'INFO'
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]
