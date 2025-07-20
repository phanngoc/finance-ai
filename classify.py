import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from vnstock import Listing
import warnings
warnings.filterwarnings('ignore')

class ImprovedStockNewsClassifier:
    def __init__(self):
        """Initialize the improved classifier"""
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.lemmatizer = WordNetLemmatizer()
        self.symbols_df = None
        self.symbol_keywords = {}
        self.vietnamese_stopwords = set([
            'là', 'của', 'và', 'có', 'một', 'các', 'được', 'trong', 'cho', 'với', 
            'từ', 'trên', 'theo', 'về', 'đã', 'sẽ', 'này', 'đó', 'để', 'khi',
            'nhưng', 'nếu', 'mà', 'hay', 'hoặc', 'thì', 'do', 'vì', 'như',
            'cũng', 'đều', 'phải', 'nên', 'bằng', 'tại', 'sau', 'trước'
        ])
        
        # Download required NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def load_stock_symbols(self):
        """Load stock symbols from vnstock"""
        try:
            listing = Listing()
            self.symbols_df = listing.all_symbols()
            print(f"Loaded {len(self.symbols_df)} stock symbols")
            
            # Create keyword mapping for each symbol
            for _, row in self.symbols_df.iterrows():
                symbol = row['symbol']
                company_name = str(row['organ_name']) if pd.notna(row['organ_name']) else ""
                
                # Extract keywords from company name
                keywords = self._extract_keywords_from_company_name(company_name)
                self.symbol_keywords[symbol] = {
                    'keywords': keywords,
                    'company_name': company_name
                }
                
        except Exception as e:
            print(f"Error loading stock symbols: {e}")
            # Use fallback data
            self._load_fallback_symbols()
    
    def _load_fallback_symbols(self):
        """Load fallback symbol data"""
        fallback_data = {
            'VIC': 'Tập đoàn Vingroup',
            'VNM': 'Công ty Sữa Việt Nam Vinamilk', 
            'HPG': 'Công ty Hòa Phát Group',
            'TCB': 'Ngân hàng Techcombank',
            'CTG': 'Ngân hàng Vietinbank',
            'BID': 'Ngân hàng BIDV',
            'MSN': 'Tập đoàn Masan',
            'VHM': 'Vinhomes',
            'DHG': 'Dược Hậu Giang',
            'HUT': 'Tasco',
            'KBC': 'Kinh Bắc',
            'BSR': 'Lọc Hóa dầu Bình Sơn',
            'SSB': 'SeABank',
            'NVL': 'Novaland',
            'HAX': 'Hàng Xanh',
            'VSC': 'Container Việt Nam',
            'HT1': 'Xi Măng Vicem Hà Tiên',
            'QNS': 'Đường Quảng Ngãi'
        }
        
        self.symbols_df = pd.DataFrame([
            {'symbol': symbol, 'organ_name': name} 
            for symbol, name in fallback_data.items()
        ])
        
        for symbol, name in fallback_data.items():
            keywords = self._extract_keywords_from_company_name(name)
            self.symbol_keywords[symbol] = {
                'keywords': keywords,
                'company_name': name
            }
    
    def _extract_keywords_from_company_name(self, company_name):
        """Extract keywords from company name"""
        if not company_name or pd.isna(company_name):
            return []
            
        # Remove common prefixes/suffixes
        text = re.sub(r'Công ty Cổ phần|CTCP|Tập đoàn|Ngân hàng|TMCP|Công ty|Cổ phần', '', str(company_name))
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Clean spaces
        
        # Tokenize and filter
        if text:
            words = word_tokenize(text.lower())
            keywords = [word for word in words if len(word) > 2 and word not in self.vietnamese_stopwords]
        else:
            keywords = []
        
        return keywords
    
    def extract_stock_symbols_from_text(self, text):
        """Extract potential stock symbols mentioned in text with scoring"""
        if pd.isna(text) or not text:
            return []
            
        symbol_scores = {}
        text_upper = str(text).upper()
        text_lower = str(text).lower()
        
        # Direct symbol matching (highest priority)
        for symbol in self.symbols_df['symbol']:
            symbol_score = 0
            
            # Check for exact symbol mention
            if f" {symbol} " in f" {text_upper} ":
                symbol_score += 10
            elif symbol in text_upper:
                symbol_score += 5
            
            # Check for keyword matching
            if symbol in self.symbol_keywords:
                keywords = self.symbol_keywords[symbol]['keywords']
                for keyword in keywords:
                    if len(keyword) > 3 and keyword in text_lower:
                        # Longer keywords get higher scores
                        symbol_score += len(keyword) * 0.5
            
            if symbol_score > 0:
                symbol_scores[symbol] = symbol_score
        
        # Sort by score and return top matches
        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Only return symbols with meaningful scores
        return [symbol for symbol, score in sorted_symbols if score >= 3][:3]  # Top 3 matches max
    
    def classify_articles_by_symbols(self, articles_df):
        """Classify articles by stock symbols with improved accuracy"""
        if self.symbols_df is None:
            self.load_stock_symbols()
        
        results = {}
        unclassified_articles = []
        
        print("Classifying articles by stock symbols...")
        
        for idx, row in articles_df.iterrows():
            title = str(row.get('title', ''))
            description = str(row.get('description', ''))
            
            # Combine title and description for analysis
            full_text = f"{title} {description}"
            
            # Extract symbols with scoring
            found_symbols = self.extract_stock_symbols_from_text(full_text)
            
            if found_symbols:
                # Use the highest scoring symbol
                primary_symbol = found_symbols[0]
                
                if primary_symbol not in results:
                    results[primary_symbol] = []
                
                # Add article info
                article_info = row.to_dict()
                article_info['found_symbols'] = found_symbols
                article_info['primary_symbol'] = primary_symbol
                article_info['confidence_score'] = len(found_symbols)  # More symbols = higher confidence
                results[primary_symbol].append(article_info)
            else:
                # Article couldn't be classified
                article_info = row.to_dict()
                article_info['found_symbols'] = []
                article_info['primary_symbol'] = None
                article_info['confidence_score'] = 0
                unclassified_articles.append(article_info)
        
        return results, unclassified_articles
    
    def save_classified_articles(self, classified_results, unclassified_articles, output_dir='./data/classified_articles'):
        """Save classified articles to separate CSV files"""
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save classified articles by symbol
        for symbol, articles in classified_results.items():
            if articles:  # Only save if there are articles
                df = pd.DataFrame(articles)
                filename = f"{output_dir}/{symbol}_articles.csv"
                df.to_csv(filename, index=False, encoding='utf-8')
                print(f"Saved {len(articles)} articles for {symbol} to {filename}")
        
        # Save unclassified articles
        if unclassified_articles:
            df_unclassified = pd.DataFrame(unclassified_articles)
            filename = f"{output_dir}/unclassified_articles.csv"
            df_unclassified.to_csv(filename, index=False, encoding='utf-8')
            print(f"Saved {len(unclassified_articles)} unclassified articles to {filename}")
        
        # Create summary report
        summary_data = []
        for symbol, articles in classified_results.items():
            if symbol in self.symbol_keywords:
                company_name = self.symbol_keywords[symbol]['company_name']
            else:
                company_name = "Unknown"
                
            avg_confidence = sum(article.get('confidence_score', 0) for article in articles) / len(articles) if articles else 0
                
            summary_data.append({
                'symbol': symbol,
                'company_name': company_name,
                'article_count': len(articles),
                'avg_confidence': round(avg_confidence, 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('article_count', ascending=False)
        
        summary_filename = f"{output_dir}/classification_summary.csv"
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8')
        print(f"Saved classification summary to {summary_filename}")
        
        return summary_df

def main():
    """Main function to run the improved classification"""
    # Initialize classifier
    classifier = ImprovedStockNewsClassifier()
    
    # Load articles data
    print("Loading articles data...")
    try:
        articles_df = pd.read_csv('vietstock_rss_data.csv')
        print(f"Loaded {len(articles_df)} articles")
    except FileNotFoundError:
        print("Error: vietstock_rss_data.csv not found!")
        return
    
    # Load stock symbols
    print("Loading stock symbols...")
    classifier.load_stock_symbols()
    
    # Classify articles
    print("Starting improved classification...")
    classified_results, unclassified_articles = classifier.classify_articles_by_symbols(articles_df)
    
    # Save results
    print("Saving results...")
    summary_df = classifier.save_classified_articles(classified_results, unclassified_articles)
    
    # Print summary
    print("\n=== IMPROVED CLASSIFICATION SUMMARY ===")
    print(f"Total articles processed: {len(articles_df)}")
    print(f"Successfully classified: {sum(len(articles) for articles in classified_results.values())}")
    print(f"Unclassified: {len(unclassified_articles)}")
    print(f"Unique symbols found: {len(classified_results)}")
    
    print("\nTop 15 symbols by article count:")
    print(summary_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
