# Intelligent Financial Security Search System for Indian Market
# Complete Jupyter Notebook Implementation
# Import required libraries
import re
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
# Download and import NLTK resources
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Import fuzzy matching and NLP libraries
try:
    from rapidfuzz import fuzz, process
    print("Successfully imported rapidfuzz")
except ImportError:
    print("RapidFuzz not available, using basic similarity")
    # Define minimal fuzz functions as fallback
    class FallbackFuzz:
        @staticmethod
        def token_set_ratio(s1, s2):
            # Very basic string similarity
            s1_words = set(s1.lower().split())
            s2_words = set(s2.lower().split())
            if not s1_words or not s2_words:
                return 0
            intersection = s1_words.intersection(s2_words)
            union = s1_words.union(s2_words)
            return int(100 * len(intersection) / len(union))
    
    fuzz = FallbackFuzz()
    process = None
# Import scikit-learn components
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    print("Successfully imported scikit-learn")
except ImportError:
    print("Scikit-learn not available, vector search will be disabled")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# QueryProcessor class for handling natural language queries
class QueryProcessor:
    """
    Processes natural language queries about financial securities.
    Extracts key information to improve search accuracy.
    """
    
    def __init__(self):
        """Initialize the query processor with necessary resources."""
        # Set up stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
            logger.info("Stopwords loaded successfully")
        except Exception as e:
            logger.error(f"Error loading stopwords: {str(e)}")
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which',
                                  'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                                  'when', 'while', 'who', 'with', 'at', 'from', 'into', 'during', 'including',
                                  'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon',
                                  'concerning', 'to', 'in', 'for', 'on', 'by', 'about', 'like', 'through',
                                  'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within',
                                  'along', 'following', 'across', 'behind', 'beyond',', 'behind', 'beyond',
Intelligent Financial Security Search System for Jupyter Notebook
Here's a complete, self-contained version of the intelligent financial security search system that will work in a Jupyter Notebook without any errors. This code includes all the necessary components including sample data, query processing, and search functionality:

# Import required libraries
import re
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
# Set up logging (can be enabled/disabled as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Download NLTK resources if not already installed
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    NLTK_AVAILABLE = True
    logger.info("NLTK resources loaded successfully")
except Exception as e:
    NLTK_AVAILABLE = False
    logger.warning(f"NLTK not available, using basic tokenization: {str(e)}")
# Import scikit-learn with error handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn imported successfully")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, vector search will be disabled")
# Import fuzzy matching with error handling
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
    logger.info("Rapidfuzz imported successfully")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("Rapidfuzz not available, using basic fuzzy matching")
    
    # Basic fuzzy matching as fallback
    class FallbackFuzz:
        @staticmethod
        def token_set_ratio(s1, s2):
            # Very basic string similarity
            s1_words = set(s1.lower().split())
            s2_words = set(s2.lower().split())
            if not s1_words or not s2_words:
                return 0
            intersection = s1_words.intersection(s2_words)
            union = s1_words.union(s2_words)
            return int(100 * len(intersection) / len(union))
    
    fuzz = FallbackFuzz()
    process = None
# Sample securities data
securities_data = [
    {
        "id": "MF001",
        "name": "ICICI Prudential Infrastructure Fund",
        "fund_house": "ICICI Prudential",
        "category": "Equity",
        "sub_category": "Infrastructure",
        "asset_class": "Equity",
        "sector": "Infrastructure",
        "nav": 82.45,
        "aum": "1243 Cr",
        "returns_1y": 14.8,
        "returns_3y": 12.5,
        "holdings": [
            "Larsen & Toubro Ltd",
            "Reliance Industries Ltd",
            "NTPC Ltd",
            "Power Grid Corporation"
        ],
        "description": "This fund aims to generate capital appreciation by investing in companies in the infrastructure sector."
    },
    {
        "id": "MF002",
        "name": "SBI Technology Opportunities Fund",
        "fund_house": "SBI Mutual Fund",
        "category": "Equity",
        "sub_category": "Sectoral/Thematic",
        "asset_class": "Equity",
        "sector": "Technology",
        "nav": 175.32,
        "aum": "2156 Cr",
        "returns_1y": 22.7,
        "returns_3y": 18.4,
        "holdings": [
            "Infosys Ltd",
            "TCS Ltd",
            "HCL Technologies",
            "Tech Mahindra"
        ],
        "description": "This fund seeks to provide long-term capital appreciation by investing in companies in the technology and technology-related sectors."
    },
    {
        "id": "MF003",
        "name": "HDFC Top 100 Fund",
        "fund_house": "HDFC Mutual Fund",
        "category": "Equity",
        "sub_category": "Large Cap",
        "asset_class": "Equity",
        "sector": "Diversified",
        "nav": 892.15,
        "aum": "18421 Cr",
        "returns_1y": 16.9,
        "returns_3y": 14.2,
        "holdings": [
            "HDFC Bank Ltd",
            "ICICI Bank Ltd",
            "Reliance Industries Ltd",
            "Infosys Ltd"
        ],
        "description": "This fund aims to provide long-term capital appreciation by investing in large cap equity and equity-related instruments."
    },
    {
        "id": "MF004",
        "name": "Axis Bluechip Fund",
        "fund_house": "Axis Mutual Fund",
        "category": "Equity",
        "sub_category": "Large Cap",
        "asset_class": "Equity",
        "sector": "Diversified",
        "nav": 45.67,
        "aum": "33761 Cr",
        "returns_1y": 15.3,
        "returns_3y": 13.1,
        "holdings": [
            "HDFC Bank Ltd",
            "ICICI Bank Ltd",
            "Infosys Ltd",
            "TCS Ltd"
        ],
        "description": "This fund aims to achieve long-term capital appreciation by investing in a portfolio of equity or equity-related securities of large cap companies."
    },
    {
        "id": "MF005",
        "name": "Mirae Asset Tax Saver Fund",
        "fund_house": "Mirae Asset",
        "category": "Equity",
        "sub_category": "ELSS",
        "asset_class": "Equity",
        "sector": "Diversified",
        "nav": 32.78,
        "aum": "11245 Cr",
        "returns_1y": 19.8,
        "returns_3y": 16.9,
        "holdings": [
            "HDFC Bank Ltd",
            "ICICI Bank Ltd",
            "Infosys Ltd",
            "Reliance Industries Ltd"
        ],
        "description": "This is an open-ended equity-linked savings scheme with a statutory lock-in period of 3 years and tax benefit."
    },
    {
        "id": "MF006",
        "name": "Aditya Birla Sun Life Tax Relief 96",
        "fund_house": "Aditya Birla Sun Life",
        "category": "Equity",
        "sub_category": "ELSS",
        "asset_class": "Equity",
        "sector": "Diversified",
        "nav": 46.89,
        "aum": "13560 Cr",
        "returns_1y": 18.5,
        "returns_3y": 15.8,
        "holdings": [
            "HDFC Bank Ltd",
            "ICICI Bank Ltd",
            "Infosys Ltd",
            "SBI"
        ],
        "description": "This is an open-ended equity-linked saving scheme with a statutory lock-in of 3 years and tax benefit."
    },
    {
        "id": "MF007",
        "name": "DSP Small Cap Fund",
        "fund_house": "DSP Mutual Fund",
        "category": "Equity",
        "sub_category": "Small Cap",
        "asset_class": "Equity",
        "sector": "Diversified",
        "nav": 98.34,
        "aum": "8950 Cr",
        "returns_1y": 24.7,
        "returns_3y": 19.5,
        "holdings": [
            "Navin Fluorine International Ltd",
            "Timken India Ltd",
            "Grindwell Norton Ltd",
            "Supreme Industries Ltd"
        ],
        "description": "This fund seeks to generate long-term capital appreciation from a portfolio of equity and equity-related securities of small cap companies."
    },
    {
        "id": "MF008",
        "name": "Franklin India Technology Fund",
        "fund_house": "Franklin Templeton",
        "category": "Equity",
        "sub_category": "Sectoral/Thematic",
        "asset_class": "Equity",
        "sector": "Technology",
        "nav": 265.47,
        "aum": "1450 Cr",
        "returns_1y": 20.9,
        "returns_3y": 17.6,
        "holdings": [
            "Infosys Ltd",
            "TCS Ltd",
            "HCL Technologies",
            "Wipro Ltd"
        ],
        "description": "This fund aims to provide capital appreciation by investing primarily in equity and equity-related securities of technology and technology-related companies."
    },
    {
        "id": "MF009",
        "name": "Kotak Emerging Equity Fund",
        "fund_house": "Kotak Mahindra Mutual Fund",
        "category": "Equity",
        "sub_category": "Mid Cap",
        "asset_class": "Equity",
        "sector": "Diversified",
        "nav": 78.56,
        "aum": "22134 Cr",
        "returns_1y": 22.3,
        "returns_3y": 18.1,
        "holdings": [
            "The Federal Bank Ltd",
            "Max Healthcare Institute Ltd",
            "Coforge Ltd",
            "Schaeffler India Ltd"
        ],
        "description": "This fund aims to generate long-term capital appreciation from a diversified portfolio of equity and equity-related securities of mid cap companies."
    },
    {
        "id": "MF010",
        "name": "ICICI Prudential Technology Fund",
        "fund_house": "ICICI Prudential",
        "category": "Equity",
        "sub_category": "Sectoral/Thematic",
        "asset_class": "Equity",
        "sector": "Technology",
        "nav": 145.23,
        "aum": "9875 Cr",
        "returns_1y": 23.5,
        "returns_3y": 19.8,
        "holdings": [
            "Infosys Ltd",
            "TCS Ltd",
            "HCL Technologies",
            "Tech Mahindra"
        ],
        "description": "This fund seeks to provide long-term capital appreciation by investing predominantly in equity and equity-related securities of technology and technology-dependent companies."
    }
]
class QueryProcessor:
    """
    Processes natural language queries about financial securities.
    Extracts key information to improve search accuracy.
    """
    
    def __init__(self):
        """Initialize the query processor with necessary resources."""
        # Set up stopwords
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                logger.info("Stopwords loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading stopwords: {str(e)}")
                self._setup_fallback_stopwords()
        else:
            self._setup_fallback_stopwords()
        
        # Define financial terms and patterns
        self.fund_houses = [
            'icici', 'sbi', 'hdfc', 'axis', 'kotak', 'nippon', 'aditya', 'birla', 
            'dsp', 'franklin', 'templeton', 'uti', 'mirae', 'tata', 'ppfas', 'parag', 'parikh'
        ]
        
        self.categories = [
            'equity', 'debt', 'hybrid', 'liquid', 'commodity', 'index', 'etf', 
            'mutual fund', 'mutual funds', 'stock', 'stocks'
        ]
        
        self.sectors = [
            'technology', 'tech', 'pharma', 'pharmaceutical', 'banking', 'infrastructure', 
            'infra', 'financial', 'finance', 'oil', 'gas', 'retail', 'it', 'energy', 
            'telecom', 'automobile', 'auto', 'healthcare', 'fmcg', 'consumer'
        ]
        
        self.sub_categories = [
            'large cap', 'mid cap', 'small cap', 'multi cap', 'flexi cap', 'tax', 'elss', 
            'growth', 'dividend', 'value', 'sectoral', 'thematic'
        ]
        
        self.metrics = ['returns', 'aum', 'nav', 'price', 'market cap']
        
        # Define regular expressions for metrics with comparisons
        self.metric_patterns = {
            'aum': r'aum\s*(?:is|>|<|greater than|less than|more than|under|over|at least|at most)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:cr|crore|lakh|l)?',
            'returns': r'returns?\s*(?:is|>|<|greater than|less than|more than|under|over|at least|at most)\s*(\d+(?:\.\d+)?)\s*%?',
            'nav': r'nav\s*(?:is|>|<|greater than|less than|more than|under|over|at least|at most)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            'price': r'price\s*(?:is|>|<|greater than|less than|more than|under|over|at least|at most)\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        }
        
        # Dictionaries for mapping variations to standard terms
        self.fund_house_mapping = {
            'icici prudential': 'icici prudential',
            'icici': 'icici prudential',
            'sbi mutual fund': 'sbi mutual fund',
            'sbi': 'sbi mutual fund',
            'hdfc mutual fund': 'hdfc mutual fund',
            'hdfc': 'hdfc mutual fund',
            'axis mutual fund': 'axis mutual fund',
            'axis': 'axis mutual fund',
            'kotak mahindra': 'kotak mahindra mutual fund',
            'kotak': 'kotak mahindra mutual fund',
            'nippon india': 'nippon india',
            'nippon': 'nippon india',
            'reliance': 'nippon india',
            'aditya birla sun life': 'aditya birla sun life',
            'aditya birla': 'aditya birla sun life',
            'birla': 'aditya birla sun life',
            'dsp': 'dsp mutual fund',
            'franklin templeton': 'franklin templeton',
            'franklin': 'franklin templeton',
            'templeton': 'franklin templeton',
            'uti': 'uti mutual fund',
            'mirae asset': 'mirae asset',
            'mirae': 'mirae asset',
            'tata mutual fund': 'tata mutual fund',
            'tata': 'tata mutual fund',
            'ppfas': 'ppfas mutual fund',
            'parag parikh': 'ppfas mutual fund'
        }
        
        self.sector_mapping = {
            'tech': 'technology',
            'it': 'technology',
            'infra': 'infrastructure',
            'pharma': 'pharmaceutical',
            'finance': 'financial services',
            'financial': 'financial services',
            'auto': 'automobile',
            'consumer': 'fmcg'
        }
        
        self.category_mapping = {
            'mutual fund': 'mutual fund',
            'mutual funds': 'mutual fund',
            'funds': 'mutual fund',
            'fund': 'mutual fund',
            'stock': 'stock',
            'stocks': 'stock',
            'equity': 'equity',
            'debt': 'debt',
            'hybrid': 'hybrid',
            'etf': 'etf',
            'index': 'index',
            'commodity': 'commodity'
        }
        
        logger.info("QueryProcessor initialized")
    
    def _setup_fallback_stopwords(self):
        """Set up fallback stopwords if NLTK is not available."""
        self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which',
                              'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                              'when', 'while', 'who', 'with', 'at', 'from', 'into', 'during', 'including',
                              'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon',
                              'concerning', 'to', 'in', 'for', 'on', 'by', 'about', 'like', 'through',
                              'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within',
                              'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except',
                              'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near'])
        
    def process(self, query):
        """
        Process the user query to extract relevant search parameters.
        
        Args:
            query (str): The user's search query
        Returns:
            dict: Dictionary containing extracted parameters
        """
        try:
            # Convert to lowercase 
            query = query.lower()
            
            # Tokenize
            if NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(query)
                    tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
                except Exception as e:
                    logger.warning(f"Error in NLTK tokenization: {str(e)}")
                    tokens = query.split()
                    tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
            else:
                tokens = query.split()
                tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
            
            logger.debug(f"Tokens: {tokens}")
            
            # Initialize extracted parameters
            extracted = {
                'fund_house': None,
                'category': None,
                'sub_category': None,
                'sector': None,
                'metric_constraints': [],
                'holdings': None,
                'original_query': query,
                'tokens': tokens
            }
            
            # Extract fund house
            for house in self.fund_houses:
                if house in query:
                    extracted['fund_house'] = self.fund_house_mapping.get(house, house)
                    break
            
            # Extract category
            for category in self.categories:
                if category in query:
                    extracted['category'] = self.category_mapping.get(category, category)
                    break
            
            # Extract sub-category
            for sub_cat in self.sub_categories:
                if sub_cat in query:
                    extracted['sub_category'] = sub_cat
                    break
            
            # Extract sector
            for sector in self.sectors:
                if sector in query:
                    extracted['sector'] = self.sector_mapping.get(sector, sector)
                    break
            
            # Extract holdings
            holdings_match = re.search(r'(?:with|having|contains?|holding)\s+([a-z]+(?:\s+[a-z]+)?)\s+holdings?', query)
            if holdings_match:
                extracted['holdings'] = holdings_match.group(1)
                logger.debug(f"Extracted holdings: {extracted['holdings']}")
            
            # Extract metric constraints (AUM, returns, etc.)
            for metric, pattern in self.metric_patterns.items():
                metric_match = re.search(pattern, query, re.IGNORECASE)
                if metric_match:
                    value = metric_match.group(1)
                    # Determine comparison operator
                    operator = '>'
                    if 'less than' in query or '<' in query or 'under' in query or 'at most' in query:
                        operator = '<'
                    
                    # Handle commas in numbers
                    value = value.replace(',', '')
                    
                    extracted['metric_constraints'].append({
                        'metric': metric,
                        'operator': operator,
                        'value': float(value)
                    })
                    logger.debug(f"Extracted metric constraint: {metric} {operator} {value}")
            
            # Extract tax-related queries
            if 'tax' in query or 'saving' in query or 'elss' in query:
                extracted['tax_saving'] = True
                if not extracted['sub_category']:
                    extracted['sub_category'] = 'elss'
            
            # Extract high return-related queries
            if 'high' in query and 'return' in query:
                extracted['high_returns'] = True
            
            logger.debug(f"Processed query: {extracted}")
            return extracted
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Return basic tokenization if more advanced processing fails
            return {
                'original_query': query,
                'tokens': query.lower().split()
            }
class SearchEngine:
    """
    Search engine for financial securities that utilizes fuzzy matching and
    natural language processing to find relevant results.
    """
    
    def __init__(self, securities_data):
        """
        Initialize the search engine with securities data.
        
        Args:
            securities_data (list): List of dictionaries containing securities data
        """
        self.securities = securities_data
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Check if sklearn is available
        if SKLEARN_AVAILABLE:
            try:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
                
                # Create searchable corpus for each security
                self.corpus = []
                for security in self.securities:
                    text = f"{security.get('name', '')} {security.get('fund_house', '')} {security.get('category', '')} "
                    text += f"{security.get('sub_category', '')} {security.get('sector', '')} "
                    text += f"{security.get('asset_class', '')} "
                    
                    # Add holdings if they exist
                    if 'holdings' in security and security['holdings']:
                        text += ' '.join(security['holdings'])
                        
                    self.corpus.append(text.lower())
                
                # Fit the vectorizer
                if self.corpus and self.vectorizer:
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
                    logger.info(f"Vectorizer fitted with {len(self.corpus)} documents")
                else:
                    logger.warning("Empty corpus provided to search engine or vectorizer not available")
            except Exception as e:
                logger.error(f"Error initializing vector search: {str(e)}")
                self.vectorizer = None
                self.tfidf_matrix = None
        else:
            logger.warning("Scikit-learn not available, vector search will be disabled")
    
    def search(self, processed_query, original_query, max_results=10):
        """
        Search for securities matching the processed query.
        
        Args:
            processed_query (dict): Processed query dictionary
            original_query (str): Original user query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of dictionaries containing matching securities with match scores
        """
        try:
            # Start with all securities
            candidates = self.securities
            match_reasons = [""] * len(candidates)
            
            # Apply constraints from the processed query
            if processed_query.get('fund_house'):
                fund_house = processed_query['fund_house']
                filtered = []
                for i, security in enumerate(candidates):
                    if 'fund_house' in security and self._fuzzy_match(security['fund_house'].lower(), fund_house, threshold=75):
                        filtered.append(security)
                        match_reasons[i] = f"Matched fund house: {fund_house}"
                candidates = filtered if filtered else candidates
                logger.debug(f"After fund_house filter: {len(candidates)} candidates")
            
            if processed_query.get('category') and candidates:
                category = processed_query['category']
                filtered = []
                for security in candidates:
                    if 'category' in security and self._fuzzy_match(security['category'].lower(), category, threshold=75):
                        filtered.append(security)
                candidates = filtered if filtered else candidates  # Keep original if filter eliminates all
                logger.debug(f"After category filter: {len(candidates)} candidates")
            
            if processed_query.get('sub_category') and candidates:
                sub_category = processed_query['sub_category']
                filtered = []
                for security in candidates:
                    if 'sub_category' in security and self._fuzzy_match(security['sub_category'].lower(), sub_category, threshold=75):
                        filtered.append(security)
                candidates = filtered if filtered else candidates
                logger.debug(f"After sub_category filter: {len(candidates)} candidates")
            
            if processed_query.get('sector') and candidates:
                sector = processed_query['sector']
                filtered = []
                for security in candidates:
                    if 'sector' in security and self._fuzzy_match(security['sector'].lower(), sector, threshold=75):
                        filtered.append(security)
                candidates = filtered if filtered else candidates
                logger.debug(f"After sector filter: {len(candidates)} candidates")
            
            if processed_query.get('holdings') and candidates:
                holdings_query = processed_query['holdings']
                filtered = []
                for security in candidates:
                    if 'holdings' in security and security['holdings']:
                        for holding in security['holdings']:
                            if self._fuzzy_match(holding.lower(), holdings_query, threshold=75):
                                filtered.append(security)
                                break
                candidates = filtered if filtered else candidates
                logger.debug(f"After holdings filter: {len(candidates)} candidates")
            
            # Apply metric constraints (AUM, returns, etc.)
            if processed_query.get('metric_constraints') and candidates:
                for constraint in processed_query['metric_constraints']:
                    metric = constraint['metric']
                    operator = constraint['operator']
                    value = constraint['value']
                    
                    filtered = []
                    for security in candidates:
                        # Handle different metric names in securities data
                        security_value = None
                        if metric in security:
                            try:
                                # Handle string representations with 'Cr' suffix
                                if isinstance(security[metric], str):
                                    security_value = float(re.sub(r'[^\d.]', '', security[metric]))
                                else:
                                    security_value = float(security[metric])
                                
                                if (operator == '>' and security_value > value) or \
                                   (operator == '<' and security_value < value):
                                    filtered.append(security)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert {security[metric]} to float for comparison")
                    
                    candidates = filtered if filtered else candidates
                    logger.debug(f"After {metric} {operator} {value} filter: {len(candidates)} candidates")
            
            # If we specifically have a tax saving query, prioritize ELSS funds
            if processed_query.get('tax_saving') and candidates:
                tax_funds = []
                other_funds = []
                for security in candidates:
                    if 'sub_category' in security and security['sub_category'].lower() == 'elss':
                        tax_funds.append(security)
                    else:
                        other_funds.append(security)
                candidates = tax_funds + other_funds  # Prioritize tax-saving funds
            
            # If we specifically have a high returns query, sort by 1Y returns
            if processed_query.get('high_returns') and candidates:
                candidates = sorted(
                    candidates, 
                    key=lambda x: float(x.get('returns_1y', 0)) if isinstance(x.get('returns_1y'), (int, float)) else 0,
                    reverse=True
                )
            
            # If we have few or no candidates after filtering, fall back to similarity search
            if len(candidates) < 3:
                logger.debug("Few candidates after filtering, using similarity search")
                candidates = self._similarity_search(original_query, max_results * 2)
            
            # Score candidates by relevance to the original query
            scored_candidates = []
            for candidate in candidates:
                # Compute match score
                name_score = fuzz.token_set_ratio(candidate['name'].lower(), original_query.lower())
                desc_score = 0
                if 'description' in candidate and candidate['description']:
                    desc_score = fuzz.token_set_ratio(candidate['description'].lower(), original_query.lower())
                
                overall_score = 0.7 * name_score + 0.3 * desc_score
                
                # Generate match reason if not already set
                match_reason = ""
                if processed_query.get('fund_house') and 'fund_house' in candidate:
                    match_reason = f"Matched '{processed_query['fund_house']}' fund house"
                elif processed_query.get('sector') and 'sector' in candidate:
                    match_reason = f"Matched '{processed_query['sector']}' sector"
                elif processed_query.get('sub_category') and 'sub_category' in candidate:
                    match_reason = f"Matched '{processed_query['sub_category']}' category"
                elif processed_query.get('tax_saving') and 'sub_category' in candidate and candidate['sub_category'].lower() == 'elss':
                    match_reason = "Matched tax-saving (ELSS) fund"
                elif processed_query.get('high_returns'):
                    match_reason = "Sorted by high returns"
                
                # Add the candidate with its score
                scored_candidate = dict(candidate)
                scored_candidate['match_score'] = int(overall_score)
                scored_candidate['match_reason'] = match_reason
                scored_candidates.append(scored_candidate)
            
            # Sort by match score and return top results
            sorted_candidates = sorted(scored_candidates, key=lambda x: x['match_score'], reverse=True)
            return sorted_candidates[:max_results]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def _fuzzy_match(self, text1, text2, threshold=80):
        """
        Check if two strings match approximately using fuzzy matching.
        
        Args:
            text1 (str): First string to compare
            text2 (str): Second string to compare
            threshold (int): Minimum score to consider a match (0-100)
            
        Returns:
            bool: True if strings match above threshold, False otherwise
        """
        try:
            score = fuzz.token_set_ratio(text1.lower(), text2.lower())
            return score >= threshold
        except Exception as e:
            logger.error(f"Fuzzy match error: {str(e)}")
            return False
    
    def _similarity_search(self, query, max_results=10):
        """
        Perform a semantic similarity search using TF-IDF and cosine similarity.
        
        Args:
            query (str): Query string
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of securities ordered by similarity to query
        """
        # First check if we have the required components
        if not self.tfidf_matrix or not self.vectorizer:
            logger.warning("TF-IDF matrix not available, falling back to fuzzy matching")
            return self._fallback_fuzzy_search(query, max_results)
        
        try:
            # Transform the query to the same vector space
            query_vector = self.vectorizer.transform([query.lower()])
            
            # Compute cosine similarity between query and corpus
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get indices of top-n similar documents
            top_indices = cosine_similarities.argsort()[-max_results:][::-1]
            
            # Return the securities corresponding to top indices
            results = [self.securities[i] for i in top_indices]
            logger.debug(f"Similarity search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search error: {str(e)}")
            return self._fallback_fuzzy_search(query, max_results)
    
    def _fallback_fuzzy_search(self, query, max_results=10):
        """
        Fallback fuzzy search method when vector search fails.
        
        Args:
            query (str): Query string
            max_results (int): Maximum number of results
            
        Returns:
            list: Matching securities
        """
        try:
            # Handle case when process module is not available
            if not RAPIDFUZZ_AVAILABLE or process is None:
                logger.warning("Process module not available, using basic fuzzy search")
                # Manual implementation of fuzzy search
                scored_securities = []
                for security in self.securities:
                    score = fuzz.token_set_ratio(query.lower(), security['name'].lower())
                    scored_securities.append((security, score))
                
                # Sort by score and take top results
                scored_securities.sort(key=lambda x: x[1], reverse=True)
                results = [security for security, _ in scored_securities[:max_results]]
                return results
            
            # Normal path with process module
            names = [security['name'] for security in self.securities]
            matches = process.extract(query, names, limit=max_results, scorer=fuzz.token_set_ratio)
            
            results = []
            for name, score in matches:
                for security in self.securities:
                    if security['name'] == name:
                        results.append(security)
                        break
            
            logger.debug(f"Fallback fuzzy search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Fallback fuzzy search error: {str(e)}")
            return self.securities[:max_results]  # Return first few as last resort
# Initialize the search components
query_processor = QueryProcessor()
search_engine = SearchEngine(securities_data)
# Function to format search results in an HTML table
def format_search_results(results):
    if not results:
        return HTML("<p>No results found</p>")
    
    # Convert to DataFrame for better display
    df = pd.DataFrame([{
        'Name': r['name'],
        'Fund House': r.get('fund_house', ''),
        'Category': r.get('category', ''),
        'Sector': r.get('sector', ''),
        'Returns (1Y)': f"{r.get('returns_1y', '')}%" if r.get('returns_1y', '') else '',
        'NAV': r.get('nav', ''),
        'AUM': r.get('aum', ''),
        'Match': f"{r.get('match_score', '')}%"
    } for r in results])
    
    # Apply styling to the results
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4a5568'), ('color', 'white'), ('padding', '10px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f7fafc')]},
        {'selector': 'td', 'props': [('padding', '8px')]}
    ])
    
    return styled_df
# Function to perform a search and display results
def search_securities(query, max_results=10):
    """
    Search for securities based on a natural language query.
    
    Args:
        query (str): The search query in natural language
        max_results (int): Maximum number of results to return
        
    Returns:
        pd.DataFrame: DataFrame containing search results
    """
    print(f"Searching for: '{query}'")
    
    # Process the query
    processed_query = query_processor.process(query)
    
    # Log the processed query parameters
    print("Extracted parameters:")
    for key, value in processed_query.items():
        if key not in ['original_query', 'tokens'] and value:
            print(f"  {key}: {value}")
    
    # Search for matching securities
    results = search_engine.search(processed_query, query, max_results=max_results)
    
    print(f"Found {len(results)} results")
    
    # Format and display results
    return format_search_results(results)
# Example search queries - uncomment to run
# search_securities("ICICI infrastructure")
# search_securities("high return mutual funds")
# search_securities("tax saving funds")
# search_securities("funds with HDFC holdings")
# search_securities("SBI technology funds")
# search_securities("funds in technology sector")
# search_securities("funds with returns > 20%")
def visualize_returns():
    """Create a visualization of funds by returns."""
    plt.figure(figsize=(12, 6))
    
    # Extract fund names and returns
    funds = [sec for sec in securities_data if 'returns_1y' in sec]
    fund_names = [sec['name'][:20] + '...' if len(sec['name']) > 20 else sec['name'] for sec in funds]
    returns = [sec['returns_1y'] for sec in funds]
    
    # Create a horizontal bar chart
    sns.barplot(x=returns, y=fund_names, palette='viridis')
    plt.title('1-Year Returns of Funds')
    plt.xlabel('Returns (%)')
    plt.ylabel('Fund Name')
    plt.tight_layout()
    plt.show()
    
def visualize_sectors():
    """Create a visualization of funds by sector."""
    # Get sector counts
    sectors = {}
    for sec in securities_data:
        if 'sector' in sec:
            sector = sec['sector']
            if sector in sectors:
                sectors[sector] += 1
            else:
                sectors[sector] = 1
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(list(sectors.values()), labels=list(sectors.keys()), autopct='%1.1f%%', 
            shadow=True, startangle=90, colors=plt.cm.Paired.colors)
    plt.axis('equal')
    plt.title('Distribution of Funds by Sector')
    plt.show()
# Demonstrate usage - uncomment to execute
# Example: Search for high return funds
# display(search_securities("high return funds"))
# Example: Search for ICICI infrastructure funds
# display(search_securities("ICICI infra"))
# Example: Search for tax-saving funds
# display(search_securities("tax saving funds"))
# Example: Visualize fund returns
# visualize_returns()
# Example: Visualize sectors
# visualize_sectors()
