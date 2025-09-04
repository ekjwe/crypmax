"""
Utility functions for QuantumTrade Bot
Common helper functions used across the application
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
import hmac
import base64
import os
import re

logger = logging.getLogger(__name__)

class Utils:
    """Utility class with static helper methods"""
    
    @staticmethod
    def format_currency(amount: float, currency: str = "USD") -> str:
        """Format amount as currency"""
        try:
            if currency == "USD":
                return f"${amount:,.2f}"
            else:
                return f"{amount:,.2f} {currency}"
        except:
            return f"{amount} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format value as percentage"""
        try:
            return f"{value:.{decimals}f}%"
        except:
            return f"{value}%"
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """Safely convert value to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def timestamp_to_datetime(timestamp: Union[int, float, str]) -> datetime:
        """Convert timestamp to datetime object"""
        try:
            if isinstance(timestamp, str):
                # Try parsing ISO format first
                try:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = float(timestamp)
            
            if isinstance(timestamp, (int, float)):
                # Handle both seconds and milliseconds timestamps
                if timestamp > 1e10:  # Milliseconds
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp)
                
        except Exception as e:
            logger.error(f"Error converting timestamp {timestamp}: {e}")
            return datetime.now()
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """Convert datetime to timestamp (milliseconds)"""
        try:
            return int(dt.timestamp() * 1000)
        except Exception as e:
            logger.error(f"Error converting datetime {dt}: {e}")
            return int(time.time() * 1000)
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        try:
            if old_value == 0:
                return 0.0
            return ((new_value - old_value) / old_value) * 100
        except:
            return 0.0
    
    @staticmethod
    def round_to_precision(value: float, precision: int) -> float:
        """Round value to specified decimal places"""
        try:
            return round(value, precision)
        except:
            return value
    
    @staticmethod
    def validate_trading_pair(pair: str) -> bool:
        """Validate trading pair format"""
        try:
            # Expected format: BTC/USD, ETH/USDT, etc.
            pattern = r'^[A-Z]{2,10}/[A-Z]{2,10}$'
            return bool(re.match(pattern, pair.upper()))
        except:
            return False
    
    @staticmethod
    def normalize_pair(pair: str) -> str:
        """Normalize trading pair format"""
        try:
            return pair.upper().replace('-', '/').replace('_', '/')
        except:
            return pair
    
    @staticmethod
    def generate_trade_id(pair: str, timestamp: datetime = None) -> str:
        """Generate unique trade ID"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Create ID from pair and timestamp
            time_str = timestamp.strftime('%Y%m%d%H%M%S%f')
            pair_clean = pair.replace('/', '').replace('-', '').replace('_', '')
            return f"trade_{pair_clean}_{time_str}"
        except Exception as e:
            logger.error(f"Error generating trade ID: {e}")
            return f"trade_{int(time.time())}"
    
    @staticmethod
    def calculate_position_value(price: float, amount: float) -> float:
        """Calculate position value"""
        try:
            return price * amount
        except:
            return 0.0
    
    @staticmethod
    def calculate_profit_loss(entry_price: float, exit_price: float, 
                            amount: float, action: str) -> Dict[str, float]:
        """Calculate profit/loss for a trade"""
        try:
            if action.lower() == "buy":
                profit = (exit_price - entry_price) * amount
            else:  # sell
                profit = (entry_price - exit_price) * amount
            
            profit_percent = Utils.calculate_percentage_change(entry_price, exit_price)
            if action.lower() == "sell":
                profit_percent = -profit_percent
            
            return {
                "profit": profit,
                "profit_percent": profit_percent
            }
        except Exception as e:
            logger.error(f"Error calculating P/L: {e}")
            return {"profit": 0.0, "profit_percent": 0.0}
    
    @staticmethod
    def validate_price(price: Union[str, float]) -> bool:
        """Validate price value"""
        try:
            price = float(price)
            return price > 0
        except:
            return False
    
    @staticmethod
    def validate_amount(amount: Union[str, float]) -> bool:
        """Validate amount value"""
        try:
            amount = float(amount)
            return amount > 0
        except:
            return False
    
    @staticmethod
    def truncate_float(value: float, decimals: int) -> float:
        """Truncate float to specified decimal places"""
        try:
            multiplier = 10 ** decimals
            return int(value * multiplier) / multiplier
        except:
            return value
    
    @staticmethod
    def format_timeframe(timeframe: str) -> str:
        """Format timeframe string"""
        timeframe_map = {
            "1m": "1 minute",
            "3m": "3 minutes",
            "5m": "5 minutes",
            "15m": "15 minutes",
            "30m": "30 minutes",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day"
        }
        return timeframe_map.get(timeframe.lower(), timeframe)
    
    @staticmethod
    def get_timeframe_seconds(timeframe: str) -> int:
        """Get timeframe duration in seconds"""
        timeframe_seconds = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        return timeframe_seconds.get(timeframe.lower(), 60)
    
    @staticmethod
    def create_signature(secret: str, data: str, algorithm: str = "sha256") -> str:
        """Create HMAC signature"""
        try:
            if algorithm == "sha256":
                return hmac.new(
                    secret.encode(),
                    data.encode(),
                    hashlib.sha256
                ).hexdigest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        except Exception as e:
            logger.error(f"Error creating signature: {e}")
            return ""
    
    @staticmethod
    def encode_base64(data: str) -> str:
        """Encode string to base64"""
        try:
            return base64.b64encode(data.encode()).decode()
        except Exception as e:
            logger.error(f"Error encoding base64: {e}")
            return ""
    
    @staticmethod
    def decode_base64(data: str) -> str:
        """Decode base64 string"""
        try:
            return base64.b64decode(data.encode()).decode()
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            return ""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations"""
        try:
            # Remove or replace invalid characters
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            
            # Limit length
            if len(filename) > 255:
                filename = filename[:255]
            
            return filename
        except:
            return "sanitized_filename"
    
    @staticmethod
    def chunk_list(lst: List, chunk_size: int) -> List[List]:
        """Split list into chunks of specified size"""
        try:
            return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
        except:
            return [lst]
    
    @staticmethod
    def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        try:
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = Utils.deep_merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        except:
            return dict1
    
    @staticmethod
    def retry_operation(operation, max_retries: int = 3, delay: float = 1.0, 
                       backoff: float = 2.0) -> Any:
        """Retry operation with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    @staticmethod
    def load_json_file(filepath: str, default: Any = None) -> Any:
        """Load JSON file with error handling"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error loading JSON file {filepath}: {e}")
            return default
    
    @staticmethod
    def save_json_file(filepath: str, data: Any) -> bool:
        """Save data to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {e}")
            return False
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(filepath)
        except:
            return 0
    
    @staticmethod
    def ensure_directory(dirpath: str) -> bool:
        """Ensure directory exists"""
        try:
            os.makedirs(dirpath, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {dirpath}: {e}")
            return False
    
    @staticmethod
    def clean_old_files(directory: str, max_age_days: int = 30, 
                       pattern: str = "*") -> int:
        """Clean old files from directory"""
        try:
            import glob
            files_removed = 0
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            for filepath in glob.glob(os.path.join(directory, pattern)):
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    files_removed += 1
            
            return files_removed
        except Exception as e:
            logger.error(f"Error cleaning old files: {e}")
            return 0
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        try:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            elif seconds < 86400:
                hours = seconds / 3600
                return f"{hours:.1f}h"
            else:
                days = seconds / 86400
                return f"{days:.1f}d"
        except:
            return f"{seconds}s"
    
    @staticmethod
    def is_market_open(timezone: str = "UTC") -> bool:
        """Check if market is open (simplified - always returns True for crypto)"""
        # Crypto markets are always open
        return True
    
    @staticmethod
    def get_next_candle_time(timeframe: str, current_time: datetime = None) -> datetime:
        """Get next candle open time for given timeframe"""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            timeframe_seconds = Utils.get_timeframe_seconds(timeframe)
            
            # Round down to timeframe boundary
            timestamp = int(current_time.timestamp())
            rounded_timestamp = (timestamp // timeframe_seconds) * timeframe_seconds
            
            # Add one timeframe period
            next_timestamp = rounded_timestamp + timeframe_seconds
            
            return datetime.fromtimestamp(next_timestamp)
        except Exception as e:
            logger.error(f"Error calculating next candle time: {e}")
            return current_time + timedelta(minutes=1)

# Create singleton instance
utils = Utils()
