"""
Configuration management for QuantumTrade Bot
"""

import json
import os
from typing import Dict, List, Any

class Config:
    """Configuration manager for the trading bot"""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.default_config = {
            "trading": {
                "pairs": ["BTC/USD", "ETH/USD", "BNB/USD", "DOGE/USD"],
                "default_pair": "BTC/USD",
                "min_investment": 1.0,
                "default_investment": 10.0,
                "max_investment": 1000.0,
                "daily_trade_limit": 15000,
                "cooldown_minutes": 1,
                "trading_timeframe": "3m",
                "analysis_timeframes": ["15m", "30m", "1h"],
                "max_loss_percent": 5.0,
                "daily_profit_target": 100.0,
                "stop_loss_percent": 2.0,
                "default_exchange": "binance"
            },
            "exchanges": {
                "binance": {
                    "enabled": True,
                    "testnet": True,
                    "min_order_value": 10.0,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "rate_limit_per_minute": 1200
                },
                "bitget": {
                    "enabled": False,
                    "testnet": True,
                    "min_order_value": 5.0,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "rate_limit_per_minute": 1000
                },
                "coinbase": {
                    "enabled": False,
                    "sandbox": True,
                    "min_order_value": 5.0,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "rate_limit_per_minute": 1000
                }
            },
            "technical_analysis": {
                "ema_periods": [9, 21, 50],
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "ma_periods": [20, 50, 200],
                "volume_threshold": 1.5,
                "spread_threshold": 0.1
            },
            "ml_model": {
                "model_file": "trading_model.pkl",
                "retrain_hours": 24,
                "features": ["price", "volume", "rsi", "ema", "ma"]
            },
            "alerts": {
                "email_enabled": True,
                "telegram_enabled": True,
                "email_smtp_server": "smtp.gmail.com",
                "email_port": 587,
                "email_from": "",
                "email_to": "",
                "email_password": "",
                "telegram_bot_token": "",
                "telegram_chat_id": ""
            },
            "database": {
                "file": "trading_data.db",
                "encryption_key": ""
            },
            "api": {
                "port": 8000,
                "host": "0.0.0.0",
                "secret_key": "your-secret-key-here"
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_config(self.default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config
        else:
            self.save_config()
            return self.default_config
    
    def _merge_config(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, section: str, key: str = None, default=None):
        """Get configuration value"""
        try:
            if key is None:
                section_data = self.config.get(section, default)
                # Ensure exchange configs are dicts, not None
                if section in ['binance', 'bitget', 'coinbase'] and section_data is None:
                    return {}
                return section_data
            section_data = self.config.get(section, {})
            if section in ['binance', 'bitget', 'coinbase'] and section_data is None:
                section_data = {}
            return section_data.get(key, default)
        except:
            return default
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
    
    def update_trading_config(self, updates: Dict[str, Any]):
        """Update trading configuration"""
        for key, value in updates.items():
            if key in self.config["trading"]:
                self.config["trading"][key] = value
        self.save_config()
    
    def get_trading_pairs(self) -> List[str]:
        """Get list of trading pairs"""
        return self.config["trading"]["pairs"]
    
    def get_investment_limits(self) -> Dict[str, float]:
        """Get investment limits"""
        return {
            "min": self.config["trading"]["min_investment"],
            "max": self.config["trading"]["max_investment"],
            "default": self.config["trading"]["default_investment"]
        }
    
    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials from environment variables"""
        return {
            "email_password": os.getenv("EMAIL_PASSWORD", self.config["alerts"]["email_password"]),
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", self.config["alerts"]["telegram_bot_token"]),
            "encryption_key": os.getenv("ENCRYPTION_KEY", self.config["database"]["encryption_key"]),
            # Exchange API credentials
            "binance_api_key": os.getenv("BINANCE_API_KEY", ""),
            "binance_secret_key": os.getenv("BINANCE_SECRET_KEY", ""),
            "bitget_api_key": os.getenv("BITGET_API_KEY", ""),
            "bitget_secret_key": os.getenv("BITGET_SECRET_KEY", ""),
            "bitget_passphrase": os.getenv("BITGET_PASSPHRASE", ""),
            "coinbase_api_key": os.getenv("COINBASE_API_KEY", ""),
            "coinbase_secret_key": os.getenv("COINBASE_SECRET_KEY", ""),
            "coinbase_passphrase": os.getenv("COINBASE_PASSPHRASE", "")
        }

    def get_exchange_config(self, exchange: str) -> Dict:
        """Get configuration for a specific exchange"""
        return self.config.get("exchanges", {}).get(exchange, {})

    def get_enabled_exchanges(self) -> List[str]:
        """Get list of enabled exchanges"""
        exchanges = []
        for exchange, config in self.config.get("exchanges", {}).items():
            if config.get("enabled", False):
                exchanges.append(exchange)
        return exchanges

    def get_default_exchange(self) -> str:
        """Get the default exchange"""
        return self.config.get("trading", {}).get("default_exchange", "binance")

    def set_exchange_enabled(self, exchange: str, enabled: bool):
        """Enable or disable an exchange"""
        if exchange in self.config.get("exchanges", {}):
            self.config["exchanges"][exchange]["enabled"] = enabled
            self.save_config()

    def update_exchange_config(self, exchange: str, updates: Dict):
        """Update configuration for a specific exchange"""
        if exchange in self.config.get("exchanges", {}):
            self.config["exchanges"][exchange].update(updates)
            self.save_config()
