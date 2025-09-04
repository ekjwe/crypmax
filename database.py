"""
Database management for QuantumTrade Bot
Handles trade storage, configuration, and data encryption
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import os
from encryption import EncryptionManager

logger = logging.getLogger(__name__)

class Database:
    """Database manager for trading bot data"""
    
    def __init__(self, db_file: str = "trading_data.db"):
        self.db_file = db_file
        self.encryption = EncryptionManager()
        self.connection = None
        logger.info(f"Database manager initialized: {db_file}")
    
    def initialize(self):
        """Initialize database tables"""
        try:
            self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            self._create_tables()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                value REAL NOT NULL,
                confidence REAL,
                signals TEXT,
                status TEXT DEFAULT 'executed',
                profit REAL DEFAULT 0,
                profit_percent REAL DEFAULT 0,
                stop_loss REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Configuration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                trades_count INTEGER DEFAULT 0,
                profit REAL DEFAULT 0,
                profit_percent REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                volume REAL DEFAULT 0,
                reset_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Market data table (for ML training)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                priority TEXT DEFAULT 'normal',
                sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
                channels TEXT
            )
        """)
        
        # Performance tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                pair TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    def add_trade(self, trade: Dict):
        """Add a new trade to the database"""
        try:
            cursor = self.connection.cursor()
            
            # Encrypt sensitive data if needed
            signals_json = json.dumps(trade.get("signals", []))
            
            cursor.execute("""
                INSERT OR REPLACE INTO trades 
                (id, timestamp, pair, action, price, amount, value, confidence, 
                 signals, status, stop_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["id"],
                trade["timestamp"],
                trade["pair"],
                trade["action"],
                trade["price"],
                trade["amount"],
                trade["value"],
                trade.get("confidence", 0),
                signals_json,
                trade.get("status", "executed"),
                trade.get("stop_loss")
            ))
            
            self.connection.commit()
            logger.info(f"Trade added to database: {trade['id']}")
            
        except Exception as e:
            logger.error(f"Error adding trade to database: {e}")
    
    def get_trades(self, limit: int = 100, pair: str = None, 
                  start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get trades from database"""
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM trades"
            params = []
            conditions = []
            
            if pair:
                conditions.append("pair = ?")
                params.append(pair)
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            trades = []
            for row in rows:
                trade = dict(row)
                # Parse signals JSON
                if trade["signals"]:
                    try:
                        trade["signals"] = json.loads(trade["signals"])
                    except:
                        trade["signals"] = []
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades from database: {e}")
            return []
    
    def update_trade_result(self, trade_id: str, profit: float, profit_percent: float):
        """Update trade result with profit/loss"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                UPDATE trades 
                SET profit = ?, profit_percent = ?
                WHERE id = ?
            """, (profit, profit_percent, trade_id))
            
            self.connection.commit()
            logger.info(f"Trade result updated: {trade_id}")
            
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
    
    def get_daily_stats(self, date_str: str = None) -> Dict:
        """Get daily trading statistics"""
        try:
            if not date_str:
                date_str = datetime.now().strftime("%Y-%m-%d")
            
            cursor = self.connection.cursor()
            
            # Get stats from daily_stats table
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (date_str,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            else:
                # Calculate stats from trades
                cursor.execute("""
                    SELECT 
                        COUNT(*) as trades_count,
                        SUM(profit) as total_profit,
                        AVG(profit_percent) as avg_profit_percent,
                        SUM(value) as total_volume,
                        SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM trades 
                    WHERE DATE(timestamp) = ?
                """, (date_str,))
                
                stats = cursor.fetchone()
                if stats:
                    trades_count = stats[0] or 0
                    total_profit = stats[1] or 0
                    avg_profit_percent = stats[2] or 0
                    total_volume = stats[3] or 0
                    winning_trades = stats[4] or 0
                    
                    win_rate = (winning_trades / trades_count * 100) if trades_count > 0 else 0
                    
                    return {
                        "date": date_str,
                        "trades_count": trades_count,
                        "profit": total_profit,
                        "profit_percent": avg_profit_percent,
                        "win_rate": win_rate,
                        "volume": total_volume
                    }
                else:
                    return {
                        "date": date_str,
                        "trades_count": 0,
                        "profit": 0,
                        "profit_percent": 0,
                        "win_rate": 0,
                        "volume": 0
                    }
                    
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return {}
    
    def save_daily_stats(self, stats: Dict):
        """Save daily statistics"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats
                (date, trades_count, profit, profit_percent, win_rate, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                stats["date"],
                stats["trades_count"],
                stats["profit"],
                stats["profit_percent"],
                stats["win_rate"],
                stats["volume"]
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error saving daily stats: {e}")
    
    def get_last_daily_reset(self) -> Optional[date]:
        """Get the last daily reset date"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT value FROM config WHERE key = 'last_daily_reset'")
            row = cursor.fetchone()
            
            if row:
                return datetime.fromisoformat(row[0]).date()
            return None
            
        except Exception as e:
            logger.error(f"Error getting last daily reset: {e}")
            return None
    
    def set_last_daily_reset(self, reset_date: date):
        """Set the last daily reset date"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO config (key, value, category)
                VALUES (?, ?, ?)
            """, ("last_daily_reset", reset_date.isoformat(), "system"))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error setting last daily reset: {e}")
    
    def store_market_data(self, pair: str, timeframe: str, data: List[Dict]):
        """Store market data for ML training"""
        try:
            cursor = self.connection.cursor()
            
            # Store encrypted market data
            data_json = json.dumps(data)
            encrypted_data = self.encryption.encrypt(data_json)
            
            cursor.execute("""
                INSERT INTO market_data (pair, timeframe, timestamp, data)
                VALUES (?, ?, ?, ?)
            """, (pair, timeframe, datetime.now().isoformat(), encrypted_data))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    def get_market_data(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get stored market data"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT data FROM market_data
                WHERE pair = ? AND timeframe = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (pair, timeframe, limit))
            
            rows = cursor.fetchall()
            market_data = []
            
            for row in rows:
                try:
                    decrypted_data = self.encryption.decrypt(row[0])
                    data = json.loads(decrypted_data)
                    market_data.extend(data)
                except:
                    logger.warning("Failed to decrypt market data")
                    continue
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []
    
    def add_alert(self, alert_type: str, message: str, priority: str = "normal", 
                  channels: List[str] = None):
        """Add alert to database"""
        try:
            cursor = self.connection.cursor()
            
            channels_json = json.dumps(channels or [])
            
            cursor.execute("""
                INSERT INTO alerts (type, message, priority, channels)
                VALUES (?, ?, ?, ?)
            """, (alert_type, message, priority, channels_json))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
    
    def get_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT * FROM alerts
                ORDER BY sent_at DESC LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            alerts = []
            
            for row in rows:
                alert = dict(row)
                if alert["channels"]:
                    try:
                        alert["channels"] = json.loads(alert["channels"])
                    except:
                        alert["channels"] = []
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Get overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(profit) as total_profit,
                    AVG(profit_percent) as avg_profit_percent,
                    MAX(profit) as best_trade,
                    MIN(profit) as worst_trade
                FROM trades
                WHERE timestamp >= date('now', '-{} days')
            """.format(days))
            
            stats = cursor.fetchone()
            
            if stats:
                total_trades = stats[0] or 0
                winning_trades = stats[1] or 0
                total_profit = stats[2] or 0
                avg_profit_percent = stats[3] or 0
                best_trade = stats[4] or 0
                worst_trade = stats[5] or 0
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                return {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": total_trades - winning_trades,
                    "win_rate": round(win_rate, 2),
                    "total_profit": round(total_profit, 2),
                    "avg_profit_percent": round(avg_profit_percent, 2),
                    "best_trade": round(best_trade, 2),
                    "worst_trade": round(worst_trade, 2),
                    "period_days": days
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data from database"""
        try:
            cursor = self.connection.cursor()
            
            # Clean old market data
            cursor.execute("""
                DELETE FROM market_data
                WHERE created_at < date('now', '-{} days')
            """.format(days))
            
            # Clean old alerts
            cursor.execute("""
                DELETE FROM alerts
                WHERE sent_at < date('now', '-30 days')
            """)
            
            self.connection.commit()
            logger.info("Old data cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db_file, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
