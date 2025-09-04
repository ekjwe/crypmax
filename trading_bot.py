"""
Core trading bot engine with advanced strategies and machine learning
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from technical_analysis import TechnicalAnalysis
from data_feed import DataFeed
from ml_model import MLModel
from risk_management import RiskManager
from database import Database
from alert_system import AlertSystem
from config import Config

logger = logging.getLogger(__name__)

class TradingBot:
    """Advanced crypto trading bot with multiple strategies"""
    
    def __init__(self, config: Config, database: Database, alert_system: AlertSystem):
        self.config = config
        self.database = database
        self.alert_system = alert_system
        
        # Initialize components
        self.data_feed = DataFeed(config.data if hasattr(config, 'data') else None)
        self.technical_analysis = TechnicalAnalysis(config)
        self.ml_model = MLModel(config)
        self.risk_manager = RiskManager(config)
        
        # Bot state
        self.running = False
        self.current_pair = config.get("trading", "default_pair")
        self.investment_amount = config.get("trading", "default_investment")
        self.last_trade_time = None
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.positions = {}
        self.trade_history = []
        
        # Performance tracking
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0
        }
        
        logger.info("Trading bot initialized")
    
    async def start(self):
        """Start the trading bot main loop"""
        self.running = True
        logger.info("Trading bot started")
        
        # Load ML model
        self.ml_model.load_model()
        
        # Reset daily stats at startup
        self._reset_daily_stats_if_needed()
        
        try:
            while self.running:
                await self._trading_cycle()
                await asyncio.sleep(60)  # 1 minute refresh rate
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
            await self.alert_system.send_alert(f"Trading bot error: {e}", "error")
            raise
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("Trading bot stopped")
    
    async def _trading_cycle(self):
        """Main trading cycle"""
        try:
            # Check if trading is allowed
            if not self._can_trade():
                return
            
            # Get market data for all timeframes
            market_data = await self._get_market_data()
            if not market_data:
                return
            
            # Perform technical analysis
            analysis = self._perform_analysis(market_data)
            
            # Get ML prediction
            ml_signal = self.ml_model.predict(market_data, analysis)
            
            # Check news sentiment (mock for now)
            news_sentiment = self._check_news_sentiment()
            
            # Generate trading signal
            signal = self._generate_trading_signal(analysis, ml_signal, news_sentiment)
            
            # Execute trade if signal is strong enough
            if signal["action"] != "hold":
                await self._execute_trade(signal, market_data)
            
            # Update statistics
            self._update_stats()
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed based on limits and cooldowns"""
        # Check daily trade limit
        if self.daily_trades >= self.config.get("trading", "daily_trade_limit"):
            return False
        
        # Check cooldown period
        cooldown_minutes = self.config.get("trading", "cooldown_minutes")
        if (self.last_trade_time and 
            datetime.now() - self.last_trade_time < timedelta(minutes=cooldown_minutes)):
            return False
        
        # Check daily profit target
        daily_target = self.config.get("trading", "daily_profit_target")
        if daily_target != float('inf') and self.daily_profit >= daily_target:
            return False
        
        # Check max loss limit
        max_loss = self.config.get("trading", "max_loss_percent")
        if max_loss > 0 and self.daily_profit <= -max_loss:
            return False
        
        return True
    
    async def _get_market_data(self) -> Optional[Dict]:
        """Get market data for current trading pair"""
        try:
            # Get current market data
            current_data = self.data_feed.get_current_data(self.current_pair)
            
            # Get historical data for analysis
            trading_data = self.data_feed.get_historical_data(
                self.current_pair, 
                "1m",
                limit=100
            )
            
            # Get data for different timeframes
            analysis_data = {}
            for timeframe in ["1m", "3m", "15m", "30m", "1h"]:
                data = self.data_feed.get_historical_data(self.current_pair, timeframe, limit=100)
                analysis_data[timeframe] = data
            
            return {
                "trading": trading_data,
                "analysis": analysis_data,
                "current_price": current_data.get("price"),
                "volume": current_data.get("volume"),
                "current_data": current_data
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def _perform_analysis(self, market_data: Dict) -> Dict:
        """Perform comprehensive technical analysis"""
        analysis = {}
        
        try:
            # Analyze each timeframe
            for timeframe, data in market_data["analysis"].items():
                tf_analysis = {
                    "ema": self.technical_analysis.calculate_ema(data),
                    "rsi": self.technical_analysis.calculate_rsi(data),
                    "sma": self.technical_analysis.calculate_sma(data),
                    "support_resistance": self.technical_analysis.find_support_resistance(data),
                    "candlestick_patterns": self.technical_analysis.detect_candlestick_patterns(data),
                    "volume_analysis": self.technical_analysis.analyze_volume(data),
                    "trend": self.technical_analysis.identify_trend(data),
                    "breakout": self.technical_analysis.detect_breakout(data)
                }
                analysis[timeframe] = tf_analysis
            
            # Overall market analysis
            analysis["overall"] = {
                "trend_consensus": self._get_trend_consensus(analysis),
                "signal_strength": self._calculate_signal_strength(analysis),
                "risk_level": self.risk_manager.assess_market_risk(market_data, analysis)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {}
    
    def _check_news_sentiment(self) -> str:
        """Check news sentiment (mock implementation for now)"""
        # TODO: Implement real news sentiment analysis
        # For now, return neutral
        return "neutral"
    
    def _generate_trading_signal(self, analysis: Dict, ml_signal: Dict, news_sentiment: str) -> Dict:
        """Generate comprehensive trading signal"""
        signals = []
        
        try:
            # Technical analysis signals
            for timeframe, ta in analysis.items():
                if timeframe == "overall":
                    continue
                    
                # EMA crossover signal
                if ta.get("ema", {}).get("signal"):
                    signals.append({
                        "type": "ema_crossover",
                        "action": ta["ema"]["signal"],
                        "strength": 0.7,
                        "timeframe": timeframe
                    })
                
                # RSI signal
                rsi_value = ta.get("rsi", {}).get("value", 50)
                if rsi_value < 30:
                    signals.append({
                        "type": "rsi_oversold",
                        "action": "buy",
                        "strength": 0.8,
                        "timeframe": timeframe
                    })
                elif rsi_value > 70:
                    signals.append({
                        "type": "rsi_overbought",
                        "action": "sell",
                        "strength": 0.8,
                        "timeframe": timeframe
                    })
                
                # Breakout signal
                if ta.get("breakout", {}).get("signal"):
                    signals.append({
                        "type": "breakout",
                        "action": ta["breakout"]["action"],
                        "strength": 0.9,
                        "timeframe": timeframe
                    })
                
                # Candlestick pattern signals
                patterns = ta.get("candlestick_patterns", [])
                for pattern in patterns:
                    if pattern.get("signal"):
                        signals.append({
                            "type": "candlestick",
                            "action": pattern["action"],
                            "strength": pattern.get("strength", 0.6),
                            "pattern": pattern["name"]
                        })
            
            # ML model signal
            if ml_signal.get("action") != "hold":
                signals.append({
                    "type": "ml_prediction",
                    "action": ml_signal["action"],
                    "strength": ml_signal.get("confidence", 0.5)
                })
            
            # News sentiment modifier
            sentiment_modifier = {
                "positive": 1.1,
                "neutral": 1.0,
                "negative": 0.9
            }.get(news_sentiment, 1.0)
            
            # Calculate final signal
            return self._calculate_final_signal(signals, sentiment_modifier)
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return {"action": "hold", "confidence": 0.0}
    
    def _calculate_final_signal(self, signals: List[Dict], sentiment_modifier: float) -> Dict:
        """Calculate final trading signal from all indicators"""
        if not signals:
            return {"action": "hold", "confidence": 0.0}
        
        buy_score = 0.0
        sell_score = 0.0
        
        for signal in signals:
            strength = signal["strength"] * sentiment_modifier
            if signal["action"] == "buy":
                buy_score += strength
            elif signal["action"] == "sell":
                sell_score += strength
        
        # Determine final action
        if buy_score > sell_score and buy_score > 1.5:
            return {
                "action": "buy",
                "confidence": min(buy_score / len(signals), 1.0),
                "signals": signals
            }
        elif sell_score > buy_score and sell_score > 1.5:
            return {
                "action": "sell",
                "confidence": min(sell_score / len(signals), 1.0),
                "signals": signals
            }
        else:
            return {
                "action": "hold",
                "confidence": 0.0,
                "signals": signals
            }
    
    async def _execute_trade(self, signal: Dict, market_data: Dict):
        """Execute trade based on signal"""
        try:
            current_price = market_data["current_price"]
            action = signal["action"]
            confidence = signal["confidence"]
            
            # Calculate position size based on risk management
            position_size = self.risk_manager.calculate_position_size(
                self.investment_amount, current_price, confidence
            )
            
            # Create trade order
            trade = {
                "id": f"trade_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "pair": self.current_pair,
                "action": action,
                "price": current_price,
                "amount": position_size,
                "value": position_size * current_price,
                "confidence": confidence,
                "signals": signal.get("signals", []),
                "status": "executed"
            }
            
            # Set stop loss
            if action == "buy":
                stop_loss_percent = self.config.get("trading", "stop_loss_percent")
                trade["stop_loss"] = current_price * (1 - stop_loss_percent / 100)
            
            # Execute the trade (real or simulated)
            success = await self._execute_real_trade(trade)
            
            if success:
                # Record trade
                self.database.add_trade(trade)
                self.trade_history.append(trade)
                self.daily_trades += 1
                self.last_trade_time = datetime.now()
                
                # Send alert
                await self.alert_system.send_trade_alert(trade)
                
                logger.info(f"Trade executed: {action} {position_size} {self.current_pair} at {current_price}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            await self.alert_system.send_alert(f"Trade execution failed: {e}", "error")
    
    async def _execute_real_trade(self, trade: Dict) -> bool:
        """Execute real trade through data feed"""
        try:
            # Execute trade through data feed (Binance or simulation)
            result = self.data_feed.execute_trade(
                action=trade["action"],
                quantity=trade["amount"],
                symbol=trade["pair"],
                price=trade.get("price")
            )
            
            if result.get('success'):
                # Update trade record with real execution details
                trade["order_id"] = result.get("order_id", "N/A")
                trade["executed_price"] = result.get("price", trade["price"])
                trade["fees"] = result.get("fees", 0)
                trade["source"] = result.get("source", "unknown")
                
                logger.info(f"✓ Trade executed via {trade['source']}: {trade['action']} {trade['amount']} {trade['pair']}")
                return True
            else:
                logger.error(f"Trade execution failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    
    def _get_trend_consensus(self, analysis: Dict) -> str:
        """Get trend consensus across timeframes"""
        trends = []
        for timeframe, ta in analysis.items():
            if timeframe != "overall" and ta.get("trend"):
                trends.append(ta["trend"])
        
        if not trends:
            return "neutral"
        
        bullish = trends.count("bullish")
        bearish = trends.count("bearish")
        
        if bullish > bearish:
            return "bullish"
        elif bearish > bullish:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_signal_strength(self, analysis: Dict) -> float:
        """Calculate overall signal strength"""
        strengths = []
        for timeframe, ta in analysis.items():
            if timeframe != "overall":
                # Add various strength indicators
                if ta.get("rsi", {}).get("extreme"):
                    strengths.append(0.8)
                if ta.get("breakout", {}).get("confirmed"):
                    strengths.append(0.9)
                if ta.get("volume_analysis", {}).get("spike"):
                    strengths.append(0.7)
        
        return sum(strengths) / len(strengths) if strengths else 0.5
    
    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if it's a new day"""
        last_reset = self.database.get_last_daily_reset()
        today = datetime.now().date()
        
        if not last_reset or last_reset < today:
            self.daily_trades = 0
            self.daily_profit = 0.0
            self.database.set_last_daily_reset(today)
            logger.info("Daily statistics reset")
    
    def _update_stats(self):
        """Update trading statistics"""
        # Calculate current portfolio value and profit
        # This would be implemented with real exchange API
        pass
    
    def get_current_status(self) -> Dict:
        """Get current bot status"""
        return {
            "running": self.running,
            "current_pair": self.current_pair,
            "investment_amount": self.investment_amount,
            "daily_trades": self.daily_trades,
            "daily_profit": self.daily_profit,
            "stats": self.stats,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None
        }
    
    def update_config(self, updates: Dict):
        """Update bot configuration"""
        if "pair" in updates:
            self.current_pair = updates["pair"]
        if "investment_amount" in updates:
            self.investment_amount = max(1.0, float(updates["investment_amount"]))
        
        logger.info(f"Bot configuration updated: {updates}")
