"""
Risk Management module for QuantumTrade Bot
Handles position sizing, stop losses, and risk calculations
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management engine for trading operations"""
    
    def __init__(self, config):
        self.config = config
        self.max_position_size = 0.1  # 10% of portfolio per trade
        self.max_daily_risk = 0.05    # 5% daily risk limit
        self.volatility_lookback = 20  # Days for volatility calculation
        
        logger.info("Risk manager initialized")
    
    def calculate_position_size(self, investment_amount: float, current_price: float, 
                              confidence: float) -> float:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Base position size
            base_size = investment_amount / current_price
            
            # Adjust for confidence level (0.1 to 1.0)
            confidence_multiplier = max(0.1, min(1.0, confidence))
            
            # Apply Kelly Criterion principles
            kelly_multiplier = self._calculate_kelly_multiplier(confidence)
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * kelly_multiplier
            
            # Apply maximum position limits
            max_allowed = self._get_max_position_size(investment_amount, current_price)
            position_size = min(position_size, max_allowed)
            
            # Ensure minimum viable position
            min_position = 0.001  # Minimum position size
            position_size = max(position_size, min_position)
            
            logger.info(f"Position size calculated: {position_size:.6f} (confidence: {confidence:.2f})")
            return round(position_size, 6)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return investment_amount / current_price
    
    def _calculate_kelly_multiplier(self, confidence: float) -> float:
        """Calculate Kelly Criterion multiplier"""
        try:
            # Simplified Kelly formula: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            
            # Map confidence to win probability
            win_prob = 0.4 + (confidence * 0.4)  # 40% to 80% based on confidence
            loss_prob = 1 - win_prob
            
            # Average odds (simplified)
            avg_win = 1.5  # Average 50% gain on winning trades
            avg_loss = 1.0  # Average 100% loss (stop loss)
            
            # Kelly multiplier
            kelly = (avg_win * win_prob - loss_prob) / avg_win
            
            # Conservative approach - use 25% of Kelly recommendation
            kelly_multiplier = max(0.1, min(0.5, kelly * 0.25))
            
            return kelly_multiplier
            
        except Exception as e:
            logger.error(f"Kelly multiplier calculation error: {e}")
            return 0.3  # Conservative default
    
    def _get_max_position_size(self, investment_amount: float, current_price: float) -> float:
        """Get maximum allowed position size"""
        try:
            # Maximum based on investment amount
            max_investment = self.config.get("trading", "max_investment", 1000.0)
            max_from_investment = min(investment_amount, max_investment) / current_price
            
            # Maximum based on portfolio percentage
            # (In production, this would use actual portfolio value)
            estimated_portfolio = 10000.0  # Mock portfolio value
            max_from_portfolio = (estimated_portfolio * self.max_position_size) / current_price
            
            return min(max_from_investment, max_from_portfolio)
            
        except Exception as e:
            logger.error(f"Max position size calculation error: {e}")
            return investment_amount / current_price
    
    def calculate_stop_loss(self, entry_price: float, action: str, volatility: float = None) -> float:
        """Calculate stop loss price"""
        try:
            # Get stop loss percentage from config
            stop_loss_percent = self.config.get("trading", "stop_loss_percent", 2.0)
            
            # Adjust for volatility if provided
            if volatility:
                # Increase stop loss for high volatility assets
                volatility_adjustment = min(volatility * 100, 2.0)  # Max 2% additional
                stop_loss_percent += volatility_adjustment
            
            # Calculate stop loss price
            if action.lower() == "buy":
                stop_loss = entry_price * (1 - stop_loss_percent / 100)
            else:  # sell
                stop_loss = entry_price * (1 + stop_loss_percent / 100)
            
            logger.info(f"Stop loss calculated: {stop_loss:.2f} ({stop_loss_percent:.2f}%)")
            return round(stop_loss, 2)
            
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            return entry_price * 0.98 if action.lower() == "buy" else entry_price * 1.02
    
    def calculate_take_profit(self, entry_price: float, action: str, risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit price based on risk-reward ratio"""
        try:
            stop_loss_percent = self.config.get("trading", "stop_loss_percent", 2.0)
            take_profit_percent = stop_loss_percent * risk_reward_ratio
            
            if action.lower() == "buy":
                take_profit = entry_price * (1 + take_profit_percent / 100)
            else:  # sell
                take_profit = entry_price * (1 - take_profit_percent / 100)
            
            return round(take_profit, 2)
            
        except Exception as e:
            logger.error(f"Take profit calculation error: {e}")
            return entry_price * 1.04 if action.lower() == "buy" else entry_price * 0.96
    
    def assess_market_risk(self, market_data: Dict, technical_analysis: Dict) -> float:
        """Assess overall market risk level (0-1)"""
        try:
            risk_factors = []
            
            # Volatility risk
            if "trading" in market_data and market_data["trading"]:
                volatility_risk = self._calculate_volatility_risk(market_data["trading"])
                risk_factors.append(volatility_risk)
            
            # Volume risk
            volume_risk = self._calculate_volume_risk(market_data)
            risk_factors.append(volume_risk)
            
            # Technical risk
            technical_risk = self._calculate_technical_risk(technical_analysis)
            risk_factors.append(technical_risk)
            
            # Market trend risk
            trend_risk = self._calculate_trend_risk(technical_analysis)
            risk_factors.append(trend_risk)
            
            # Calculate overall risk (weighted average)
            weights = [0.3, 0.2, 0.3, 0.2]  # Volatility, Volume, Technical, Trend
            overall_risk = sum(risk * weight for risk, weight in zip(risk_factors, weights))
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception as e:
            logger.error(f"Market risk assessment error: {e}")
            return 0.5  # Medium risk default
    
    def _calculate_volatility_risk(self, price_data: list) -> float:
        """Calculate risk based on price volatility"""
        try:
            if len(price_data) < 10:
                return 0.5
            
            # Calculate price changes
            closes = [candle["close"] for candle in price_data[-20:]]
            price_changes = []
            
            for i in range(1, len(closes)):
                change = (closes[i] - closes[i-1]) / closes[i-1]
                price_changes.append(abs(change))
            
            # Calculate average volatility
            avg_volatility = sum(price_changes) / len(price_changes)
            
            # Normalize to 0-1 scale (assuming 5% daily volatility is high risk)
            volatility_risk = min(1.0, avg_volatility / 0.05)
            
            return volatility_risk
            
        except Exception as e:
            logger.error(f"Volatility risk calculation error: {e}")
            return 0.5
    
    def _calculate_volume_risk(self, market_data: Dict) -> float:
        """Calculate risk based on volume patterns"""
        try:
            if "trading" not in market_data or not market_data["trading"]:
                return 0.5
            
            volumes = [candle["volume"] for candle in market_data["trading"][-10:]]
            
            if len(volumes) < 2:
                return 0.5
            
            # Calculate volume consistency
            avg_volume = sum(volumes) / len(volumes)
            volume_std = math.sqrt(sum((v - avg_volume) ** 2 for v in volumes) / len(volumes))
            
            # High volume inconsistency = higher risk
            volume_risk = min(1.0, volume_std / avg_volume) if avg_volume > 0 else 0.5
            
            return volume_risk
            
        except Exception as e:
            logger.error(f"Volume risk calculation error: {e}")
            return 0.5
    
    def _calculate_technical_risk(self, technical_analysis: Dict) -> float:
        """Calculate risk based on technical indicators"""
        try:
            risk_factors = []
            
            for timeframe, analysis in technical_analysis.items():
                if timeframe == "overall":
                    continue
                
                # RSI risk (extreme values indicate higher risk)
                rsi_value = analysis.get("rsi", {}).get("value", 50)
                if rsi_value < 20 or rsi_value > 80:
                    risk_factors.append(0.8)  # High risk
                elif rsi_value < 30 or rsi_value > 70:
                    risk_factors.append(0.6)  # Medium risk
                else:
                    risk_factors.append(0.3)  # Low risk
                
                # Support/resistance risk
                sr = analysis.get("support_resistance", {})
                if sr.get("support") and sr.get("resistance"):
                    current_price = 45000  # Mock current price
                    support = sr["support"]
                    resistance = sr["resistance"]
                    
                    # Risk increases near support/resistance levels
                    if abs(current_price - support) / support < 0.02:
                        risk_factors.append(0.7)
                    elif abs(current_price - resistance) / resistance < 0.02:
                        risk_factors.append(0.7)
                    else:
                        risk_factors.append(0.4)
            
            return sum(risk_factors) / len(risk_factors) if risk_factors else 0.5
            
        except Exception as e:
            logger.error(f"Technical risk calculation error: {e}")
            return 0.5
    
    def _calculate_trend_risk(self, technical_analysis: Dict) -> float:
        """Calculate risk based on trend analysis"""
        try:
            if "overall" not in technical_analysis:
                return 0.5
            
            overall = technical_analysis["overall"]
            trend_consensus = overall.get("trend_consensus", "neutral")
            signal_strength = overall.get("signal_strength", 0.5)
            
            # Strong trends = lower risk, weak/conflicting trends = higher risk
            if trend_consensus in ["bullish", "bearish"] and signal_strength > 0.7:
                return 0.3  # Low risk
            elif trend_consensus == "neutral" or signal_strength < 0.3:
                return 0.8  # High risk
            else:
                return 0.5  # Medium risk
                
        except Exception as e:
            logger.error(f"Trend risk calculation error: {e}")
            return 0.5
    
    def check_daily_risk_limits(self, current_loss: float, portfolio_value: float) -> bool:
        """Check if daily risk limits are exceeded"""
        try:
            daily_loss_percent = abs(current_loss) / portfolio_value
            max_daily_risk = self.config.get("trading", "max_loss_percent", 5.0) / 100
            
            if daily_loss_percent >= max_daily_risk:
                logger.warning(f"Daily risk limit exceeded: {daily_loss_percent:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Daily risk check error: {e}")
            return True
    
    def calculate_sharpe_ratio(self, returns: list, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for performance measurement"""
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            # Calculate average return and standard deviation
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = math.sqrt(variance)
            
            if std_dev == 0:
                return 0.0
            
            # Sharpe ratio = (average return - risk free rate) / standard deviation
            sharpe = (avg_return - risk_free_rate / 252) / std_dev  # Daily risk-free rate
            
            return round(sharpe, 3)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: list) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            if not equity_curve or len(equity_curve) < 2:
                return 0.0
            
            peak = equity_curve[0]
            max_drawdown = 0.0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return round(max_drawdown * 100, 2)  # Return as percentage
            
        except Exception as e:
            logger.error(f"Max drawdown calculation error: {e}")
            return 0.0
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk management metrics"""
        return {
            "max_position_size": self.max_position_size * 100,  # As percentage
            "max_daily_risk": self.max_daily_risk * 100,
            "stop_loss_percent": self.config.get("trading", "stop_loss_percent", 2.0),
            "risk_free_rate": 2.0,  # Annual risk-free rate
            "volatility_lookback": self.volatility_lookback
        }
    
    def adjust_risk_parameters(self, performance_data: Dict):
        """Dynamically adjust risk parameters based on performance"""
        try:
            win_rate = performance_data.get("win_rate", 50)
            sharpe_ratio = performance_data.get("sharpe_ratio", 0)
            max_drawdown = performance_data.get("max_drawdown", 0)
            
            # Adjust position sizing based on performance
            if win_rate > 60 and sharpe_ratio > 1.0:
                # Good performance - slightly increase position size
                self.max_position_size = min(0.15, self.max_position_size * 1.1)
            elif win_rate < 40 or max_drawdown > 10:
                # Poor performance - reduce position size
                self.max_position_size = max(0.05, self.max_position_size * 0.9)
            
            logger.info(f"Risk parameters adjusted: max_position_size={self.max_position_size:.2%}")
            
        except Exception as e:
            logger.error(f"Risk parameter adjustment error: {e}")
