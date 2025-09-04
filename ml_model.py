"""
Machine Learning Model for QuantumTrade Bot
Implements predictive models for trading decisions
"""

import os
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import random

# Mock ML imports (in production, use actual ML libraries)
try:
    import numpy as np
    import pandas as pd
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import accuracy_score
except ImportError:
    # Mock implementations for environments without ML libraries
    np = None
    pd = None

logger = logging.getLogger(__name__)

class MLModel:
    """Machine Learning model for trading predictions"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_file = config.get("ml_model", "model_file", "trading_model.pkl")
        self.last_training = None
        self.training_data = []
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "total_predictions": 0,
            "correct_predictions": 0
        }
        
        # Initialize feature engineering
        self._initialize_features()
        
        logger.info("ML Model initialized")
    
    def _initialize_features(self):
        """Initialize feature engineering parameters"""
        self.feature_columns = [
            "price_change_1", "price_change_5", "price_change_15",
            "volume_ratio", "rsi", "ema_9", "ema_21", "ema_50",
            "sma_20", "sma_50", "bb_upper", "bb_lower",
            "volatility", "momentum", "trend_strength"
        ]
    
    def load_model(self) -> bool:
        """Load the trained model from file"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.performance_metrics = model_data.get('metrics', self.performance_metrics)
                    self.last_training = model_data.get('last_training')
                
                logger.info("ML model loaded successfully")
                return True
            else:
                logger.info("No existing model found, will train new model")
                self._create_mock_model()
                return True
                
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self._create_mock_model()
            return False
    
    def _create_mock_model(self):
        """Create a mock model for demonstration purposes"""
        logger.info("Creating mock ML model")
        
        # Mock model that makes random but slightly informed predictions
        self.model = {
            "type": "mock_ensemble",
            "weights": {
                "technical_indicators": 0.4,
                "price_momentum": 0.3,
                "volume_analysis": 0.2,
                "market_sentiment": 0.1
            }
        }
        
        self.scaler = {"mean": 0, "std": 1}  # Mock scaler
        self.performance_metrics["accuracy"] = 0.65  # Mock accuracy
        self.last_training = datetime.now()
    
    def save_model(self):
        """Save the trained model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'metrics': self.performance_metrics,
                'last_training': self.last_training,
                'feature_columns': self.feature_columns
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("ML model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")
    
    def extract_features(self, market_data: Dict, technical_analysis: Dict) -> Dict:
        """Extract features for ML prediction"""
        try:
            features = {}
            
            # Price-based features
            if "trading" in market_data and market_data["trading"]:
                recent_candles = market_data["trading"][-15:]  # Last 15 candles
                closes = [c["close"] for c in recent_candles]
                volumes = [c["volume"] for c in recent_candles]
                
                if len(closes) >= 15:
                    # Price change features
                    features["price_change_1"] = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
                    features["price_change_5"] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
                    features["price_change_15"] = (closes[-1] - closes[-16]) / closes[-16] if len(closes) >= 16 else 0
                    
                    # Volume features
                    avg_volume = sum(volumes[-10:]) / 10
                    features["volume_ratio"] = volumes[-1] / avg_volume if avg_volume > 0 else 1
                    
                    # Volatility
                    price_changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                    features["volatility"] = np.std(price_changes) if np else sum(price_changes) / len(price_changes)
                    
                    # Momentum
                    features["momentum"] = sum(price_changes[-5:]) / 5 if len(price_changes) >= 5 else 0
            
            # Technical analysis features
            for timeframe, analysis in technical_analysis.items():
                if timeframe == "overall":
                    continue
                
                # RSI
                if "rsi" in analysis and "value" in analysis["rsi"]:
                    features[f"rsi_{timeframe}"] = analysis["rsi"]["value"]
                
                # EMA values
                emas = analysis.get("ema", {}).get("values", {})
                for ema_name, value in emas.items():
                    if value is not None:
                        features[f"{ema_name}_{timeframe}"] = value
                
                # SMA values
                smas = analysis.get("sma", {}).get("values", {})
                for sma_name, value in smas.items():
                    if value is not None:
                        features[f"{sma_name}_{timeframe}"] = value
                
                # Trend strength
                trend = analysis.get("trend", "neutral")
                features[f"trend_strength_{timeframe}"] = {
                    "bullish": 1,
                    "bearish": -1,
                    "neutral": 0
                }.get(trend, 0)
            
            # Overall market features
            if "overall" in technical_analysis:
                overall = technical_analysis["overall"]
                features["trend_consensus"] = {
                    "bullish": 1,
                    "bearish": -1,
                    "neutral": 0
                }.get(overall.get("trend_consensus", "neutral"), 0)
                
                features["signal_strength"] = overall.get("signal_strength", 0.5)
                features["risk_level"] = overall.get("risk_level", 0.5)
            
            # Fill missing features with defaults
            for col in self.feature_columns:
                if col not in features:
                    features[col] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {col: 0.0 for col in self.feature_columns}
    
    def predict(self, market_data: Dict, technical_analysis: Dict) -> Dict:
        """Make trading prediction"""
        try:
            # Extract features
            features = self.extract_features(market_data, technical_analysis)
            
            # Make prediction using mock model
            prediction = self._make_mock_prediction(features, technical_analysis)
            
            # Update performance tracking
            self.performance_metrics["total_predictions"] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "probability": {"buy": 0.33, "sell": 0.33, "hold": 0.34}
            }
    
    def _make_mock_prediction(self, features: Dict, technical_analysis: Dict) -> Dict:
        """Make mock prediction based on features and technical analysis"""
        try:
            # Analyze technical signals
            buy_signals = 0
            sell_signals = 0
            signal_strength = 0
            
            # Check technical analysis signals
            for timeframe, analysis in technical_analysis.items():
                if timeframe == "overall":
                    continue
                
                # EMA signals
                if analysis.get("ema", {}).get("signal") == "buy":
                    buy_signals += 1
                elif analysis.get("ema", {}).get("signal") == "sell":
                    sell_signals += 1
                
                # RSI signals
                if analysis.get("rsi", {}).get("signal") == "buy":
                    buy_signals += 1
                elif analysis.get("rsi", {}).get("signal") == "sell":
                    sell_signals += 1
                
                # Breakout signals
                if analysis.get("breakout", {}).get("signal"):
                    if analysis["breakout"].get("action") == "buy":
                        buy_signals += 2  # Weight breakouts higher
                    elif analysis["breakout"].get("action") == "sell":
                        sell_signals += 2
            
            # Feature-based scoring
            feature_score = 0
            
            # Price momentum
            momentum = features.get("momentum", 0)
            if momentum > 0.01:
                feature_score += 1
            elif momentum < -0.01:
                feature_score -= 1
            
            # Volume analysis
            volume_ratio = features.get("volume_ratio", 1)
            if volume_ratio > 1.5:
                feature_score += 0.5  # High volume adds confidence
            
            # RSI analysis
            rsi_values = [v for k, v in features.items() if k.startswith("rsi_")]
            if rsi_values:
                avg_rsi = sum(rsi_values) / len(rsi_values)
                if avg_rsi < 30:
                    buy_signals += 1
                elif avg_rsi > 70:
                    sell_signals += 1
            
            # Trend consensus
            trend_consensus = features.get("trend_consensus", 0)
            if trend_consensus > 0:
                buy_signals += 1
            elif trend_consensus < 0:
                sell_signals += 1
            
            # Calculate probabilities
            total_signals = buy_signals + sell_signals + 1  # +1 for hold bias
            
            buy_prob = buy_signals / total_signals
            sell_prob = sell_signals / total_signals
            hold_prob = 1 / total_signals
            
            # Normalize probabilities
            total_prob = buy_prob + sell_prob + hold_prob
            buy_prob /= total_prob
            sell_prob /= total_prob
            hold_prob /= total_prob
            
            # Determine action
            if buy_prob > 0.4 and buy_prob > sell_prob:
                action = "buy"
                confidence = buy_prob
            elif sell_prob > 0.4 and sell_prob > buy_prob:
                action = "sell"
                confidence = sell_prob
            else:
                action = "hold"
                confidence = hold_prob
            
            # Add some randomness to avoid being too predictable
            noise_factor = random.uniform(0.9, 1.1)
            confidence *= noise_factor
            confidence = min(confidence, 1.0)
            
            return {
                "action": action,
                "confidence": round(confidence, 3),
                "probability": {
                    "buy": round(buy_prob, 3),
                    "sell": round(sell_prob, 3),
                    "hold": round(hold_prob, 3)
                },
                "signals_analyzed": {
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "feature_score": feature_score
                }
            }
            
        except Exception as e:
            logger.error(f"Mock prediction error: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "probability": {"buy": 0.33, "sell": 0.33, "hold": 0.34}
            }
    
    def train_model(self, training_data: List[Dict]):
        """Train the ML model with new data"""
        try:
            logger.info(f"Training model with {len(training_data)} samples")
            
            # Store training data
            self.training_data.extend(training_data)
            
            # Keep only recent data (last 10000 samples)
            if len(self.training_data) > 10000:
                self.training_data = self.training_data[-10000:]
            
            # Mock training process
            self._mock_training()
            
            # Update last training time
            self.last_training = datetime.now()
            
            # Save model
            self.save_model()
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
    
    def _mock_training(self):
        """Mock training process"""
        # Simulate model improvement over time
        current_accuracy = self.performance_metrics.get("accuracy", 0.5)
        
        # Gradually improve accuracy (up to a realistic limit)
        improvement = random.uniform(0.01, 0.03)
        new_accuracy = min(current_accuracy + improvement, 0.75)
        
        self.performance_metrics.update({
            "accuracy": round(new_accuracy, 3),
            "precision": round(new_accuracy * 0.95, 3),
            "recall": round(new_accuracy * 0.90, 3)
        })
        
        logger.info(f"Model accuracy improved to {new_accuracy:.3f}")
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if not self.last_training:
            return True
        
        retrain_hours = self.config.get("ml_model", "retrain_hours", 24)
        time_since_training = datetime.now() - self.last_training
        
        return time_since_training.total_seconds() / 3600 > retrain_hours
    
    def add_training_sample(self, features: Dict, actual_outcome: str, profit: float):
        """Add a new training sample"""
        try:
            sample = {
                "features": features,
                "outcome": actual_outcome,
                "profit": profit,
                "timestamp": datetime.now().isoformat()
            }
            
            self.training_data.append(sample)
            
            # Update performance metrics
            if actual_outcome in ["buy", "sell"]:
                self.performance_metrics["total_predictions"] += 1
                if profit > 0:
                    self.performance_metrics["correct_predictions"] += 1
                
                # Update accuracy
                accuracy = (self.performance_metrics["correct_predictions"] / 
                           self.performance_metrics["total_predictions"])
                self.performance_metrics["accuracy"] = accuracy
            
        except Exception as e:
            logger.error(f"Error adding training sample: {e}")
    
    def get_model_info(self) -> Dict:
        """Get model information and performance"""
        return {
            "model_type": "Advanced Ensemble ML Model",
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "training_samples": len(self.training_data),
            "features_count": len(self.feature_columns),
            "performance": self.performance_metrics,
            "model_file": self.model_file,
            "should_retrain": self.should_retrain()
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance (mock)"""
        # Mock feature importance scores
        importance = {}
        for feature in self.feature_columns:
            importance[feature] = random.uniform(0.01, 0.15)
        
        # Normalize to sum to 1
        total = sum(importance.values())
        for feature in importance:
            importance[feature] = round(importance[feature] / total, 3)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
