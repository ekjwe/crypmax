"""
Web Interface for QuantumTrade Bot
Flask-based dashboard for monitoring and controlling the trading bot
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio
import threading

logger = logging.getLogger(__name__)

class WebInterface:
    """Flask web interface for the trading bot"""
    
    def __init__(self, trading_bot, database, config):
        self.trading_bot = trading_bot
        self.database = database
        self.config = config
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.secret_key = config.get("api", "secret_key", "your-secret-key-here")
        
        # Register routes
        self._register_routes()
        
        logger.info("Web interface initialized")
    
    def _register_routes(self):
        """Register all Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            try:
                # Get bot status
                bot_status = self.trading_bot.get_current_status()
                
                # Get recent trades
                recent_trades = self.database.get_trades(limit=10)
                
                # Get daily stats
                daily_stats = self.database.get_daily_stats()
                
                # Get performance stats
                performance_stats = self.database.get_performance_stats(days=30)
                
                # Get ML model info
                ml_info = self.trading_bot.ml_model.get_model_info()
                
                # Get risk metrics
                risk_metrics = self.trading_bot.risk_manager.get_risk_metrics()
                
                # Get alert status
                alert_status = self.trading_bot.alert_system.get_alert_status()
                
                dashboard_data = {
                    "bot_status": bot_status,
                    "recent_trades": recent_trades,
                    "daily_stats": daily_stats,
                    "performance_stats": performance_stats,
                    "ml_info": ml_info,
                    "risk_metrics": risk_metrics,
                    "alert_status": alert_status,
                    "trading_pairs": self.config.get_trading_pairs(),
                    "current_time": datetime.now().isoformat()
                }
                
                return render_template('index.html', data=dashboard_data)
                
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                return f"Dashboard error: {e}", 500
        
        @self.app.route('/trades')
        def trades():
            """Trades page"""
            try:
                # Get pagination parameters
                page = int(request.args.get('page', 1))
                per_page = 50
                
                # Get filter parameters
                pair = request.args.get('pair')
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')
                
                # Get trades
                trades_data = self.database.get_trades(
                    limit=per_page * 10,  # Get more for filtering
                    pair=pair,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Calculate pagination
                total_trades = len(trades_data)
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                paginated_trades = trades_data[start_idx:end_idx]
                
                # Calculate summary statistics
                if trades_data:
                    total_profit = sum(trade.get('profit', 0) for trade in trades_data)
                    winning_trades = sum(1 for trade in trades_data if trade.get('profit', 0) > 0)
                    win_rate = (winning_trades / len(trades_data)) * 100
                    total_volume = sum(trade.get('value', 0) for trade in trades_data)
                else:
                    total_profit = 0
                    win_rate = 0
                    total_volume = 0
                
                trades_summary = {
                    "total_trades": total_trades,
                    "total_profit": total_profit,
                    "win_rate": win_rate,
                    "total_volume": total_volume,
                    "winning_trades": winning_trades,
                    "losing_trades": total_trades - winning_trades
                }
                
                return render_template('trades.html', 
                                     trades=paginated_trades,
                                     summary=trades_summary,
                                     trading_pairs=self.config.get_trading_pairs(),
                                     current_page=page,
                                     total_pages=max(1, (total_trades + per_page - 1) // per_page),
                                     filters={
                                         'pair': pair,
                                         'start_date': start_date,
                                         'end_date': end_date
                                     })
                
            except Exception as e:
                logger.error(f"Trades page error: {e}")
                return f"Trades page error: {e}", 500
        
        @self.app.route('/api/status')
        def api_status():
            """Get bot status API"""
            try:
                status = self.trading_bot.get_current_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Status API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/trades')
        def api_trades():
            """Get trades API"""
            try:
                limit = int(request.args.get('limit', 20))
                pair = request.args.get('pair')
                
                trades = self.database.get_trades(limit=limit, pair=pair)
                return jsonify(trades)
            except Exception as e:
                logger.error(f"Trades API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/config', methods=['GET', 'POST'])
        def api_config():
            """Get/Update configuration API"""
            try:
                if request.method == 'GET':
                    return jsonify({
                        "trading": self.config.get("trading"),
                        "technical_analysis": self.config.get("technical_analysis"),
                        "pairs": self.config.get_trading_pairs()
                    })
                
                elif request.method == 'POST':
                    updates = request.json
                    
                    # Update trading bot configuration
                    if 'pair' in updates or 'investment_amount' in updates:
                        self.trading_bot.update_config(updates)
                    
                    # Update config file
                    if 'trading' in updates:
                        self.config.update_trading_config(updates['trading'])
                    
                    return jsonify({"status": "success", "message": "Configuration updated"})
                    
            except Exception as e:
                logger.error(f"Config API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/start', methods=['POST'])
        def api_start():
            """Start trading bot API"""
            try:
                if not self.trading_bot.running:
                    # Start bot in background thread
                    bot_thread = threading.Thread(
                        target=lambda: asyncio.run(self.trading_bot.start()), 
                        daemon=True
                    )
                    bot_thread.start()
                    
                    return jsonify({"status": "success", "message": "Trading bot started"})
                else:
                    return jsonify({"status": "info", "message": "Trading bot already running"})
                    
            except Exception as e:
                logger.error(f"Start API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/stop', methods=['POST'])
        def api_stop():
            """Stop trading bot API"""
            try:
                self.trading_bot.stop()
                return jsonify({"status": "success", "message": "Trading bot stopped"})
                
            except Exception as e:
                logger.error(f"Stop API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/stats')
        def api_stats():
            """Get comprehensive statistics API"""
            try:
                # Daily stats
                daily_stats = self.database.get_daily_stats()
                
                # Performance stats
                performance_stats = self.database.get_performance_stats(days=30)
                
                # ML model info
                ml_info = self.trading_bot.ml_model.get_model_info()
                
                # Risk metrics
                risk_metrics = self.trading_bot.risk_manager.get_risk_metrics()
                
                return jsonify({
                    "daily": daily_stats,
                    "performance": performance_stats,
                    "ml_model": ml_info,
                    "risk": risk_metrics
                })
                
            except Exception as e:
                logger.error(f"Stats API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts/test', methods=['POST'])
        def api_test_alerts():
            """Test alert system API"""
            try:
                # Run test in background
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                test_results = loop.run_until_complete(
                    self.trading_bot.alert_system.test_alerts()
                )
                
                loop.close()
                
                return jsonify({
                    "status": "success",
                    "results": test_results
                })
                
            except Exception as e:
                logger.error(f"Alert test API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/market_data/<pair>')
        def api_market_data(pair):
            """Get market data for a specific pair"""
            try:
                # Get recent market data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                market_data = self.trading_bot.data_feed.get_historical_data(pair, "1h", limit=24)
                current_data = self.trading_bot.data_feed.get_current_data(pair)
                
                loop.close()
                
                return jsonify({
                    "ohlcv": market_data,
                    "ticker": current_data
                })
                
            except Exception as e:
                logger.error(f"Market data API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/manual_trade', methods=['POST'])
        def api_manual_trade():
            """Execute manual trade API"""
            try:
                trade_data = request.json
                
                required_fields = ['pair', 'action', 'amount']
                if not all(field in trade_data for field in required_fields):
                    return jsonify({"error": "Missing required fields"}), 400
                
                # Create manual trade signal
                signal = {
                    "action": trade_data['action'],
                    "confidence": 1.0,
                    "signals": [{"type": "manual", "action": trade_data['action'], "strength": 1.0}]
                }
                
                # Get current market data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                current_data = self.trading_bot.data_feed.get_current_data(trade_data['pair'])
                current_price = current_data.get('price') if current_data else None
                if not current_price:
                    return jsonify({"error": "Could not get current price"}), 400
                
                market_data = {"current_price": current_price}
                
                # Execute trade
                loop.run_until_complete(
                    self.trading_bot._execute_trade(signal, market_data)
                )
                
                loop.close()
                
                return jsonify({"status": "success", "message": "Manual trade executed"})
                
            except Exception as e:
                logger.error(f"Manual trade API error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/binance-status')
        def binance_status_page():
            """Binance status page"""
            return render_template('binance_status.html')

        @self.app.route('/api/binance-status')
        def get_binance_status():
            """Get Binance connection status"""
            try:
                status = self.trading_bot.data_feed.get_connection_info()
                
                # Add additional info
                result = {
                    **status,
                    'min_order_value': 10.0,
                    'default_symbol': 'BTCUSDT',
                    'account_info': None
                }
                
                # If connected, try to get account info
                if status.get('connected'):
                    try:
                        balance = self.trading_bot.data_feed.get_account_balance('USDT')
                        result['account_info'] = {
                            'account_type': 'SPOT',
                            'can_trade': True,
                            'active_balances': 1 if balance > 0 else 0
                        }
                    except Exception as e:
                        logger.error(f"Error getting account info: {e}")
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error getting Binance status: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/test-market-data')
        def test_market_data():
            """Test market data connection"""
            try:
                market_data = self.trading_bot.data_feed.get_current_data('BTC/USD')
                
                return jsonify({
                    'success': True,
                    'symbol': market_data['symbol'],
                    'price': market_data['price'],
                    'source': market_data['source'],
                    'volume': market_data.get('volume', 0),
                    'change_24h': market_data.get('change_24h', 0)
                })
            except Exception as e:
                logger.error(f"Market data test failed: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('404.html'), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return render_template('500.html'), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        try:
            logger.info(f"Starting web interface on {host}:{port}")
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Web interface error: {e}")
            raise
