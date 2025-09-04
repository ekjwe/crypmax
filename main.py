#!/usr/bin/env python3
"""
QuantumTrade - Advanced Crypto Trading Bot
Main application entry point
"""

import os
import sys
import asyncio
import threading
import time
import logging
from datetime import datetime

from trading_bot import TradingBot
from web_interface import WebInterface
from config import Config
from database import Database
from alert_system import AlertSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class QuantumTradeApp:
    """Main application class for QuantumTrade Bot"""
    
    def __init__(self):
        self.config = Config()
        self.database = Database()
        self.alert_system = AlertSystem(self.config)
        self.trading_bot = TradingBot(self.config, self.database, self.alert_system)
        self.web_interface = WebInterface(self.trading_bot, self.database, self.config)
        self.running = False
        
    def start(self):
        """Start the trading bot and web interface"""
        logger.info("Starting QuantumTrade Bot...")
        
        try:
            # Initialize database
            self.database.initialize()
            logger.info("Database initialized")
            
            # Start trading bot in separate thread
            self.running = True
            bot_thread = threading.Thread(target=self._run_trading_bot, daemon=True)
            bot_thread.start()
            logger.info("Trading bot thread started")
            
            # Start web interface (this blocks)
            logger.info("Starting web interface on port 5000...")
            self.web_interface.run(host='0.0.0.0', port=5000, debug=False)
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            self.stop()
        except Exception as e:
            logger.error(f"Application error: {e}")
            self.restart()
    
    def _run_trading_bot(self):
        """Run the trading bot in an async loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.trading_bot.start())
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
            if self.running:
                self.restart()
        finally:
            loop.close()
    
    def stop(self):
        """Stop the application gracefully"""
        logger.info("Stopping QuantumTrade Bot...")
        self.running = False
        if hasattr(self.trading_bot, 'stop'):
            self.trading_bot.stop()
    
    def restart(self):
        """Restart the application"""
        logger.info("Restarting QuantumTrade Bot...")
        self.stop()
        time.sleep(5)  # Wait before restart
        
        # Restart the application
        os.execv(sys.executable, ['python'] + sys.argv)

def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("QuantumTrade Bot v1.0")
    logger.info("Advanced Crypto Trading Bot")
    logger.info("=" * 50)
    
    app = QuantumTradeApp()
    app.start()

if __name__ == "__main__":
    main()
