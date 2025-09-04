"""
Alert System for QuantumTrade Bot
Handles email and Telegram notifications
"""

import asyncio
import smtplib
import logging
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests
import os

logger = logging.getLogger(__name__)

class AlertSystem:
    """Alert system for trading notifications"""
    
    def __init__(self, config):
        self.config = config
        self.email_enabled = config.get("alerts", "email_enabled", True)
        self.telegram_enabled = config.get("alerts", "telegram_enabled", True)
        
        # Get credentials from environment variables
        self.email_config = {
            "smtp_server": config.get("alerts", "email_smtp_server", "smtp.gmail.com"),
            "port": config.get("alerts", "email_port", 587),
            "from_email": os.getenv("EMAIL_FROM", config.get("alerts", "email_from", "")),
            "to_email": os.getenv("EMAIL_TO", config.get("alerts", "email_to", "")),
            "password": os.getenv("EMAIL_PASSWORD", config.get("alerts", "email_password", ""))
        }
        
        self.telegram_config = {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", config.get("alerts", "telegram_bot_token", "")),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID", config.get("alerts", "telegram_chat_id", ""))
        }
        
        logger.info("Alert system initialized")
    
    async def send_trade_alert(self, trade: Dict):
        """Send trade execution alert"""
        try:
            subject = f"🚨 Trade Executed - {trade['action'].upper()} {trade['pair']}"
            
            message = self._format_trade_message(trade)
            
            # Send alerts
            await self.send_alert(message, "trade", subject)
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
    
    async def send_alert(self, message: str, alert_type: str = "info", 
                        subject: str = None, priority: str = "normal"):
        """Send alert via all configured channels"""
        try:
            channels_used = []
            
            # Format subject
            if not subject:
                emoji_map = {
                    "trade": "💹",
                    "error": "🚨",
                    "warning": "⚠️",
                    "info": "ℹ️",
                    "profit": "💰",
                    "loss": "📉"
                }
                emoji = emoji_map.get(alert_type, "ℹ️")
                subject = f"{emoji} QuantumTrade Bot - {alert_type.title()}"
            
            # Send email
            if self.email_enabled and self._is_email_configured():
                email_sent = await self._send_email(subject, message)
                if email_sent:
                    channels_used.append("email")
            
            # Send Telegram
            if self.telegram_enabled and self._is_telegram_configured():
                telegram_sent = await self._send_telegram(message)
                if telegram_sent:
                    channels_used.append("telegram")
            
            # Fallback to console if no channels worked
            if not channels_used:
                print(f"\n{subject}\n{message}\n")
                channels_used.append("console")
            
            logger.info(f"Alert sent via: {', '.join(channels_used)}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            print(f"\nALERT: {message}\n")  # Fallback to console
    
    def _format_trade_message(self, trade: Dict) -> str:
        """Format trade information for alerts"""
        try:
            action_emoji = "📈" if trade["action"] == "buy" else "📉"
            
            message = f"""
{action_emoji} **TRADE EXECUTED**

**Pair:** {trade['pair']}
**Action:** {trade['action'].upper()}
**Price:** ${trade['price']:,.2f}
**Amount:** {trade['amount']:.6f}
**Value:** ${trade['value']:,.2f}
**Confidence:** {trade.get('confidence', 0) * 100:.1f}%
**Time:** {trade['timestamp']}

**Signals:**
"""
            
            # Add signal information
            signals = trade.get('signals', [])
            if signals:
                for signal in signals[:5]:  # Show top 5 signals
                    signal_type = signal.get('type', 'unknown')
                    strength = signal.get('strength', 0) * 100
                    message += f"• {signal_type}: {strength:.0f}%\n"
            else:
                message += "• No specific signals recorded\n"
            
            if trade.get('stop_loss'):
                message += f"\n**Stop Loss:** ${trade['stop_loss']:,.2f}"
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Error formatting trade message: {e}")
            return f"Trade executed: {trade.get('action', 'unknown')} {trade.get('pair', 'unknown')}"
    
    async def _send_email(self, subject: str, message: str) -> bool:
        """Send email alert"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            
            # Create HTML and text versions
            html_message = self._convert_to_html(message)
            
            text_part = MIMEText(message, 'plain')
            html_part = MIMEText(html_message, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['from_email'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_email'], text)
            server.quit()
            
            logger.info("Email alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_telegram(self, message: str) -> bool:
        """Send Telegram alert"""
        try:
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_config['chat_id']
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            # Convert markdown to Telegram format
            telegram_message = message.replace('**', '*')
            
            payload = {
                'chat_id': chat_id,
                'text': telegram_message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def _convert_to_html(self, text: str) -> str:
        """Convert formatted text to HTML"""
        html = text.replace('\n', '<br>')
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('*', '<em>').replace('*', '</em>')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                {html}
            </div>
        </body>
        </html>
        """
    
    def _is_email_configured(self) -> bool:
        """Check if email is properly configured"""
        return (self.email_config['from_email'] and 
                self.email_config['to_email'] and 
                self.email_config['password'])
    
    def _is_telegram_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return (self.telegram_config['bot_token'] and 
                self.telegram_config['chat_id'])
    
    async def send_daily_summary(self, stats: Dict):
        """Send daily trading summary"""
        try:
            subject = "📊 Daily Trading Summary"
            
            message = f"""
**DAILY TRADING SUMMARY**
**Date:** {stats.get('date', datetime.now().strftime('%Y-%m-%d'))}

**Performance:**
• Total Trades: {stats.get('trades_count', 0)}
• Win Rate: {stats.get('win_rate', 0):.1f}%
• Total Profit: ${stats.get('profit', 0):,.2f}
• Profit %: {stats.get('profit_percent', 0):.2f}%
• Volume Traded: ${stats.get('volume', 0):,.2f}

**Status:** {'✅ Target Reached' if stats.get('profit', 0) > 0 else '📈 Keep Trading'}
            """.strip()
            
            await self.send_alert(message, "info", subject)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    async def send_error_alert(self, error_message: str, component: str = "system"):
        """Send error alert"""
        try:
            subject = f"🚨 Error in {component.title()}"
            
            message = f"""
**ERROR ALERT**

**Component:** {component}
**Time:** {datetime.now().isoformat()}
**Error:** {error_message}

**Action Required:** Please check the trading bot logs and restart if necessary.
            """.strip()
            
            await self.send_alert(message, "error", subject, "high")
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
    
    async def send_profit_milestone(self, profit: float, milestone: str):
        """Send profit milestone alert"""
        try:
            subject = f"🎉 Profit Milestone Reached!"
            
            message = f"""
**PROFIT MILESTONE**

**Achievement:** {milestone}
**Current Profit:** ${profit:,.2f}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Congratulations! Your trading bot has reached a new milestone! 🚀
            """.strip()
            
            await self.send_alert(message, "profit", subject)
            
        except Exception as e:
            logger.error(f"Error sending profit milestone alert: {e}")
    
    async def test_alerts(self) -> Dict[str, bool]:
        """Test all alert channels"""
        test_results = {}
        
        test_message = """
**ALERT TEST**

This is a test message from QuantumTrade Bot.
If you receive this, your alerts are working correctly! ✅

**Test Details:**
• Time: {time}
• Channels: All configured channels
• Status: Testing
        """.format(time=datetime.now().isoformat())
        
        # Test email
        if self.email_enabled and self._is_email_configured():
            test_results['email'] = await self._send_email("🧪 QuantumTrade Alert Test", test_message)
        else:
            test_results['email'] = False
        
        # Test Telegram
        if self.telegram_enabled and self._is_telegram_configured():
            test_results['telegram'] = await self._send_telegram(test_message)
        else:
            test_results['telegram'] = False
        
        return test_results
    
    def get_alert_status(self) -> Dict:
        """Get current alert system status"""
        return {
            "email": {
                "enabled": self.email_enabled,
                "configured": self._is_email_configured(),
                "server": self.email_config['smtp_server']
            },
            "telegram": {
                "enabled": self.telegram_enabled,
                "configured": self._is_telegram_configured()
            }
        }
