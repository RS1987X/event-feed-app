# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/delivery.py
"""
Alert delivery system supporting multiple channels (email, webhook, Telegram, GUI).
"""
from __future__ import annotations
import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from pathlib import Path

from .store import AlertStore

logger = logging.getLogger(__name__)


class AlertDelivery:
    """
    Delivers alerts to users via multiple channels.
    
    Supported channels:
    - email: Send email notifications via SMTP
    - webhook: POST alerts to HTTP endpoints
    - telegram: Send messages via Telegram Bot API (FREE!)
    - gui: Queue alerts for GUI display (if applicable)
    
    Telegram is FREE and easy to set up:
    1. Create a bot with @BotFather on Telegram
    2. Get your bot token
    3. Users send /start to your bot to get their chat_id
    4. Configure user preferences with chat_id
    """
    
    def __init__(
        self,
        smtp_config: Optional[Dict[str, str]] = None,
        telegram_config: Optional[Dict[str, str]] = None,
        store: Optional[AlertStore] = None,
        templates_dir: Optional[Path] = None
    ):
        """
        Initialize alert delivery system.
        
        Args:
            smtp_config: SMTP configuration dict with keys:
                         host, port, username, password, from_address
            telegram_config: Telegram configuration dict with keys:
                            bot_token (required)
            store: AlertStore instance for recording deliveries
            templates_dir: Directory containing email templates
        """
        self.smtp_config = smtp_config or {}
        self.telegram_config = telegram_config or {}
        self.store = store or AlertStore()
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
    
    def deliver(self, alert: Dict[str, Any], users: List[Dict[str, Any]]):
        """
        Deliver alert to all matching users via their preferred channels.
        
        Args:
            alert: Alert object to deliver
            users: List of user preference dicts (from AlertStore.get_all_active_users())
        """
        if not users:
            logger.warning(f"No users to deliver alert {alert['alert_id']} to")
            return
        
        logger.info(f"Delivering alert {alert['alert_id']} to {len(users)} users")
        
        for user in users:
            # Check if user wants this alert type
            if alert["alert_type"] not in user.get("alert_types", []):
                logger.debug(f"User {user['user_id']} not subscribed to {alert['alert_type']}")
                continue
            
            # Check significance threshold
            if alert["significance_score"] < user.get("min_significance", 0.7):
                logger.debug(
                    f"Alert score {alert['significance_score']} below user threshold "
                    f"{user.get('min_significance')}"
                )
                continue
            
            # Check watchlist (if specified)
            watchlist = user.get("watchlist", [])
            if watchlist and alert.get("company_name") not in watchlist:
                logger.debug(f"Company {alert.get('company_name')} not in user watchlist")
                continue
            
            # Deliver via preferred channels
            channels = user.get("delivery_channels", ["email"])
            
            for channel in channels:
                try:
                    if channel == "email":
                        self._deliver_email(alert, user)
                    elif channel == "telegram":
                        self._deliver_telegram(alert, user)
                    elif channel == "webhook":
                        self._deliver_webhook(alert, user)
                    elif channel == "gui":
                        self._deliver_gui(alert, user)
                    else:
                        logger.warning(f"Unknown delivery channel: {channel}")
                except Exception as e:
                    logger.error(
                        f"Failed to deliver alert {alert['alert_id']} via {channel} "
                        f"to {user['user_id']}: {e}",
                        exc_info=True
                    )
                    self.store.record_delivery(
                        alert["alert_id"],
                        user["user_id"],
                        channel,
                        "failed",
                        str(e)
                    )
    
    def _deliver_email(self, alert: Dict[str, Any], user: Dict[str, Any]):
        """Send email notification."""
        if not self.smtp_config:
            logger.warning("SMTP not configured, skipping email delivery")
            return
        
        email_address = user.get("email_address")
        if not email_address:
            logger.warning(f"No email address for user {user['user_id']}")
            return
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._format_email_subject(alert)
            msg["From"] = self.smtp_config.get("from_address", "alerts@event-feed-app.com")
            msg["To"] = email_address
            
            # Plain text version
            text_body = self._render_text_email(alert)
            msg.attach(MIMEText(text_body, "plain"))
            
            # HTML version (if template exists)
            html_body = self._render_html_email(alert)
            if html_body:
                msg.attach(MIMEText(html_body, "html"))
            
            # Send email
            with smtplib.SMTP(
                self.smtp_config.get("host", "localhost"),
                int(self.smtp_config.get("port", 587))
            ) as server:
                server.starttls()
                if self.smtp_config.get("username"):
                    server.login(
                        self.smtp_config["username"],
                        self.smtp_config.get("password", "")
                    )
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {email_address} for {alert['alert_id']}")
            self.store.record_delivery(
                alert["alert_id"],
                user["user_id"],
                "email",
                "success"
            )
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}", exc_info=True)
            self.store.record_delivery(
                alert["alert_id"],
                user["user_id"],
                "email",
                "failed",
                str(e)
            )
            raise
    
    def _deliver_telegram(self, alert: Dict[str, Any], user: Dict[str, Any]):
        """
        Send Telegram notification.
        
        Telegram is FREE and easy to use:
        1. Create a bot: Talk to @BotFather on Telegram, use /newbot
        2. Get bot token: BotFather gives you a token like "123456:ABC-DEF..."
        3. Get chat_id: User sends /start to your bot, you get chat_id from webhook
        4. Send messages: POST to https://api.telegram.org/bot<token>/sendMessage
        
        No cost, instant delivery, supports markdown formatting!
        """
        if not self.telegram_config.get("bot_token"):
            logger.warning("Telegram bot_token not configured, skipping telegram delivery")
            return
        
        chat_id = user.get("telegram_chat_id")
        if not chat_id:
            logger.warning(f"No telegram_chat_id for user {user['user_id']}")
            return
        
        try:
            # Format message with Telegram markdown
            text = self._render_telegram_message(alert)
            
            # Send via Telegram Bot API
            bot_token = self.telegram_config["bot_token"]
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",  # Enable markdown formatting
                "disable_web_page_preview": False,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Telegram alert sent to chat_id {chat_id} for {alert['alert_id']}")
            self.store.record_delivery(
                alert["alert_id"],
                user["user_id"],
                "telegram",
                "success"
            )
        
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}", exc_info=True)
            self.store.record_delivery(
                alert["alert_id"],
                user["user_id"],
                "telegram",
                "failed",
                str(e)
            )
            raise
    
    def _deliver_webhook(self, alert: Dict[str, Any], user: Dict[str, Any]):
        """Send webhook notification."""
        webhook_url = user.get("webhook_url")
        if not webhook_url:
            logger.warning(f"No webhook URL for user {user['user_id']}")
            return
        
        try:
            response = requests.post(
                webhook_url,
                json=alert,
                timeout=10,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "EventFeedApp-AlertSystem/1.0"
                }
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent to {webhook_url} for {alert['alert_id']}")
            self.store.record_delivery(
                alert["alert_id"],
                user["user_id"],
                "webhook",
                "success"
            )
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}", exc_info=True)
            self.store.record_delivery(
                alert["alert_id"],
                user["user_id"],
                "webhook",
                "failed",
                str(e)
            )
            raise
    
    def _deliver_gui(self, alert: Dict[str, Any], user: Dict[str, Any]):
        """Queue alert for GUI display (placeholder for now)."""
        # This would integrate with your PyQt6 GUI from main.py
        # For now, just record the delivery
        logger.info(f"GUI alert queued for {alert['alert_id']}")
        self.store.record_delivery(
            alert["alert_id"],
            user["user_id"],
            "gui",
            "success"
        )
    
    def _format_email_subject(self, alert: Dict[str, Any]) -> str:
        """Format email subject line."""
        icon = "🔔"
        alert_type = alert.get("alert_type", "Alert").replace("_", " ").title()
        company = alert.get("company_name", "Company")
        
        return f"{icon} {alert_type}: {company}"
    
    def _render_text_email(self, alert: Dict[str, Any]) -> str:
        """Render plain text email body."""
        lines = [
            f"{alert.get('company_name', 'A company')} has updated their guidance:",
            "",
        ]
        
        # Add metrics
        metrics = alert.get("metrics", [])
        if metrics:
            for metric in metrics:
                direction_icon = self._get_direction_icon(metric.get("direction"))
                metric_name = metric.get("metric", "Unknown").replace("_", " ").title()
                
                # Format value/range
                value_str = self._format_metric_value(metric)
                
                lines.append(f"  {direction_icon} {metric_name}: {value_str}")
        else:
            lines.append(f"  • {alert.get('summary', 'Guidance updated')}")
        
        # Add period
        period = alert.get("metadata", {}).get("period")
        if period and period not in ("UNKNOWN", "PERIOD-UNKNOWN"):
            lines.append(f"\nPeriod: {period}")
        
        # Add significance score
        score = int(alert.get("significance_score", 0) * 100)
        lines.append(f"\nSignificance Score: {score}/100")
        
        # Add link
        press_release_url = alert.get("metadata", {}).get("press_release_url")
        if press_release_url:
            lines.append(f"\nView full press release: {press_release_url}")
        
        # Footer
        lines.extend([
            "",
            "---",
            "This is an automated alert from EventFeedApp.",
            "To manage your alert preferences, contact your administrator."
        ])
        
        return "\n".join(lines)
    
    def _render_telegram_message(self, alert: Dict[str, Any]) -> str:
        """
        Render Telegram message with Markdown formatting.
        
        Telegram supports: *bold*, _italic_, [links](url), `code`
        """
        def _escape_md(text: str) -> str:
            if not isinstance(text, str):
                return ""
            # Minimal escaping for Telegram Markdown (not V2)
            return (
                text.replace("_", "\\_")
                    .replace("*", "\\*")
                    .replace("`", "\\`")
                    .replace("[", "\\[")
            )
        # Header with company name
        lines = [
            f"🔔 *{alert.get('company_name', 'Company')} - Guidance Alert*",
            "",
        ]
        
        # Show guidance count if multiple
        guidance_count = alert.get("guidance_count", 1)
        if guidance_count > 1:
            lines.append(f"_{guidance_count} guidance updates detected_")
            lines.append("")
        
        # Add metrics with emoji indicators
        metrics = alert.get("metrics", [])
        if metrics:
            # Limit to first 5 metrics to avoid message size limits
            display_metrics = metrics[:5]
            for metric in display_metrics:
                direction = metric.get("direction")
                if direction == "up":
                    emoji = "📈"
                elif direction == "down":
                    emoji = "📉"
                else:
                    emoji = "➡️"
                
                metric_name = metric.get("metric", "Unknown").replace("_", " ").title()
                value_str = self._format_metric_value(metric)
                
                lines.append(f"{emoji} *{metric_name}*: {value_str}")
            
            if len(metrics) > 5:
                lines.append(f"_...and {len(metrics) - 5} more metrics_")
        else:
            lines.append(f"• _{alert.get('summary', 'Guidance updated')}_")
        
        # Optional title
        title = alert.get("metadata", {}).get("title")
        if title:
            lines.append(f"_“{_escape_md(title)}”_")
            lines.append("")

        # Add period
        period = alert.get("metadata", {}).get("period")
        if period and period not in ("UNKNOWN", "PERIOD-UNKNOWN"):
            lines.append(f"\n📅 Period: `{period}`")
        
        # Add significance score
        score = int(alert.get("significance_score", 0) * 100)
        lines.append(f"⭐ Significance: *{score}/100*")
        
        # Add press release link if available, else include a short snippet
        press_release_url = alert.get("metadata", {}).get("press_release_url")
        snippet = alert.get("metadata", {}).get("body_snippet")
        include_snippet = self.telegram_config.get("include_snippet_when_no_url", True)
        if press_release_url:
            safe_url = _escape_md(press_release_url)
            lines.append(f"\n🔗 View Press Release:\n{safe_url}")
        elif include_snippet and snippet:
            safe_snippet = _escape_md(snippet)
            lines.append(f"\n� Snippet:\n{safe_snippet}")

        # Always include the event id so the user can look it up with local tools
        event_id = alert.get("event_id")
        if event_id:
            lines.append(f"\n🆔 ID: `{event_id}`")
        
        return "\n".join(lines)
    
    def _render_html_email(self, alert: Dict[str, Any]) -> Optional[str]:
        """Render HTML email body (optional)."""
        # Load template if exists
        template_path = self.templates_dir / "email_guidance.html"
        if not template_path.exists():
            return None
        
        try:
            with open(template_path) as f:
                template = f.read()
            
            # Simple string substitution (use Jinja2 for production)
            html = template.replace("{{company_name}}", alert.get("company_name", "Company"))
            html = html.replace("{{summary}}", alert.get("summary", "Guidance updated"))
            html = html.replace("{{score}}", str(int(alert.get("significance_score", 0) * 100)))
            
            return html
        
        except Exception as e:
            logger.warning(f"Failed to render HTML email: {e}")
            return None
    
    def _get_direction_icon(self, direction: Optional[str]) -> str:
        """Get icon for direction."""
        if direction == "up":
            return "↑"
        elif direction == "down":
            return "↓"
        else:
            return "→"
    
    def _format_metric_value(self, metric: Dict[str, Any]) -> str:
        """Format metric value for display."""
        parts = []
        
        if metric.get("value"):
            value = metric["value"]
            unit = metric.get("unit", "")
            parts.append(f"{value}{unit}")
        
        if metric.get("range"):
            parts.append(metric["range"])
        
        if metric.get("basis"):
            basis = metric["basis"]
            if basis == "organic":
                parts.append("(organic)")
            elif basis == "cc_fx":
                parts.append("(constant currency)")
        
        return " ".join(parts) if parts else "Updated"
