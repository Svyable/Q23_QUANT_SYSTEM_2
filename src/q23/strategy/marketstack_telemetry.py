"""q23.strategy.marketstack_telemetry

Telemetry and status tracking for Marketstack API integration.

Provides structured logging and status reporting for Marketstack API calls,
following SOLID principles with clear separation of concerns.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from q23.shared.config import cfg


class MarketstackStatus(Enum):
    """Marketstack API status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class FetchResult(Enum):
    """Result of a Marketstack fetch operation."""
    SUCCESS = "success"
    FAILED = "failed"
    EMPTY = "empty"
    SKIPPED = "skipped"


@dataclass
class MarketstackFetchRecord:
    """Record of a single Marketstack API fetch operation."""
    timestamp: float
    strategy_id: Optional[str]
    symbols_count: int
    date_from: Optional[str]
    date_to: Optional[str]
    fetched_date: Optional[str]
    result: FetchResult
    rows_fetched: int
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['result'] = self.result.value
        data['timestamp_iso'] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        return data


@dataclass
class MarketstackTelemetry:
    """Telemetry state for Marketstack API."""
    status: MarketstackStatus
    api_key_set: bool
    enabled: bool
    last_fetch: Optional[MarketstackFetchRecord] = None
    fetch_history: List[MarketstackFetchRecord] = field(default_factory=list)
    total_fetches: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    total_rows_fetched: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    
    def record_fetch(self, record: MarketstackFetchRecord) -> None:
        """Record a fetch operation."""
        self.last_fetch = record
        self.fetch_history.append(record)
        self.total_fetches += 1
        
        if record.result == FetchResult.SUCCESS:
            self.successful_fetches += 1
            self.total_rows_fetched += record.rows_fetched
        elif record.result == FetchResult.FAILED:
            self.failed_fetches += 1
        
        # Keep only last 100 records
        if len(self.fetch_history) > 100:
            self.fetch_history = self.fetch_history[-100:]
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_fetches == 0:
            return 0.0
        return (self.successful_fetches / self.total_fetches) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "api_key_set": self.api_key_set,
            "enabled": self.enabled,
            "last_fetch": self.last_fetch.to_dict() if self.last_fetch else None,
            "total_fetches": self.total_fetches,
            "successful_fetches": self.successful_fetches,
            "failed_fetches": self.failed_fetches,
            "success_rate": self.get_success_rate(),
            "total_rows_fetched": self.total_rows_fetched,
            "recent_fetches": [r.to_dict() for r in self.fetch_history[-10:]],
            "config": self.config,
        }


class MarketstackTelemetryManager:
    """Manages Marketstack telemetry state (singleton pattern)."""
    
    _instance: Optional[MarketstackTelemetryManager] = None
    _telemetry: MarketstackTelemetry
    
    def __init__(self):
        if MarketstackTelemetryManager._instance is not None:
            raise RuntimeError("MarketstackTelemetryManager is a singleton")
        
        self._telemetry = self._initialize_telemetry()
        MarketstackTelemetryManager._instance = self
    
    @classmethod
    def get_instance(cls) -> MarketstackTelemetryManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._telemetry = cls._instance._initialize_telemetry()
        return cls._instance
    
    def _initialize_telemetry(self) -> MarketstackTelemetry:
        """Initialize telemetry from config."""
        api_key_set = bool(cfg.marketstack.API_KEY and cfg.marketstack.API_KEY.strip())
        enabled = cfg.marketstack.ENABLED and api_key_set
        
        status = MarketstackStatus.ENABLED if enabled else MarketstackStatus.DISABLED
        
        config = {
            "base_url": cfg.marketstack.BASE_URL,
            "rate_limit_delay": cfg.marketstack.RATE_LIMIT_DELAY,
            "default_lookback_days": cfg.marketstack.DEFAULT_LOOKBACK_DAYS,
            "force_live_data": getattr(cfg.marketstack, 'FORCE_LIVE_DATA', False),
            "pre_market_cutoff_hour": getattr(cfg.marketstack, 'PRE_MARKET_CUTOFF_HOUR', 9),
            "post_market_cutoff_hour": getattr(cfg.marketstack, 'POST_MARKET_CUTOFF_HOUR', 16),
        }
        
        return MarketstackTelemetry(
            status=status,
            api_key_set=api_key_set,
            enabled=enabled,
            config=config,
        )
    
    def get_telemetry(self) -> MarketstackTelemetry:
        """Get current telemetry state."""
        return self._telemetry
    
    def record_fetch(
        self,
        strategy_id: Optional[str],
        symbols_count: int,
        date_from: Optional[str],
        date_to: Optional[str],
        fetched_date: Optional[str],
        result: FetchResult,
        rows_fetched: int,
        error_message: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a fetch operation."""
        record = MarketstackFetchRecord(
            timestamp=time.time(),
            strategy_id=strategy_id,
            symbols_count=symbols_count,
            date_from=date_from,
            date_to=date_to,
            fetched_date=fetched_date,
            result=result,
            rows_fetched=rows_fetched,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        self._telemetry.record_fetch(record)
    
    def update_status(self, status: MarketstackStatus) -> None:
        """Update API status."""
        self._telemetry.status = status
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        tel = self._telemetry
        if not tel.enabled:
            return "❌ DISABLED"
        if not tel.api_key_set:
            return "⚠️  NO API KEY"
        if tel.status == MarketstackStatus.ERROR:
            return "❌ ERROR"
        if tel.status == MarketstackStatus.RATE_LIMITED:
            return "⏳ RATE LIMITED"
        
        if tel.total_fetches == 0:
            return "✅ READY (no fetches yet)"
        
        success_rate = tel.get_success_rate()
        return f"✅ ACTIVE ({tel.total_fetches} fetches, {success_rate:.1f}% success)"
    
    def get_recent_activity_summary(self, max_records: int = 5) -> List[str]:
        """Get recent activity summary lines."""
        tel = self._telemetry
        if not tel.fetch_history:
            return ["No fetch history"]
        
        lines = []
        for record in tel.fetch_history[-max_records:]:
            result_icon = "✅" if record.result == FetchResult.SUCCESS else "❌"
            date_str = record.fetched_date or f"{record.date_from}→{record.date_to}" or "N/A"
            time_str = datetime.fromtimestamp(record.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
            lines.append(
                f"{result_icon} {time_str} | {date_str} | {record.symbols_count} symbols | "
                f"{record.rows_fetched} rows | {record.duration_ms:.0f}ms"
            )
        return lines


# Convenience functions
def get_telemetry_manager() -> MarketstackTelemetryManager:
    """Get telemetry manager instance."""
    return MarketstackTelemetryManager.get_instance()


def record_marketstack_fetch(
    strategy_id: Optional[str],
    symbols_count: int,
    date_from: Optional[str],
    date_to: Optional[str],
    fetched_date: Optional[str],
    result: FetchResult,
    rows_fetched: int,
    error_message: Optional[str] = None,
    duration_ms: float = 0.0,
) -> None:
    """Convenience function to record a fetch."""
    get_telemetry_manager().record_fetch(
        strategy_id=strategy_id,
        symbols_count=symbols_count,
        date_from=date_from,
        date_to=date_to,
        fetched_date=fetched_date,
        result=result,
        rows_fetched=rows_fetched,
        error_message=error_message,
        duration_ms=duration_ms,
    )
