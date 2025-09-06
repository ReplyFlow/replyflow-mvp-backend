"""
Simple analytics store for ReplyFlow MVP.

This module provides basic functionality to record usage metrics and error
logs for the ReplyFlow application.  Usage events track how often API
endpoints or features are accessed, while error logs capture exceptions
raised during request handling.  Data is stored in a JSON file on disk
to persist across server restarts.

Functions:
    record_event(event_name: str, count: int = 1) -> None
        Increment a usage counter for a given event.

    record_error(context: str, message: str) -> None
        Append an error entry with timestamp, context and message.

    get_stats() -> dict
        Load and return the current analytics data structure.

Environment variables:
    REPLYFLOW_ANALYTICS_FILE: optional path to the analytics JSON file.
        Defaults to 'analytics.json' in the current working directory.
"""

import os
import json
import threading
from datetime import datetime, timezone

_lock = threading.Lock()

ANALYTICS_FILE = os.getenv("REPLYFLOW_ANALYTICS_FILE", "analytics.json")


def _load_data() -> dict:
    """Load analytics data from disk.

    Returns a dictionary with "events" and "errors" keys.  If the file
    doesn't exist or cannot be parsed, a new structure is returned.
    """
    if not os.path.exists(ANALYTICS_FILE):
        return {"events": {}, "errors": []}
    try:
        with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure required keys exist
        if not isinstance(data, dict):
            return {"events": {}, "errors": []}
        data.setdefault("events", {})
        data.setdefault("errors", [])
        return data
    except Exception:
        return {"events": {}, "errors": []}


def _save_data(data: dict) -> None:
    """Persist analytics data to disk."""
    try:
        with open(ANALYTICS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        # If saving fails, silently ignore; analytics will be lost
        pass


def record_event(event_name: str, count: int = 1) -> None:
    """Increment a usage counter for the specified event.

    Args:
        event_name: A string identifying the event or feature.
        count: How much to increment the counter by (default 1).
    """
    if not event_name:
        return
    with _lock:
        data = _load_data()
        events = data.setdefault("events", {})
        events[event_name] = events.get(event_name, 0) + count
        _save_data(data)


def record_error(context: str, message: str) -> None:
    """Record an error entry with context and message.

    Each error entry includes a timestamp in UTC ISO format.

    Args:
        context: A short string indicating where the error occurred (e.g. event name or endpoint).
        message: The error message or description.
    """
    if not message:
        return
    with _lock:
        data = _load_data()
        errors = data.setdefault("errors", [])
        errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "message": message,
        })
        _save_data(data)


def get_stats() -> dict:
    """Return a copy of the current analytics data.

    Returns:
        A dictionary with "events" and "errors" lists.
    """
    with _lock:
        return _load_data().copy()