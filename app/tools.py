"""app/tools.py — Safe, narrowly-scoped Python tools JARVIS can call.

The LLM *requests* a tool; this module *executes* it.
All tools return plain strings so they're easy to pass back to the model.
"""
import datetime
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# ── Tool registry ────────────────────────────────────────────────────────────

TOOL_SPECS = [
    {
        "name": "get_current_time",
        "description": "Return the current date and time for a given timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'America/New_York'. Defaults to UTC.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Return a mock weather report for a city. "
            "(Replace with a real weather API call in production.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'New York'.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Schedule a reminder message after N minutes (stored in-process).",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reminder text."},
                "minutes": {"type": "integer", "description": "Minutes from now."},
            },
            "required": ["message", "minutes"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a simple arithmetic expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A safe arithmetic expression, e.g. '(3 + 5) * 2'.",
                }
            },
            "required": ["expression"],
        },
    },
]


# ── Implementations ──────────────────────────────────────────────────────────

def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, Exception):
        tz = ZoneInfo("UTC")
        timezone = "UTC"
    now = datetime.datetime.now(tz)
    return f"Current time in {timezone}: {now.strftime('%A, %B %d %Y %I:%M %p %Z')}"


def get_weather(city: str) -> str:
    # Stub — swap in a real API (OpenWeatherMap, wttr.in, etc.)
    mock_data = {
        "New York": "72°F, partly cloudy",
        "London": "58°F, overcast",
        "Tokyo": "81°F, sunny",
        "Mumbai": "88°F, humid",
    }
    report = mock_data.get(city, f"22°C, clear skies")
    return f"Weather in {city}: {report}"


_reminders: list[dict] = []

def set_reminder(message: str, minutes: int) -> str:
    due = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
    _reminders.append({"message": message, "due": due.isoformat()})
    return f"Reminder set: '{message}' in {minutes} minute(s) at {due.strftime('%I:%M %p')}."


def calculate(expression: str) -> str:
    # Safe eval — only allow digits, operators and whitespace
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: unsafe expression."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# ── Dispatcher ───────────────────────────────────────────────────────────────

_TOOL_MAP = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "set_reminder": set_reminder,
    "calculate": calculate,
}


def dispatch(tool_name: str, args: dict) -> str:
    """Call the named tool with the given args dict. Returns a string result."""
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
