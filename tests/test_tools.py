"""tests/test_tools.py — Unit tests for JARVIS tools (no API key required)."""
import pytest
from app.tools import dispatch, get_current_time, get_weather, set_reminder, calculate


def test_get_current_time_utc():
    result = get_current_time("UTC")
    assert "UTC" in result
    assert "Current time" in result


def test_get_current_time_bad_tz():
    result = get_current_time("NotARealZone")
    assert "UTC" in result  # fallback


def test_get_weather_known():
    result = get_weather("London")
    assert "London" in result
    assert "°" in result


def test_get_weather_unknown():
    result = get_weather("Atlantis")
    assert "Atlantis" in result


def test_set_reminder():
    result = set_reminder("Take a break", 10)
    assert "Take a break" in result
    assert "10 minute" in result


def test_calculate_valid():
    result = calculate("2 + 2")
    assert "4" in result


def test_calculate_complex():
    result = calculate("(10 + 5) * 3")
    assert "45" in result


def test_calculate_unsafe():
    result = calculate("__import__('os').system('ls')")
    assert "unsafe" in result


def test_dispatch_known():
    result = dispatch("get_current_time", {"timezone": "UTC"})
    assert "Current time" in result


def test_dispatch_unknown():
    result = dispatch("does_not_exist", {})
    assert "Unknown tool" in result
