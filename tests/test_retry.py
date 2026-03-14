"""Tests for retry logic with exponential backoff."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from engram.retry import _is_transient, with_retry


class TestIsTransient:
    def test_transient_status_codes(self):
        for code in (502, 503, 504, 429):
            exc = Exception()
            exc.status_code = code
            assert _is_transient(exc) is True

    def test_non_transient_status_code(self):
        exc = Exception()
        exc.status_code = 404
        assert _is_transient(exc) is False

    def test_connection_error_by_name(self):
        class ConnectionError(Exception):
            pass

        assert _is_transient(ConnectionError()) is True

    def test_timeout_error_by_name(self):
        class TimeoutError(Exception):
            pass

        assert _is_transient(TimeoutError()) is True

    def test_regular_exception_not_transient(self):
        assert _is_transient(ValueError("bad value")) is False


class TestWithRetry:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value=42)
        wrapped = with_retry(max_retries=3, backoff_base=0.01)(fn)
        assert wrapped() == 42
        assert fn.call_count == 1

    def test_retries_on_transient_then_succeeds(self):
        exc = Exception()
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        wrapped = with_retry(max_retries=3, backoff_base=0.01)(fn)
        assert wrapped() == "ok"
        assert fn.call_count == 3

    def test_raises_after_max_retries(self):
        exc = Exception("server error")
        exc.status_code = 503
        fn = MagicMock(side_effect=exc)
        wrapped = with_retry(max_retries=2, backoff_base=0.01)(fn)
        with pytest.raises(Exception, match="server error"):
            wrapped()
        assert fn.call_count == 3  # initial + 2 retries

    def test_non_transient_raises_immediately(self):
        fn = MagicMock(side_effect=ValueError("bad"))
        wrapped = with_retry(max_retries=3, backoff_base=0.01)(fn)
        with pytest.raises(ValueError, match="bad"):
            wrapped()
        assert fn.call_count == 1

    def test_backoff_increases(self):
        exc = Exception()
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        wrapped = with_retry(max_retries=3, backoff_base=0.05)(fn)
        t0 = time.monotonic()
        wrapped()
        elapsed = time.monotonic() - t0
        # Should sleep 0.05 + 0.1 = 0.15s minimum
        assert elapsed >= 0.1
