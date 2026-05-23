from __future__ import annotations

import unittest

from tts_server.middleware.api_key_auth import extract_api_key


class APIKeyExtractTest(unittest.TestCase):
    def test_bearer_header(self) -> None:
        scope = {
            "headers": [(b"authorization", b"Bearer sk-test-123")],
            "query_string": b"",
        }
        self.assertEqual(extract_api_key(scope), "sk-test-123")

    def test_x_api_key_header(self) -> None:
        scope = {
            "headers": [(b"x-api-key", b"my-key")],
            "query_string": b"",
        }
        self.assertEqual(extract_api_key(scope), "my-key")

    def test_query_param(self) -> None:
        scope = {
            "headers": [],
            "query_string": b"api_key=ws-secret",
        }
        self.assertEqual(extract_api_key(scope), "ws-secret")

    def test_bearer_takes_precedence(self) -> None:
        scope = {
            "headers": [(b"authorization", b"Bearer from-header")],
            "query_string": b"api_key=from-query",
        }
        self.assertEqual(extract_api_key(scope), "from-header")


if __name__ == "__main__":
    unittest.main()
