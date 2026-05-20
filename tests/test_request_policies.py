from __future__ import annotations

import unittest

from tts_server.domain.request_policies import coerce_speed, coerce_text, resolve_fc_rest_epochs


class RequestPoliciesTest(unittest.TestCase):
    def test_coerce_text_list(self) -> None:
        self.assertEqual(coerce_text([" hello ", 123, None, "world"]), "hello 123 world")

    def test_speed_range(self) -> None:
        value, err = coerce_speed("1.5")
        self.assertEqual(value, 1.5)
        self.assertIsNone(err)
        value, err = coerce_speed("999")
        self.assertIsNone(value)
        self.assertIn("out of range", err or "")

    def test_fc_rest_epoch_resolution(self) -> None:
        fc, rest, err = resolve_fc_rest_epochs({"epochs": 10, "epochs_fc": 4})
        self.assertIsNone(err)
        self.assertEqual(fc, 4)
        self.assertEqual(rest, 10)


if __name__ == "__main__":
    unittest.main()

