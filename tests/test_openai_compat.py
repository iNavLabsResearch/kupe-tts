from __future__ import annotations

import unittest

from tts_server.domain.openai_compat import (
    extract_voice_name,
    resolve_openai_voice,
    validate_speech_model,
)
from tts_server.schemas import OpenAIVoiceRef, OpenAISpeechRequest


class OpenAICompatTest(unittest.TestCase):
    def test_extract_voice_string(self) -> None:
        self.assertEqual(extract_voice_name("ajay"), "ajay")

    def test_extract_voice_object(self) -> None:
        self.assertEqual(extract_voice_name(OpenAIVoiceRef(id="soham")), "soham")
        self.assertEqual(extract_voice_name({"id": "monika"}), "monika")

    def test_resolve_profile_name(self) -> None:
        profiles = {"ajay": object(), "soham": object()}
        self.assertEqual(resolve_openai_voice("soham", profiles, "ajay"), "soham")
        self.assertEqual(resolve_openai_voice("SOHAM", profiles, "ajay"), "soham")

    def test_resolve_openai_preset(self) -> None:
        profiles = {"ajay": object(), "soham": object(), "monika": object()}
        self.assertEqual(resolve_openai_voice("alloy", profiles, "ajay"), "ajay")
        self.assertEqual(resolve_openai_voice("nova", profiles, "ajay"), "monika")

    def test_unknown_voice_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_openai_voice("unknown_voice", {"ajay": object()}, "ajay")

    def test_validate_model(self) -> None:
        validate_speech_model("tts-1")
        validate_speech_model("gpt-4o-mini-tts")
        with self.assertRaises(ValueError):
            validate_speech_model("gpt-5-tts")

    def test_speech_request_voice_dict(self) -> None:
        req = OpenAISpeechRequest.model_validate(
            {
                "input": "hello",
                "model": "tts-1",
                "voice": {"id": "ajay"},
            }
        )
        self.assertEqual(extract_voice_name(req.voice), "ajay")


if __name__ == "__main__":
    unittest.main()
