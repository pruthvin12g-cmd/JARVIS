import unittest
from unittest import mock

import numpy as np

from llm import groq_bridge
from transcription import whisper_transcriber as wt


class ManualRecordingTranscriptionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.audio = np.array([100, -200, 300], dtype=np.int16)

    def test_selected_provider_is_independent_of_groq_key(self):
        with mock.patch.object(groq_bridge, "_config", {"stt": "groq"}), \
             mock.patch.object(groq_bridge, "_api_key", ""):
            self.assertEqual(groq_bridge.get_stt(), "groq")
            self.assertFalse(groq_bridge.has_groq_key())

        with mock.patch.object(groq_bridge, "_config", {"stt": "local"}), \
             mock.patch.object(groq_bridge, "_api_key", ""):
            self.assertEqual(groq_bridge.get_stt(), "local")
            self.assertFalse(groq_bridge.has_groq_key())

    async def test_groq_mode_returns_groq_transcript(self):
        with mock.patch.object(groq_bridge, "get_stt", return_value="groq"), \
             mock.patch.object(groq_bridge, "has_groq_key", return_value=True), \
             mock.patch.object(
                 groq_bridge, "transcribe_audio", new_callable=mock.AsyncMock,
                 return_value="recognized speech",
             ) as transcribe_audio:
            result = await wt.transcribe_recording(self.audio)

        self.assertEqual(result, "recognized speech")
        self.assertEqual(groq_bridge.STT_MODEL, "whisper-large-v3-turbo")
        transcribe_audio.assert_awaited_once_with(
            wt._to_wav_bytes(self.audio), prompt="", filename="speech.wav"
        )

    async def test_groq_failure_never_falls_back_to_local(self):
        failure_results = (None, "", "   ", RuntimeError("provider error"))
        for failure in failure_results:
            result = (
                {"side_effect": failure}
                if isinstance(failure, Exception)
                else {"return_value": failure}
            )
            with self.subTest(failure=repr(failure)), \
                 mock.patch.object(groq_bridge, "get_stt", return_value="groq"), \
                 mock.patch.object(groq_bridge, "has_groq_key", return_value=True), \
                 mock.patch.object(
                     groq_bridge, "transcribe_audio",
                     new_callable=mock.AsyncMock,
                     **result,
                 ), \
                 mock.patch.object(wt, "_do_transcribe") as local_transcribe:
                with self.assertRaises(wt.GroqTranscriptionError):
                    await wt.transcribe_recording(self.audio)

                local_transcribe.assert_not_called()

    async def test_groq_mode_reports_missing_api_key_without_local_fallback(self):
        with mock.patch.object(groq_bridge, "get_stt", return_value="groq"), \
             mock.patch.object(groq_bridge, "has_groq_key", return_value=False), \
             mock.patch.object(
                 groq_bridge, "transcribe_audio", new_callable=mock.AsyncMock,
             ) as transcribe_audio, \
             mock.patch.object(wt, "_do_transcribe") as local_transcribe:
            with self.assertRaisesRegex(
                wt.GroqTranscriptionError, "Groq API key is unavailable"
            ):
                await wt.transcribe_recording(self.audio)

        transcribe_audio.assert_not_awaited()
        local_transcribe.assert_not_called()

    async def test_local_transcription_requires_explicit_local_mode(self):
        with mock.patch.object(groq_bridge, "get_stt", return_value="local") as get_stt, \
             mock.patch.object(groq_bridge, "has_groq_key") as has_groq_key, \
             mock.patch.object(
                 groq_bridge, "transcribe_audio", new_callable=mock.AsyncMock,
             ) as transcribe_audio, \
             mock.patch.object(wt, "_do_transcribe", return_value="local speech") as local:
            result = await wt.transcribe_recording(self.audio)

        self.assertEqual(result, "local speech")
        get_stt.assert_called_once()
        has_groq_key.assert_not_called()
        transcribe_audio.assert_not_awaited()
        local.assert_called_once()


if __name__ == "__main__":
    unittest.main()
