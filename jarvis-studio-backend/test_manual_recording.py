import asyncio
import threading
import unittest
from unittest import mock

import numpy as np

import main
from transcription import whisper_transcriber as wt


class ManualRecordingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main._recording_lock = asyncio.Lock()
        main._recording_state = "idle"
        main._recording_server_id = "test-server"
        main._recording_generation = 0
        main._recording_revision = 0
        main._recording_stop = None
        main._recording_cancel = None
        main._recording_task = None
        main._recording_owner_client_id = ""
        main._recording_terminal_request_id = None
        main._pipeline_lock = asyncio.Lock()
        main._user_busy = False
        self.started = threading.Event()
        self.clip = np.array([100, 200, 300], dtype=np.int16)
        self.capture = mock.patch.object(main, "record_manual_audio",
                                         side_effect=self._capture)
        self.transcribe = mock.patch.object(
            main, "transcribe_recording", new_callable=mock.AsyncMock,
            return_value="test command")
        self.pipeline = mock.patch.object(main, "_run_pipeline",
                                          new_callable=mock.AsyncMock)
        self.emit = mock.patch.object(main, "emit", new_callable=mock.AsyncMock)
        self.capture_mock = self.capture.start()
        self.transcribe_mock = self.transcribe.start()
        self.pipeline_mock = self.pipeline.start()
        self.emit_mock = self.emit.start()
        self.addCleanup(self.capture.stop)
        self.addCleanup(self.transcribe.stop)
        self.addCleanup(self.pipeline.stop)
        self.addCleanup(self.emit.stop)

    def _capture(self, stop_event, cancel_event):
        self.started.set()
        while not stop_event.wait(0.01):
            pass
        if cancel_event.is_set():
            return np.array([], dtype=np.int16)
        return self.clip.copy()

    async def _wait_for_capture(self):
        self.assertTrue(await asyncio.to_thread(self.started.wait, 2))

    async def test_microphone_capture_starts_only_after_explicit_start(self):
        self.capture_mock.assert_not_called()
        await main.handle_start_recording()
        await self._wait_for_capture()
        self.assertEqual(main._recording_state, "recording")
        self.capture_mock.assert_called_once()
        await main.handle_cancel_recording()
        await main._recording_task

    async def test_stop_submits_once_and_runs_pipeline(self):
        await main.handle_start_recording({"request_id": "start-1"})
        await self._wait_for_capture()
        self.transcribe_mock.assert_not_awaited()

        task = main._recording_task
        await main.handle_stop_recording({"request_id": "stop-1"})
        await main.handle_stop_recording({"request_id": "duplicate-stop"})
        await task

        self.transcribe_mock.assert_awaited_once()
        self.pipeline_mock.assert_awaited_once_with(
            "test command", emit_transcription=True)
        self.assertEqual(main._recording_state, "idle")
        events = [call.args[1] for call in self.emit_mock.await_args_list
                  if call.args[0] == "recording_state"]
        self.assertEqual([event["state"] for event in events],
                         ["recording", "transcribing", "idle"])
        self.assertEqual([event["generation"] for event in events], [1, 1, 1])
        self.assertEqual([event["revision"] for event in events], [1, 2, 3])
        self.assertEqual([event["server_id"] for event in events],
                         ["test-server"] * 3)
        self.assertEqual(events[0]["request_id"], "start-1")
        self.assertEqual(events[0]["action"], "start")
        self.assertEqual(events[1]["request_id"], "stop-1")
        self.assertEqual(events[1]["action"], "stop")

    async def test_cancel_discards_audio_without_transcription(self):
        await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task
        await main.handle_cancel_recording()
        await main.handle_stop_recording()
        await task

        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()
        self.assertEqual(main._recording_state, "idle")

    async def test_groq_transcription_error_is_shown_without_crashing_worker(self):
        message = "Groq transcription failed. Check your connection and try again."
        self.transcribe_mock.side_effect = main.GroqTranscriptionError(message)
        await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task
        await main.handle_stop_recording()
        await task

        self.assertIn(
            mock.call("response", message),
            self.emit_mock.await_args_list,
        )
        self.assertEqual(main._recording_state, "idle")
        self.pipeline_mock.assert_not_awaited()

    async def test_disconnect_cancels_active_capture(self):
        release_capture = threading.Event()

        def capture_until_released(stop_event, cancel_event):
            self.started.set()
            release_capture.wait(2)
            if cancel_event.is_set():
                return np.array([], dtype=np.int16)
            return self.clip.copy()

        self.capture_mock.side_effect = capture_until_released
        with mock.patch.object(main, "current_client_id", return_value="recording-owner"):
            await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task

        with mock.patch.object(main, "_remote_screen_lock", None), \
             mock.patch.object(main, "is_current_sender_remote", return_value=False), \
             mock.patch.object(main, "current_client_id", return_value="recording-owner"):
            await main.handle_remote_disconnect(None)
        release_capture.set()
        await task

        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()
        self.assertEqual(main._recording_state, "idle")

    async def test_disconnect_of_other_window_preserves_owned_recording(self):
        release_capture = threading.Event()

        def capture_until_released(stop_event, cancel_event):
            self.started.set()
            release_capture.wait(2)
            if cancel_event.is_set():
                return np.array([], dtype=np.int16)
            return self.clip.copy()

        self.capture_mock.side_effect = capture_until_released
        with mock.patch.object(main, "current_client_id", return_value="recording-owner"):
            await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task

        with mock.patch.object(main, "_remote_screen_lock", None), \
             mock.patch.object(main, "is_current_sender_remote", return_value=False), \
             mock.patch.object(main, "current_client_id", return_value="other-window"):
            await main.handle_remote_disconnect(None)

        self.assertEqual(main._recording_state, "recording")
        self.assertFalse(main._recording_cancel.is_set())
        await main.handle_cancel_recording()
        release_capture.set()
        await task

        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()
        self.assertEqual(main._recording_state, "idle")

    async def test_owner_disconnect_during_transcription_cancels_pipeline(self):
        transcribing = asyncio.Event()

        async def hold_transcription(_audio):
            transcribing.set()
            await asyncio.Event().wait()

        self.transcribe_mock.side_effect = hold_transcription
        with mock.patch.object(main, "current_client_id", return_value="recording-owner"):
            await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task
        await main.handle_stop_recording()
        await asyncio.wait_for(transcribing.wait(), timeout=2)

        with mock.patch.object(main, "_remote_screen_lock", None), \
             mock.patch.object(main, "is_current_sender_remote", return_value=False), \
             mock.patch.object(main, "current_client_id", return_value="recording-owner"):
            await main.handle_remote_disconnect(None)
        await asyncio.gather(task, return_exceptions=True)

        self.pipeline_mock.assert_not_awaited()
        self.assertEqual(main._recording_state, "idle")

    async def test_owner_disconnect_while_stopping_waits_for_delayed_read_cleanup(self):
        read_started = threading.Event()
        release_read = threading.Event()
        chunk = np.arange(wt.CHUNK_SIZE, dtype=np.int16).tobytes()
        stream = mock.Mock()

        def delayed_read(*_args, **_kwargs):
            read_started.set()
            if not release_read.wait(2):
                raise TimeoutError("test did not release the delayed read")
            return chunk

        stream.read.side_effect = delayed_read
        audio_device = mock.Mock()
        audio_device.open.return_value = stream
        self.capture.stop()

        with mock.patch.object(main, "record_manual_audio", wt.record_manual_audio), \
             mock.patch.object(wt.pyaudio, "PyAudio", return_value=audio_device), \
             mock.patch.object(main, "current_client_id", return_value="recording-owner"):
            await main.handle_start_recording({"request_id": "start-delayed"})
            self.assertTrue(await asyncio.to_thread(read_started.wait, 2))
            task = main._recording_task
            await main.handle_stop_recording({"request_id": "stop-delayed"})

            with mock.patch.object(main, "_remote_screen_lock", None), \
                 mock.patch.object(main, "is_current_sender_remote", return_value=False):
                await main.handle_remote_disconnect(None)

            stream.stop_stream.assert_not_called()
            stream.close.assert_not_called()
            audio_device.terminate.assert_not_called()
            release_read.set()
            await task

        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        audio_device.terminate.assert_called_once()
        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()
        self.assertEqual(main._recording_state, "idle")

    async def test_rejected_start_returns_request_correlated_error(self):
        await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task
        send_to_current = mock.AsyncMock()

        with mock.patch.object(main, "send_to_current", send_to_current), \
             mock.patch.object(main, "current_client_id", return_value="other-window"):
            await main.handle_start_recording({"request_id": "start-2"})

        send_to_current.assert_awaited_once_with("recording_rejected", {
            "request_id": "start-2",
            "action": "start",
            "message": "A recording session is already active.",
        })
        self.capture_mock.assert_called_once()
        await main.handle_cancel_recording()
        await task

    async def test_cancel_acceptance_and_idle_are_correlated_to_request(self):
        await main.handle_start_recording({"request_id": "start-cancel"})
        await self._wait_for_capture()
        task = main._recording_task
        self.assertTrue(await main.handle_cancel_recording({
            "request_id": "cancel-1",
        }))
        await task

        events = [call.args[1] for call in self.emit_mock.await_args_list
                  if call.args[0] == "recording_state"]
        self.assertEqual([event["state"] for event in events],
                         ["recording", "cancelling", "idle"])
        self.assertEqual(events[1]["request_id"], "cancel-1")
        self.assertEqual(events[1]["action"], "cancel")
        self.assertEqual(events[2]["request_id"], "cancel-1")
        self.assertEqual(events[2]["action"], "cancel")
        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()

    async def test_recording_snapshot_carries_current_version_without_advancing_it(self):
        await main.handle_start_recording({"request_id": "snapshot-start"})
        await self._wait_for_capture()
        before = main._recording_revision
        snapshot = main._recording_state_snapshot()

        self.assertEqual(snapshot["server_id"], "test-server")
        self.assertEqual(snapshot["generation"], 1)
        self.assertEqual(snapshot["revision"], before)
        self.assertEqual(snapshot["state"], "recording")
        self.assertIsNone(snapshot["request_id"])
        self.assertEqual(main._recording_revision, before)

        await main.handle_cancel_recording()
        await main._recording_task

    async def test_real_cancelled_capture_is_not_transcribed(self):
        started = threading.Event()
        release_read = threading.Event()
        chunk = np.arange(wt.CHUNK_SIZE, dtype=np.int16).tobytes()
        stream = mock.Mock()
        reads = 0

        def read(*_args, **_kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                return chunk
            started.set()
            if not release_read.wait(2):
                raise TimeoutError("test did not release the mocked read")
            return chunk

        stream.read.side_effect = read
        audio_device = mock.Mock()
        audio_device.open.return_value = stream
        captured = []
        self.capture.stop()

        def capture_and_keep_result(stop_event, cancel_event):
            result = wt.record_manual_audio(stop_event, cancel_event)
            captured.append(result)
            return result

        with mock.patch.object(main, "record_manual_audio", capture_and_keep_result), \
             mock.patch.object(wt.pyaudio, "PyAudio", return_value=audio_device):
            await main.handle_start_recording()
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task = main._recording_task
            await main.handle_cancel_recording()
            release_read.set()
            await task

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].size, 0)
        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        audio_device.terminate.assert_called_once()

    async def test_duplicate_start_and_overlapping_session_are_ignored(self):
        await main.handle_start_recording()
        await self._wait_for_capture()
        first_task = main._recording_task
        await main.handle_start_recording()
        self.assertIs(main._recording_task, first_task)
        self.capture_mock.assert_called_once()
        await main.handle_cancel_recording()
        await first_task

    async def test_capture_error_returns_to_idle_without_transcription(self):
        self.capture_mock.side_effect = RuntimeError("device unavailable")
        await main.handle_start_recording()
        task = main._recording_task
        await task

        self.assertEqual(main._recording_state, "idle")
        self.transcribe_mock.assert_not_awaited()
        self.pipeline_mock.assert_not_awaited()
        self.emit_mock.assert_any_await(
            "response", "I couldn't complete that recording. Please try again.")

    async def test_transcription_error_returns_to_idle_without_pipeline(self):
        self.transcribe_mock.side_effect = RuntimeError("transcription failed")
        await main.handle_start_recording()
        await self._wait_for_capture()
        task = main._recording_task
        await main.handle_stop_recording()
        await task

        self.assertEqual(main._recording_state, "idle")
        self.pipeline_mock.assert_not_awaited()
        self.emit_mock.assert_any_await(
            "response", "I couldn't complete that recording. Please try again.")


class ManualCaptureCleanupTests(unittest.TestCase):
    def test_cancel_discards_captured_frames_and_closes_audio_resources(self):
        cancel_event = threading.Event()
        chunk = np.arange(wt.CHUNK_SIZE, dtype=np.int16).tobytes()
        stream = mock.Mock()
        stream.read.side_effect = lambda *_args, **_kwargs: (
            cancel_event.set() or chunk
        )
        audio_device = mock.Mock()
        audio_device.open.return_value = stream

        with mock.patch.object(wt.pyaudio, "PyAudio", return_value=audio_device):
            audio = wt.record_manual_audio(threading.Event(), cancel_event)

        self.assertEqual(stream.read.call_count, 1)
        self.assertEqual(audio.size, 0)
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        audio_device.terminate.assert_called_once()

    def test_stream_and_audio_device_close_when_read_fails(self):
        stream = mock.Mock()
        stream.read.side_effect = RuntimeError("read failed")
        audio_device = mock.Mock()
        audio_device.open.return_value = stream

        with mock.patch.object(wt.pyaudio, "PyAudio", return_value=audio_device):
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                wt.record_manual_audio(threading.Event(), threading.Event())

        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        audio_device.terminate.assert_called_once()

    def test_maximum_duration_closes_stream_and_returns_captured_audio(self):
        stream = mock.Mock()
        stream.read.return_value = np.zeros(wt.CHUNK_SIZE, dtype=np.int16).tobytes()
        audio_device = mock.Mock()
        audio_device.open.return_value = stream

        with mock.patch.object(wt.pyaudio, "PyAudio", return_value=audio_device):
            audio = wt.record_manual_audio(threading.Event(), threading.Event())

        self.assertEqual(stream.read.call_count, wt.MAX_RECORD_SECS * 12.5)
        self.assertEqual(audio.size, stream.read.call_count * wt.CHUNK_SIZE)
        stream.close.assert_called_once()
        audio_device.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
