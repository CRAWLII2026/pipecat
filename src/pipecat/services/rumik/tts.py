#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rumik Silk text-to-speech service implementations.

This module provides both WebSocket and HTTP-based text-to-speech services
using Rumik's Silk API for streaming and batch audio synthesis.

Two models are supported:
- **muga**: Expressive model, steered by tone tags (e.g. ``[happy]``)
- **mulberry**: Faster model, steered by a natural-language description
  and/or a preset speaker voice

Audio is 24 kHz mono signed 16-bit PCM. The HTTP endpoint wraps it in WAV;
the WebSocket stream sends raw PCM chunks.

See https://docs.rumik.ai/ for full API details.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import aiohttp
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import (
    InterruptibleTTSService,
    TextAggregationMode,
    TTSService,
)
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class RumikTTSSettings(TTSSettings):
    """Settings for RumikTTSService and RumikHttpTTSService.

    Parameters:
        model: TTS model — ``"muga"`` (expressive) or ``"mulberry"`` (faster).
        voice: Preset speaker voice name (mulberry only). Maps to the
            ``speaker`` field in the Rumik API. One of: emma, mia, sophia,
            ava, ira, siya, aisha, zoya, lucas, noah, theo, adam.
        description: Natural-language voice description (mulberry only).
        temperature: Sampling temperature (0.01-1.0). Defaults to 0.6.
        top_p: Nucleus sampling threshold. Defaults to 0.95.
        top_k: Top-k sampling value. Defaults to 50.
    """

    model: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    description: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_k: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {"speaker": "voice"}


class RumikTTSService(InterruptibleTTSService):
    """WebSocket-based text-to-speech service using Rumik's Silk API.

    Provides streaming TTS with real-time audio generation and interruption
    support. Uses Rumik's WebSocket API for low-latency audio streaming.

    The connection flow is:
    1. POST to ``/v1/tts/ws-connect`` to mint a one-shot WebSocket session
    2. Connect to the returned ``ws_url`` with the ``token``
    3. Send JSON frames with text and synthesis parameters
    4. Receive raw PCM int16 LE @ 24 kHz mono as binary frames
    5. Receive ``{"type": "done"}`` when synthesis is complete

    Interruption (barge-in) is handled by reconnecting the WebSocket, which
    cancels any in-progress synthesis on the server side.

    Example::

        tts = RumikTTSService(
            api_key="rk_live_...",
            gateway_url="https://silk-api.rumik.ai",
            settings=RumikTTSService.Settings(
                model="mulberry",
                voice="ira",
                description="a warm 30s indian voice, smooth timbre",
            ),
        )
    """

    Settings = RumikTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        gateway_url: str = "https://silk-api.rumik.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize Rumik TTS service.

        Args:
            api_key: Rumik API key (e.g. ``rk_live_...``).
            gateway_url: Rumik Silk API base URL.
            sample_rate: Audio sample rate in Hz. Defaults to 24000.
            settings: TTS settings including model, voice, description.
            text_aggregation_mode: How to aggregate incoming text.
            **kwargs: Additional arguments passed to parent class.
        """
        default_settings = self.Settings(
            model="muga",
            voice=None,
            description=None,
            language=None,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=False,
            pause_frame_processing=True,
            append_trailing_space=True,
            sample_rate=sample_rate or 24000,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._gateway_url = gateway_url.rstrip("/")
        self._receive_task = None
        self._ws_url = None
        self._ws_token = None
        self._tts_done_event = asyncio.Event()

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            # Step 1: Mint a one-shot WebSocket session
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._gateway_url}/v1/tts/ws-connect",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._settings.model or "muga",
                        "text": " ",
                    },
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"[RUMIK-TRACE] ws-connect failed ({resp.status}): {error_text}", flush=True)
                        raise Exception(
                            f"Rumik ws-connect failed ({resp.status}): {error_text}"
                        )
                    data = await resp.json()
                    self._ws_url = data["ws_url"]
                    self._ws_token = data["token"]
                    print(f"[RUMIK-TRACE] ws-connect success, ws_url={self._ws_url}", flush=True)

            # Step 2: Connect to the WebSocket
            url = f"{self._ws_url}?token={self._ws_token}"
            self._websocket = await websocket_connect(url)

            await self._call_event_handler("on_connected")
            logger.debug(f"{self}: Connected to Rumik WebSocket")
        except Exception as e:
            await self.push_error(error_msg=f"Rumik connect error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                try:
                    await self._websocket.send(json.dumps({"type": "close"}))
                except Exception:
                    pass
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Rumik disconnect error: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._ws_url = None
            self._ws_token = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Rumik WebSocket not connected")

    def _build_msg(self, text: str) -> dict:
        msg: dict[str, Any] = {"text": text}
        model = self._settings.model or "muga"
        if model == "mulberry":
            if self._settings.voice:
                msg["speaker"] = self._settings.voice
            if self._settings.description:
                msg["description"] = self._settings.description
        if self._settings.temperature is not None:
            msg["temperature"] = self._settings.temperature
        if self._settings.top_p is not None:
            msg["top_p"] = self._settings.top_p
        if self._settings.top_k is not None:
            msg["top_k"] = self._settings.top_k
        return msg

    async def _close_context(self, context_id: str):
        await self.stop_all_metrics()
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
            except Exception:
                pass

    async def on_audio_context_interrupted(self, context_id: str):
        await self._close_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        await self._close_context(context_id)
        await super().on_audio_context_completed(context_id)

    async def _receive_messages(self):
        """Receive and process WebSocket messages from Rumik.

        Required by WebsocketService. Delegates to the internal receive logic.
        """
        await self._receive_task_handler(self._report_error)

    async def _receive_task_handler(self, report_error):
        """Process incoming WebSocket messages from Rumik."""
        try:
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    # Raw PCM int16 LE @ 24 kHz mono
                    context_id = self.get_active_audio_context_id()
                    if not context_id:
                        continue
                    frame = TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    await self.append_to_audio_context(context_id, frame)
                    print(f"[RUMIK-TRACE] received audio: {len(message)} bytes", flush=True)
                else:
                    # JSON text frame
                    try:
                        msg = json.loads(message)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    msg_type = msg.get("type")
                    print(f"[RUMIK-TRACE] received JSON msg: {msg}", flush=True)
                    if msg_type == "done":
                        print(f"[RUMIK-TRACE] received 'done' message", flush=True)
                        await self.stop_ttfb_metrics()
                        context_id = self.get_active_audio_context_id()
                        if context_id:
                            await self.append_to_audio_context(
                                context_id, TTSStoppedFrame(context_id=context_id)
                            )
                            await self.remove_audio_context(context_id)
                        self._tts_done_event.set()
                    elif msg.get("error"):
                        print(f"[RUMIK-TRACE] received error: {msg.get('error')}", flush=True)
                        await self.push_frame(TTSStoppedFrame())
                        await self.stop_all_metrics()
                        await self.push_error(
                            error_msg=f"Rumik error: {msg.get('error')}"
                        )
                        self.reset_active_audio_context()
                        self._tts_done_event.set()
                    elif msg_type == "cancelled":
                        print(f"[RUMIK-TRACE] received 'cancelled' message", flush=True)
                        logger.debug(f"{self}: Rumik synthesis cancelled")
                        self._tts_done_event.set()
                    elif msg_type == "timeout":
                        print(f"[RUMIK-TRACE] received 'timeout' message", flush=True)
                        logger.warning(f"{self}: Rumik WebSocket timeout")
                        await self.push_error(error_msg="Rumik WebSocket timeout")
                        self._tts_done_event.set()
        except Exception as e:
            print(f"[RUMIK-TRACE] receive error: {e}", flush=True)
            await report_error(ErrorFrame(error=f"Rumik receive error: {e}", exception=e))

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Rumik's streaming WebSocket API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        print(f"[RUMIK-TRACE] run_tts called: text='{text[:80]}'", flush=True)
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                print(f"[RUMIK-TRACE] run_tts: websocket closed, reconnecting", flush=True)
                await self._connect()

            try:
                if not self.audio_context_available(context_id):
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)

                self._tts_done_event.clear()
                msg = self._build_msg(text)
                print(f"[RUMIK-TRACE] run_tts: sending msg to Rumik: {json.dumps(msg)[:200]}", flush=True)
                await self._get_websocket().send(json.dumps(msg))
                print(f"[RUMIK-TRACE] run_tts: sent message to Rumik, waiting for audio", flush=True)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Rumik TTS error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            try:
                await asyncio.wait_for(self._tts_done_event.wait(), timeout=30.0)
                print(f"[RUMIK-TRACE] run_tts: done event received, TTS complete", flush=True)
            except asyncio.TimeoutError:
                print(f"[RUMIK-TRACE] run_tts: TIMEOUT waiting for done event", flush=True)
                yield ErrorFrame(error="Rumik TTS timeout: no response in 30s")
                yield TTSStoppedFrame(context_id=context_id)
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Rumik TTS error: {e}")


class RumikHttpTTSService(TTSService):
    """HTTP-based text-to-speech service using Rumik's Silk API.

    Provides batch TTS synthesis using Rumik's HTTP endpoint. Suitable for
    use cases where streaming is not required.

    Example::

        tts = RumikHttpTTSService(
            api_key="rk_live_...",
            gateway_url="https://silk-api.rumik.ai",
            aiohttp_session=session,
            settings=RumikHttpTTSService.Settings(
                model="mulberry",
                voice="ira",
                description="a warm 30s indian voice",
            ),
        )
    """

    Settings = RumikTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        gateway_url: str = "https://silk-api.rumik.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Rumik HTTP TTS service.

        Args:
            api_key: Rumik API key (e.g. ``rk_live_...``).
            aiohttp_session: Shared aiohttp session for HTTP requests.
            gateway_url: Rumik Silk API base URL.
            sample_rate: Audio sample rate in Hz. Defaults to 24000.
            settings: TTS settings including model, voice, description.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        default_settings = self.Settings(
            model="muga",
            voice=None,
            description=None,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate or 24000,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._gateway_url = gateway_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Rumik's HTTP API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS (HTTP) [{text}]")

        try:
            payload: dict[str, Any] = {
                "model": self._settings.model or "muga",
                "text": text,
            }
            if self._settings.model == "mulberry":
                if self._settings.voice:
                    payload["speaker"] = self._settings.voice
                if self._settings.description:
                    payload["description"] = self._settings.description
            if self._settings.temperature is not None:
                payload["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                payload["top_p"] = self._settings.top_p
            if self._settings.top_k is not None:
                payload["top_k"] = self._settings.top_k

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            url = f"{self._gateway_url}/v1/tts"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Rumik API error: {error_text}")
                    return

                audio_data = await response.read()

            await self.start_tts_usage_metrics(text)

            # Strip WAV header (first 44 bytes) if present
            if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
                logger.debug("Stripping WAV header from Rumik audio data")
                audio_data = audio_data[44:]

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

            yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
