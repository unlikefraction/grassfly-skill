import argparse
import asyncio
import json
import os
import sys

import numpy as np
import opuslib
import soxr
import websockets

try:
    from loguru import logger as loguru_logger

    loguru_logger.remove()
except Exception:
    pass

# Add parent directory to path to import env.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import env
except ImportError:
    print("Error: Could not import env.py. Make sure it exists in the parent directory.")
    sys.exit(1)

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.gemini_live import GeminiLiveLLMService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

BACKEND_URL = "wss://opencall-backend.unlikefraction.com"
USERNAME = "gemini-oncall" # this is who you are. this is your caller id.
NETWORK_RATE = 48000
NETWORK_CHANNELS = 1
NETWORK_CHUNK_SIZE = 960
INPUT_RATE = 16000

SYSTEM_PROMPT_BASE = (
    "You are an AI assistant named Silicon, having a phone call with a human."
    "Your job is to either convey information given to you in the following system prompt, or to help the human by understanding their question to the best of your ability."
    "You dont have direct access to user's information, so if the user wants to know something that you dont already know, tell them you'll get back to them after you find out the answer. Tell them you'll call them back."
    "Sometimes you will also be given information that the user had requested for earlier, and you tell that to the human."
    "Remember, you can only convey information given to you in the system prompt, and you cannot make up information that you dont have. If you dont know the answer to something, say you dont know but you'll find out and call them back."
)

# link to your SOUL.md and other files that will help the assistant understand the user better.
# render it in text and add to the base system prompt.


class NetworkInputTransport(BaseInputTransport):
    def __init__(self, params: TransportParams, input_queue: asyncio.Queue):
        super().__init__(params)
        self.input_queue = input_queue
        self.stop_event = asyncio.Event()
        self.decoder = opuslib.Decoder(NETWORK_RATE, NETWORK_CHANNELS)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._create_audio_task()
        asyncio.create_task(self.read_loop())

    async def read_loop(self):
        while not self.stop_event.is_set():
            audio_bytes = await self.input_queue.get()
            if audio_bytes is None:
                return
            await self.push_audio_frame(
                InputAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=INPUT_RATE,
                    num_channels=1,
                )
            )

    async def stop(self, frame: EndFrame):
        self.stop_event.set()
        await self.input_queue.put(None)
        await super().stop(frame)


class NetworkOutputTransport(BaseOutputTransport):
    def __init__(self, params: TransportParams, output_queue: asyncio.Queue, transcript: list):
        super().__init__(params)
        self.output_queue = output_queue
        self.transcript = transcript
        self.assistant_line_open = False
        self.assistant_chunks = []
        self.last_assistant_chunk = ""

    def _close_assistant_line(self):
        if not self.assistant_line_open:
            return
        text = " ".join(self.assistant_chunks).strip()
        print()
        if text:
            self.transcript.append(("assistant", text))
        self.assistant_line_open = False
        self.assistant_chunks = []
        self.last_assistant_chunk = ""

    def flush(self):
        self._close_assistant_line()

    async def process_frame(self, frame, direction):
        if isinstance(frame, (StartFrame, EndFrame)):
            await super().process_frame(frame, direction)
            return

        if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
            audio_np = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32)
            if frame.sample_rate != NETWORK_RATE:
                audio_np = soxr.resample(audio_np, frame.sample_rate, NETWORK_RATE)
            audio_np = np.clip(audio_np, -32768, 32767)
            await self.output_queue.put(audio_np.astype(np.int16).tobytes())
            return

        if isinstance(frame, (LLMTextFrame, TTSTextFrame)) and getattr(frame, "text", None):
            text = frame.text.strip()
            if text:
                if text != self.last_assistant_chunk:
                    if not self.assistant_line_open:
                        print(f"Assistant: {text}", end="", flush=True)
                        self.assistant_line_open = True
                    else:
                        print(f" {text}", end="", flush=True)
                    self.assistant_chunks.append(text)
                    self.last_assistant_chunk = text
            return

        if frame.__class__.__name__ in {"TTSStoppedFrame", "LLMFullResponseEndFrame"}:
            self._close_assistant_line()
            return

class TranscriptProcessor(FrameProcessor):
    def __init__(self, transcript: list, output_transport: NetworkOutputTransport):
        super().__init__()
        self.transcript = transcript
        self.output_transport = output_transport

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM and isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame)
        ):
            text = frame.text.strip()
            if text:
                self.output_transport.flush()
                self.transcript.append(("user", text))
                print(f"User: {text}")
        await self.push_frame(frame, direction)


async def main():
    parser = argparse.ArgumentParser(description="Gemini caller")
    parser.add_argument("--target", required=True, help="Target username")
    parser.add_argument("--convey", help="Additional information to be conveyed to the user during the call")
    args = parser.parse_args()

    model = "models/gemini-2.5-flash-native-audio-preview-12-2025"
    print(f"Connecting to Gemini: {model}")

    if args.convey:
        system_prompt = SYSTEM_PROMPT_BASE + "\n\nInformation to convey: " + args.convey
    else:
        system_prompt = SYSTEM_PROMPT_BASE

    llm = GeminiLiveLLMService(api_key=env.GEMINI_API_KEY, model=model, system_instruction=system_prompt)
    print("Gemini connected")

    uri = f"{BACKEND_URL}/ws/{USERNAME}"
    headers = {"Authorization": f"Bearer {env.GEMINI_API_KEY}"}

    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    transcript = []

    transport_params = TransportParams(
        audio_out_enabled=True,
        audio_in_enabled=True,
        audio_in_sample_rate=INPUT_RATE,
        audio_out_sample_rate=24000,
    )

    network_input = NetworkInputTransport(transport_params, input_queue)
    network_output = NetworkOutputTransport(transport_params, output_queue, transcript)
    transcript_processor = TranscriptProcessor(transcript, network_output)

    pipeline = Pipeline([network_input, transcript_processor, llm, network_output])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        await websocket.send(json.dumps({"type": "init_stream", "target": args.target}))
        print(f"Calling @{args.target}")
        def exit_now():
            os._exit(0)

        call_connected = False
        ringing_printed = False
        audio_packets_since_answer = 0
        silent_ticks = 0
        receiving_audio = False
        caller_id = args.target
        stop_event = asyncio.Event()

        async def receive_loop():
            nonlocal call_connected, ringing_printed
            nonlocal audio_packets_since_answer, silent_ticks, receiving_audio, caller_id

            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    if call_connected and receiving_audio:
                        silent_ticks += 1
                        if silent_ticks >= 10:
                            print("Stopped Receiving Audio")
                            receiving_audio = False
                            audio_packets_since_answer = 0
                    continue
                except websockets.ConnectionClosed:
                    print(f"Call disconncted by @{caller_id}")
                    exit_now()

                if isinstance(message, bytes):
                    silent_ticks = 0
                    try:
                        pcm_48k = network_input.decoder.decode(message, NETWORK_CHUNK_SIZE)
                    except opuslib.OpusError:
                        continue

                    audio_np = np.frombuffer(pcm_48k, dtype=np.int16).astype(np.float32)
                    pcm_16k = soxr.resample(audio_np, NETWORK_RATE, INPUT_RATE)
                    pcm_16k = np.clip(pcm_16k, -32768, 32767).astype(np.int16).tobytes()
                    await input_queue.put(pcm_16k)

                    if call_connected and not receiving_audio:
                        audio_packets_since_answer += 1
                        if audio_packets_since_answer >= 10:
                            print("Receiving audio")
                            receiving_audio = True
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "state_update":
                    status = data.get("status")
                    peer = data.get("peer")
                    if peer:
                        caller_id = peer

                    if status == "calling":
                        print("calling")
                    elif status == "ringing" and not ringing_printed:
                        print("ringing")
                        ringing_printed = True
                    elif status == "connected":
                        print("answered")
                        call_connected = True
                        audio_packets_since_answer = 0
                        silent_ticks = 0
                        receiving_audio = False
                    elif status == "idle" and call_connected:
                        print(f"Call disconncted by @{caller_id}")
                        exit_now()
                    elif status == "busy":
                        print("busy")
                        exit_now()
                elif data.get("type") == "error":
                    print(f"error: {data.get('message', 'unknown')}" )
                    exit_now()

        async def send_loop():
            encoder = opuslib.Encoder(NETWORK_RATE, NETWORK_CHANNELS, opuslib.APPLICATION_VOIP)
            bytes_per_frame = NETWORK_CHUNK_SIZE * 2
            pending = b""

            while True:
                try:
                    chunk = await asyncio.wait_for(output_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    if stop_event.is_set() and not pending:
                        return
                    continue
                if chunk is None:
                    return

                pending += chunk
                while len(pending) >= bytes_per_frame:
                    frame = pending[:bytes_per_frame]
                    pending = pending[bytes_per_frame:]
                    try:
                        opus_data = encoder.encode(frame, NETWORK_CHUNK_SIZE)
                        await websocket.send(opus_data)
                    except websockets.ConnectionClosed:
                        return

        pipeline_task = asyncio.create_task(runner.run(task))
        send_task = asyncio.create_task(send_loop())
        await receive_loop()
        stop_event.set()

        await input_queue.put(None)
        await output_queue.put(None)
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)

        await asyncio.wait_for(task.cancel(reason="call disconnected"), timeout=2)
        await asyncio.wait_for(runner.cancel(), timeout=2)
        pipeline_task.cancel()
        await asyncio.gather(pipeline_task, return_exceptions=True)
        try:
            await asyncio.wait_for(task.cleanup(), timeout=2)
        except asyncio.TimeoutError:
            pass
        try:
            await asyncio.wait_for(runner.cleanup(), timeout=2)
        except asyncio.TimeoutError:
            pass

    await input_queue.put(None)
    await output_queue.put(None)
    network_output.flush()

    if transcript:
        print("\nTranscript:")
        for role, text in transcript:
            print(f"{role}: {text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped")
