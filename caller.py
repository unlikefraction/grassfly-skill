import asyncio
import pyaudio
import websockets
import json
import argparse
import sys
import opuslib

# Configuration
BACKEND_URL = "wss://opencall-backend.unlikefraction.com" # WSS for secure tunnel
USERNAME = "python-headless-1"
RATE = 48000
CHUNK = 960  # 20ms at 48kHz
CHANNELS = 1
FORMAT = pyaudio.paInt16

import datetime

async def run_client(target_username):
    def log(msg):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    uri = f"{BACKEND_URL}/ws/{USERNAME}"
    log(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            log("Connected to Signaling Server")
            
            # Handshake: Tell server who we want to talk to
            await websocket.send(json.dumps({
                "type": "init_stream",
                "target": target_username
            }))
            log(f"Requested stream with {target_username}")
            
            # Audio Setup
            p = pyaudio.PyAudio()
            
            # Opus Setup
            encoder = opuslib.Encoder(RATE, CHANNELS, opuslib.APPLICATION_VOIP)
            decoder = opuslib.Decoder(RATE, CHANNELS)
            
            # Input Stream (Mic)
            input_stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Output Stream (Speaker)
            output_stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True
            )
            
            log("Audio streams initialized. Streaming with Opus... (Ctrl+C to stop)")
            
            async def send_audio():
                try:
                    while True:
                        # Blocking read, run in executor to not block loop
                        pcm_data = await asyncio.get_event_loop().run_in_executor(
                            None, input_stream.read, CHUNK, False
                        )
                        
                        # Encode PCM to Opus
                        opus_data = encoder.encode(pcm_data, CHUNK)
                        
                        await websocket.send(opus_data)
                        await asyncio.sleep(0) # Yield
                except Exception as e:
                    log(f"Send Error: {e}")

            async def receive_audio():
                loop = asyncio.get_event_loop()
                call_active = False
                try:
                    async for message in websocket:
                        if isinstance(message, bytes):
                            try:
                                # Decode Opus to PCM
                                pcm_data = decoder.decode(message, CHUNK)
                                # Play Audio (Offload to thread to prevent blocking event loop)
                                await loop.run_in_executor(None, output_stream.write, pcm_data)
                            except opuslib.OpusError as e:
                                log(f"Opus Decode Error: {e}")
                            
                            # log(".", end="", flush=True) # visual indicator - Disabled for cleaner log
                        else:
                            # Handle Control Messages
                            try:
                                data = json.loads(message)
                                msg_type = data.get("type")
                                
                                if msg_type == "state_update":
                                    status = data.get("status")
                                    peer = data.get("peer")
                                    log(f"[Status Update] {status.upper()} (Peer: {peer})")
                                    
                                    if status == "calling":
                                        log("Request Sent... Waiting for device to wake up...")
                                    elif status == "ringing":
                                        log(">>> VERIFIED: Remote Device is RINGING! <<<")
                                        call_active = True
                                    elif status == "connected":
                                        log("Remote ANSWERED. Audio starting...")
                                        call_active = True
                                    elif status == "busy":
                                        log("Remote is BUSY (On another call).")
                                        break # Exit
                                    elif status == "idle":
                                        if call_active:
                                            log("Call ENDED.")
                                            break # Exit loop
                                        else:
                                            log("Waiting/Idle...")
                                            # If we were "calling" and got "idle", it means they declined or failed
                                        
                                elif msg_type == "error":
                                    log(f"Error: {data.get('message')}")
                                    break
                            except json.JSONDecodeError:
                                log(f"Received non-JSON text message: {message}")
                except Exception as e:
                    log(f"Receive Error: {e}")

            # Run both tasks
            await asyncio.gather(send_audio(), receive_audio())
            
    except Exception as e:
        log(f"Connection Failed: {e}")
    finally:
        try:
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <target_username>")
        sys.exit(1)
    
    target = sys.argv[1]
    try:
        asyncio.run(run_client(target))
    except KeyboardInterrupt:
        print("Stopped.")
