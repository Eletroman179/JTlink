import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
import requests
import webrtcvad
import collections
import sys
import resampy
import json
import re
import os
import termios
import tty
from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.text import Text
from rich.live import Live
import subprocess

console = Console()

def copy(text):
    subprocess.run(['wl-copy'], input=text.encode(), check=True)


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch1 + ch2 + ch3
        elif ch1 == "\x03":
            raise KeyboardInterrupt()
        else:
            return ch1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def draw_menu(items, start=0):
    longest = max(len(x) for x in items)
    selected = start

    # Print empty lines to reserve space
    sys.stdout.write("\n" * len(items))
    sys.stdout.flush()

    while True:
        # Move cursor up to the start of menu
        sys.stdout.write(f"\x1b[{len(items)}F")
        sys.stdout.flush()

        for i, item in enumerate(items):
            padded = item.center(longest)
            if i == selected:
                sys.stdout.write(f"\033[7m[*] {padded}\033[0m\n")  # Reverse video highlight
            else:
                sys.stdout.write(f"[ ] {padded}\n")

        sys.stdout.flush()

        key = read_key()

        if key == "\x1b[A":
            selected = (selected - 1) % len(items)
        elif key == "\x1b[B":
            selected = (selected + 1) % len(items)
        elif key in ("\r", "\n"):
            # Redraw menu with only selected marked, no highlight
            sys.stdout.write(f"\x1b[{len(items)}F")
            for i, item in enumerate(items):
                prefix = "[*]" if i == selected else "[ ]"
                padded = item.center(longest)
                sys.stdout.write(f"{prefix} {padded}\n")
            sys.stdout.flush()
            return items[selected]

def extract_filename(text):
    matches = re.findall(r'\b[\w\-.]+\.[a-zA-Z0-9]{1,5}\b', text)
    return matches[0] if matches and os.path.isfile(matches[0]) else None

def ask(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
    "model": "llama3",
    "prompt": prompt,
    "stream": True,
    "system": (
    "You are a helpful assistant. You are able to speak naturally and clearly. "
    "When your response includes bash commands, wrap them exactly like this: "
    "<[RUN_BASH]>your commands here</[RUN_BASH]> "
    "Example: <[RUN_BASH]>echo 'hello world'</[RUN_BASH]> "
    "Do not include any extra characters, symbols, or text inside or outside these tags. "
    "When your response includes code (e.g. Python), wrap it exactly like this: "
    "```your code here```"
    "Example: ```print('hello world')``` "
    "Only wrap actual bash commands or code. Do not wrap explanations, descriptions, or non-code text. "
    "Use this format consistently. Do not use markdown formatting."
    )
    }
    full_response = ""
    output = Text("", style="bold cyan")

    try:
        with requests.post(url, json=data, stream=True) as res:
            res.raise_for_status()
            with Live(Panel(output, title="Ollama said", border_style="green"), refresh_per_second=20, console=console) as live:
                for line in res.iter_lines():
                    if line:
                        try:
                            js = json.loads(line.decode("utf-8"))
                            chunk = js.get("response", "")
                            output.append(chunk)
                            full_response += chunk
                            live.update(Panel(output, title="Ollama said", border_style="cyan"))
                        except json.JSONDecodeError:
                            pass
    except KeyboardInterrupt:
        console.print("[bold red]🛑 Exiting...[/]")
        exit(0)
        
    return full_response

def ask_with_dynamic_file_context(user_text):
    filename = extract_filename(user_text)
    if filename:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        prompt = f"Here is the content of the file {filename}:\n{content}\n\nUser question: {user_text}\nAnswer:"
    else:
        prompt = user_text
    return ask(prompt)

def find_mic_by_name(keyword="USB"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0 and keyword.lower() in d['name'].lower(): # type: ignore
            print(f"Found device {i}: {d['name']}") # type: ignore
            return i
    print(f"No device with keyword '{keyword}' found.")
    return None

class AudioBuffer:
    def __init__(self, frame_bytes):
        self.frame_bytes = frame_bytes
        self.buffer = bytearray()

    def add(self, data):
        self.buffer.extend(data)

    def get_frame(self):
        if len(self.buffer) >= self.frame_bytes:
            frame = self.buffer[:self.frame_bytes]
            self.buffer = self.buffer[self.frame_bytes:]
            return frame
        return None

def record_until_silence(mic_index=0, filename="input.wav", aggressiveness=3, silence_duration=1.0):
    info = sd.query_devices(mic_index)
    orig_samplerate = int(info['default_samplerate']) # type: ignore
    channels = 1

    print(f"Recording from '{info['name']}' at {orig_samplerate} Hz") # type: ignore

    vad = webrtcvad.Vad(aggressiveness)
    sd.default.device = (mic_index, None) # type: ignore

    frame_duration = 30
    frame_size = int(orig_samplerate * frame_duration / 1000)
    frame_bytes = frame_size * 2

    audio_buffer = AudioBuffer(frame_bytes)
    speech_frames = collections.deque(maxlen=int(silence_duration * 1000 / frame_duration))
    recorded_frames = []

    def callback(indata, frames_count, time, status):
        if status:
            # ignore minor overflow warning to reduce noise
            if str(status) != 'Input overflowed':
                print(status, file=sys.stderr)
        data_bytes = indata[:, 0].tobytes()
        audio_buffer.add(data_bytes)

        while True:
            frame = audio_buffer.get_frame()
            if frame is None:
                break
            frame_np = np.frombuffer(frame, dtype='int16').astype(np.float32)
            frame_16k = resampy.resample(frame_np, orig_samplerate, 16000)
            frame_16k = np.clip(frame_16k, -32768, 32767).astype(np.int16).tobytes()

            is_speech = vad.is_speech(frame_16k, 16000)
            speech_frames.append(is_speech)
            recorded_frames.append(frame)

    speaking = False
    with sd.InputStream(samplerate=orig_samplerate, channels=channels, dtype='int16', callback=callback):
        while True:
            if len(speech_frames) == speech_frames.maxlen:
                silence = all(not f for f in speech_frames)
                if recorded_frames:
                    last_frame = np.frombuffer(recorded_frames[-1], dtype='int16')
                    volume = np.abs(last_frame).mean() / 32768
                    bar = "█" * min(int(volume * 30), 30) + "-" * (30 - min(int(volume * 30), 30))
                    status = "SPEAKING" if not silence else "SILENCE"
                else:
                    bar = "-" * 30
                    status = "SILENCE"

                print(f"\r🎙 Mic Level: {bar} {status}   ", end="\r", flush=True)

                if silence and speaking:
                    print()
                    print("Silence detected, stopping recording.")
                    break
                elif not silence:
                    speaking = True

            sd.sleep(frame_duration)

    audio_data = b''.join(recorded_frames)
    audio_np = np.frombuffer(audio_data, dtype='int16')
    audio_16k = resampy.resample(audio_np.astype(np.float32), orig_samplerate, 16000)
    audio_16k = np.clip(audio_16k, -32768, 32767).astype(np.int16)
    write(filename, 16000, audio_16k)
    print(f"Recording saved as {filename} (resampled to 16000 Hz)")
    return filename

def transcribe(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

def run():
    mic = find_mic_by_name("USB")
    if mic is None:
        console.print("[bold yellow]USB mic not found, using default input.[/]")
        mic = sd.default.device[0]

    history = ""

    while True:
        console.print("\n[bold cyan]🎙 Listening... Speak now![/bold cyan]")
        try:
            filename = record_until_silence(mic_index=mic, aggressiveness=3, silence_duration=1)  # type: ignore
        except KeyboardInterrupt:
            console.print("[bold red]🛑 Exiting...[/]")
            exit(0)
            
        text = transcribe(filename).strip() # type: ignore
        console.print(Panel(Text(text, style="bold green"), title="You said"))

        if text.lower() in ("exit", "quit", "stop"):
            console.print("[bold red]🛑 Exiting...[/]")
            break

        history += f"\nUser: {text}"
        response = ask_with_dynamic_file_context(history)

        # Extract bash commands from AI response
        matches = re.findall(r"<\[RUN_BASH\]>(.*?)</\[RUN_BASH\]>", response, re.DOTALL)

        if matches:
            console.print(
                Panel(
                    Text("\n".join(matches), style="bold yellow"),
                    title="Ollama commands",
                    expand=True,
                )
            )
            print("Do you want to run it?")
            if draw_menu(["Yes", "No"]) == "Yes":
                for cmd in matches:
                    cmd_clean = cmd.strip()
                    # Remove leading $ if present:
                    if cmd_clean.startswith('$'):
                        cmd_clean = cmd_clean[1:].lstrip()
                    console.print(f"Running command: {cmd_clean}")
                    os.system(f"bash -c '{cmd_clean}'")
                    
        # Extract bash commands from AI response
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)

        if matches:
            copy("\n".join(matches))
            console.print(
                Panel(
                    Text("\n".join(matches), style="bold yellow"),
                    title="Ollama code",
                    expand=True,
                )
            )


        history += f"\nOllama: {response}"
        
    os.remove("input.wav")

if __name__ == "__main__":
    run()