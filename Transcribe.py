import os
import sys
import subprocess
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time
from datetime import datetime
import torch
import Voice

# Silence warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class VoiceTranscriber:
    def __init__(self, callback_function):
        self.callback = callback_function
        self.wake_words = ["fred", "hey fred", "okay fred"]
        self.stop_words = ["goodbye", "bye fred", "stop listening"]
        
        # Audio configuration
        self.samplerate = 16000
        self.channels = 1
        self.block_duration = 5
        self.blocksize = int(self.block_duration * self.samplerate)
        
        # Initialize queues and events
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.terminate_event = threading.Event()
        
        # Initialize Whisper model
        self.setup_model()
        
        # State tracking
        self.is_listening = False
        self.is_running = False
        self.current_conversation = []

    def setup_model(self):
        """Initialize the Whisper model with optimal settings"""
        model_size = "medium"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Model initialized on {device} using {compute_type}")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """Process audio stream and detect wake words/commands"""
        while not self.terminate_event.is_set():
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_data = audio_data.flatten().astype(np.float32)

                try:
                    segments, _ = self.model.transcribe(
                        audio_data, 
                        language="en",
                        beam_size=5,
                        word_timestamps=True
                    )

                    for segment in segments:
                        text = segment.text.strip().lower()
                        if text:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            
                            # Check for wake words when not listening
                            if not self.is_listening:
                                if any(wake_word in text for wake_word in self.wake_words):
                                    print(f"\n[{timestamp}] Wake word detected! Listening...")
                                    Voice.piper_speak("Yes, I'm here.")
                                    self.is_listening = True
                                    self.current_conversation = []
                                    continue

                            # Process speech while listening
                            if self.is_listening:
                                # Check for stop words
                                if any(stop_word in text for stop_word in self.stop_words):
                                    print(f"\n[{timestamp}] Stop word detected. Going to sleep.")
                                    Voice.piper_speak("Goodbye for now.")
                                    self.is_listening = False
                                    self.current_conversation = []
                                    continue

                                # Process normal conversation
                                if len(text) > 3:  # Ignore very short sounds
                                    print(f"[{timestamp}] {text}")
                                    response = self.callback(text)
                                    self.current_conversation.append(text)

                except Exception as e:
                    print(f"Error during transcription: {str(e)}")

            time.sleep(0.1)

    def start(self):
        """Start the voice transcription system"""
        if not self.is_running:
            self.is_running = True
            self.terminate_event.clear()

            # Start processing thread
            self.process_thread = threading.Thread(
                target=self.process_audio,
                daemon=True
            )
            self.process_thread.start()

            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize
            )
            self.stream.start()

            print("\nVoice system initialized. Waiting for wake word...")

    def stop(self):
        """Stop the voice transcription system"""
        if self.is_running:
            self.terminate_event.set()
            self.stream.stop()
            self.stream.close()
            self.process_thread.join()
            self.is_running = False
            self.is_listening = False
            print("\nVoice system stopped.")

def initialize_voice_system(callback_function):
    """Initialize and return a voice transcriber instance"""
    transcriber = VoiceTranscriber(callback_function)
    transcriber.start()
    return transcriber

# Example usage
if __name__ == "__main__":
    def example_callback(text):
        print(f"Callback received: {text}")
        return "Processing complete"

    transcriber = initialize_voice_system(example_callback)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        transcriber.stop()

