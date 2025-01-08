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
import Voice
import torch
from shared_resources import voice_queue

# Silence warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class VoiceTranscriber:
    def __init__(self, callback_function):
        self.silence_threshold = 0.0018  # Adjust based on your microphone
        self.silence_duration = 1.0    # Seconds of silence to mark end of speech
        self.last_speech_time = time.time()
        self.speech_buffer = []
        
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

                # Calculate audio level
                audio_level = np.abs(audio_data).mean()
                print(f"\rAudio level: {audio_level:.4f}", end="")

                # Only process audio if level is above threshold
                if audio_level > 0.0018:
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
                                print(f"\nDetected: {text}")
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                
                                # Check for wake words when not listening
                                if not self.is_listening:
                                    if any(wake_word in text for wake_word in self.wake_words):
                                        print(f"\n[{timestamp}] Wake word detected! Listening...")
                                        Voice.piper_speak("Yes, I'm here.")
                                        self.is_listening = True
                                        self.speech_buffer = []
                                        continue

                                # Process speech while listening
                                if self.is_listening:
                                    # Check for stop words
                                    if any(stop_word in text for stop_word in self.stop_words):
                                        print(f"\n[{timestamp}] Stop word detected. Going to sleep.")
                                        Voice.piper_speak("Goodbye for now.")
                                        self.is_listening = False
                                        self.speech_buffer = []
                                        continue

                                    # Process normal conversation
                                    if len(text) > 3:  # Ignore very short sounds
                                        self.last_speech_time = time.time()
                                        self.speech_buffer.append(text)

                    except Exception as e:
                        print(f"\nError during transcription: {str(e)}")
                else:
                    # If audio level is low and we have buffered speech, process it
                    if self.is_listening and self.speech_buffer and time.time() - self.last_speech_time > self.silence_duration:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        complete_utterance = " ".join(self.speech_buffer)
                        print(f"\n[{timestamp}] Processing complete utterance: {complete_utterance}")
                        self.speech_buffer = []
                        
                        # Temporarily stop listening while processing
                        self.is_listening = False
                        
                        # Process the message and get response
                        response = self.callback(complete_utterance)
                        
                        # Wait for voice response to complete
                        while not voice_queue.empty():
                            time.sleep(0.1)
                            
                        # Resume listening after response
                        self.is_listening = True
                        print("\nListening for next input...")

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

