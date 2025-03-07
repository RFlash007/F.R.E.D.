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
        self.silence_threshold = 0.0015  # Initial threshold
        self.calibration_samples = []
        self.calibration_duration = 2  # seconds
        self.silence_duration = 0.7    # Seconds of silence to mark end of speech
        self.last_speech_time = time.time()
        self.speech_buffer = []
        
        self.callback = callback_function
        self.wake_words = [
            "fred", "hey fred", "okay fred", 
            "hi fred", "excuse me fred", "fred are you there"
        ]
        self.stop_words = [
            "goodbye", "bye fred", "stop listening", 
            "that's all", "thank you fred", "sleep now"
        ]
        self.acknowledgments = [
            "Yes, I'm here.",
            "How can I help?",
            "I'm listening.",
            "What can I do for you?",
            "At your service."
        ]
        self.farewell_responses = [
            "Goodbye for now.",
            "Let me know if you need anything else.",
            "Have a great day.",
            "I'll be here when you need me."
        ]
        
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
        self.ui = None  # Add this line to store UI reference
        self.is_speaking = False
        self.interrupt_phrases = [
            "wait", "hold on", "stop", "pause", 
            "excuse me", "one moment"
        ]

    def setup_model(self):
        """Initialize the Whisper model with optimal settings"""
        model_size = "medium"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def set_ui(self, ui):
        """Set the UI instance for displaying transcribed text"""
        self.ui = ui

    def get_random_response(self, responses):
        """Get a random response from a list to make conversations more natural"""
        return np.random.choice(responses)

    def process_audio(self):
        """Process audio stream and detect wake words/commands"""
        while not self.terminate_event.is_set():
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_data = audio_data.flatten().astype(np.float32)

                # Calculate audio level
                audio_level = np.abs(audio_data).mean()
                
                # Debug audio levels periodically
                if self.is_listening:
                    print(f"\rAudio level: {audio_level:.6f} (Threshold: {self.silence_threshold:.6f})", end="")

                # Only process audio if level is above threshold
                if audio_level > self.silence_threshold:
                    try:
                        segments, _ = self.model.transcribe(
                            audio_data, 
                            language="en",
                            beam_size=5,
                            word_timestamps=True
                        )

                        for segment in segments:
                            text = segment.text.strip().lower()
                            
                            if text and text != "thanks for watching!":
                                print(f"\nDetected text: {text}")  # Debug detected text
                            
                                # Check for wake words when not listening
                                if not self.is_listening:
                                    if any(wake_word in text for wake_word in self.wake_words):
                                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Wake word detected! Listening...")
                                        response = self.get_random_response(self.acknowledgments)
                                        if self.ui:
                                            self.ui.display_message(response, "assistant")
                                        Voice.piper_speak(response)
                                        self.is_listening = True
                                        self.speech_buffer = []
                                        self.last_speech_time = time.time()
                                        continue

                                # Process speech while listening
                                if self.is_listening:
                                    # Check for stop words
                                    if any(stop_word in text for stop_word in self.stop_words):
                                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stop word detected. Going to sleep.")
                                        if self.ui:
                                            self.ui.display_message(f"{text}", "user")
                                        # Pass "goodbye" to process_message to trigger proper shutdown sequence
                                        self.callback("goodbye")
                                        self.is_listening = False
                                        self.speech_buffer = []
                                        while not voice_queue.empty():
                                            time.sleep(0.1)
                                        return
                                    
                                    # Add speech to buffer if it's not too short
                                    if len(text.split()) > 1:  # Only add if more than one word
                                        print(f"\nAdding to speech buffer: {text}")  # Debug buffer additions
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
                        
                        # Display the user's complete utterance in the UI
                        if self.ui:
                            self.ui.display_message(f"{complete_utterance}", "user")
                        
                        self.speech_buffer = []
                        
                        # Temporarily stop listening while processing
                        self.is_listening = False
                        
                        try:
                            # Process the message and get response
                            print("\nProcessing message through callback...")
                            self.is_speaking = True
                            response = self.callback(complete_utterance)
                            
                            # Display the response in the UI
                            if self.ui:
                                self.ui.display_message(f"{response}", "assistant")
                            
                            # Wait for voice response to complete
                            while not voice_queue.empty():
                                time.sleep(0.1)
                                
                            self.is_speaking = False
                            
                            # Resume listening after response
                            self.is_listening = True
                            print("\nListening for next input...")
                        except Exception as e:
                            print(f"\nError in callback processing: {str(e)}")
                            self.is_listening = True  # Ensure we resume listening even if there's an error

            time.sleep(0.1)

    def calibrate_silence_threshold(self):
        """Calibrate the silence threshold based on ambient noise"""
        print("Calibrating microphone... Please remain quiet.")
        start_time = time.time()
        
        while time.time() - start_time < self.calibration_duration:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                audio_level = np.abs(audio_data).mean()
                self.calibration_samples.append(audio_level)
        
        if self.calibration_samples:
            # Set threshold slightly above the average ambient noise
            self.silence_threshold = np.mean(self.calibration_samples) * 1.1
            print(f"Silence threshold calibrated to: {self.silence_threshold:.6f}")

    def start(self):
        """Start the voice transcription system"""
        if not self.is_running:
            self.is_running = True
            self.terminate_event.clear()

            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.blocksize
            )
            self.stream.start()

            # Calibrate silence threshold
            self.calibrate_silence_threshold()

            # Start processing thread
            self.process_thread = threading.Thread(
                target=self.process_audio,
                daemon=True
            )
            self.process_thread.start()

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

    def process_callback(self, text):
        """Enhanced callback to handle speaking state"""
        self.is_speaking = True
        response = self.callback(text)
        self.is_speaking = False
        return response

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

