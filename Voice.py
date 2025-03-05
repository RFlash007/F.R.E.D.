import subprocess
import os
import winsound
import threading
import time

# Add this at the top of Voice.py
piper_lock = threading.Lock()

# Define absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_EXE = os.path.join(SCRIPT_DIR, "piper.exe")
MODEL_PATH = os.path.join(SCRIPT_DIR, "jarvis-high.onnx")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "test1.wav")

def cleanup_output_file(output_file, delay=1):
    """Clean up the generated audio file after playing."""
    time.sleep(delay)  # Wait for the file to be released
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        print(f"Warning: Could not cleanup output file: {e}")

def piper_speak(text, model_path=None, output_file=None):
    if not text.strip():
        print("Warning: Empty text provided")
        return False
        
    with piper_lock:
        try:
            # Use provided paths or defaults
            model = model_path if model_path else MODEL_PATH
            output = output_file if output_file else OUTPUT_FILE
            
            # Verify files exist
            if not os.path.exists(PIPER_EXE):
                print(f"Error: Piper executable not found at {PIPER_EXE}")
                return False
                
            if not os.path.exists(model):
                print(f"Error: Model file not found at {model}")
                return False
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output), exist_ok=True)
            
            # Clean up any existing output file
            cleanup_output_file(output, delay=0)
            
            cmd = [PIPER_EXE, "-m", model, "-f", output]
            
            # Run piper with error handling and timeout
            result = subprocess.run(
                cmd,
                input=text.encode(),
                check=True,
                capture_output=True,
                timeout=30  # Add timeout to prevent hanging
            )
            
            # Play the generated audio
            if os.path.exists(output):
                winsound.PlaySound(output, winsound.SND_FILENAME)
                # Start cleanup in a separate thread
                threading.Thread(target=cleanup_output_file, args=(output,), daemon=True).start()
                return True
            else:
                print(f"Error: Output file not generated at {output}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Error: TTS process timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Piper execution failed: {e}")
            print(f"Stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"Error in piper_speak: {str(e)}")
            return False



