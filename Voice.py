import subprocess
import os
import winsound
import threading

# Add this at the top of Voice.py
piper_lock = threading.Lock()

# Define absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_EXE = os.path.join(SCRIPT_DIR, "piper.exe")
MODEL_PATH = os.path.join(SCRIPT_DIR, "en_US-danny.low.onnx")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "test1.wav")

def piper_speak(text, model_path=None, output_file=None):
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
            
            cmd = [PIPER_EXE, "-m", model, "-f", output]
            
            # Run piper with error handling
            result = subprocess.run(
                cmd,
                input=text.encode(),
                check=True,
                capture_output=True
            )
            
            # Play the generated audio
            if os.path.exists(output):
                winsound.PlaySound(output, winsound.SND_FILENAME)
                return True
            else:
                print(f"Error: Output file not generated at {output}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Piper execution failed: {e}")
            print(f"Stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"Error in piper_speak: {str(e)}")
            return False



