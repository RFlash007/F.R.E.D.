import subprocess
import os
import winsound

def piper_speak(text, model_path=".\\jarvis-medium.onnx", output_file="test1.wav"):
    cmd = [".\\piper.exe", "-m", model_path, "-f", output_file]
    subprocess.run(cmd, input=text.encode(), check=True)
    winsound.PlaySound(output_file, winsound.SND_FILENAME)



