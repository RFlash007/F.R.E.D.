import os
import sys
import winshell
import win32com.client
import ctypes
import shutil
from pathlib import Path

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        'cache',
        'voice_cache',
        'conversation_history',
        'Tasks',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")

def create_desktop_shortcut():
    """Create a desktop shortcut for FRED"""
    try:
        # Get the desktop path
        desktop = winshell.desktop()
        
        # Create a shortcut
        path = os.path.join(desktop, "FRED.lnk")
        target = os.path.join(os.getcwd(), "FRED.exe")
        icon = os.path.join(os.getcwd(), "assets", "fred_icon.ico")
        
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.IconLocation = icon if os.path.exists(icon) else target
        shortcut.save()
        
        print(f"✓ Created desktop shortcut at: {path}")
        return True
    except Exception as e:
        print(f"× Failed to create desktop shortcut: {str(e)}")
        return False

def check_for_gpu():
    """Check if a CUDA-compatible GPU is available"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if has_cuda else 0
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "None"
        
        if has_cuda:
            print(f"✓ CUDA-compatible GPU detected: {device_name}")
        else:
            print("! No CUDA-compatible GPU detected. F.R.E.D. will run in CPU mode, which may be slower.")
    except Exception as e:
        print(f"! Error checking GPU availability: {str(e)}")

def is_admin():
    """Check if the script is running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def main():
    print("=== F.R.E.D. Post-Installation ===")
    
    # Ensure required directories exist
    ensure_directories()
    
    # Create desktop shortcut
    shortcut_created = create_desktop_shortcut()
    
    # Check for GPU
    check_for_gpu()
    
    # Installation complete
    print("\nF.R.E.D. installation completed successfully!")
    print("To start F.R.E.D., use the desktop shortcut or run FRED.exe from the installation directory.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    if is_admin():
        main()
    else:
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1) 