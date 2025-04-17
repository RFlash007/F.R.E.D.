# F.R.E.D. Desktop Application Installer

## Installation Instructions

1. **Prerequisites**:
   - Windows 10 or higher
   - At least 4GB of RAM
   - 2GB of free disk space
   - Webcam (required for vision features)
   - Microphone (required for voice commands)
   - Python 3.10 or higher (only needed for installation)

2. **Installation Steps**:
   - Run `Install_FRED.bat` by double-clicking it
   - When prompted, allow the installer to run with administrator privileges
   - The installer will set up FRED and create a desktop shortcut
   - Once installation is complete, you can start FRED from the desktop shortcut

3. **First-Time Setup**:
   - When you first run FRED, it may need to download additional model files
   - Allow FRED to access your camera and microphone when prompted
   - Some models may take time to load on the first run

## System Requirements

### Minimum Requirements
- Operating System: Windows 10 (64-bit)
- Processor: Intel Core i5 or equivalent
- Memory: 4GB RAM
- Graphics: Any dedicated GPU with CUDA support (for optimal performance)
- Storage: 2GB available space
- Camera: Any compatible webcam
- Audio: Microphone and speakers

### Recommended Requirements
- Operating System: Windows 10/11 (64-bit)
- Processor: Intel Core i7/AMD Ryzen 7 or better
- Memory: 8GB RAM or more
- Graphics: NVIDIA GPU with at least 4GB VRAM
- Storage: 5GB available SSD space
- Camera: HD webcam (1080p)
- Audio: Quality microphone and speakers

## Troubleshooting

If you encounter any issues:

1. **Application fails to start**:
   - Make sure you have all the required Microsoft Visual C++ Redistributables installed
   - Try running the application as administrator

2. **Camera or microphone not detected**:
   - Ensure your devices are connected and working properly
   - Check Windows privacy settings to allow applications to access your camera and microphone

3. **Slow performance**:
   - Check if your GPU is being utilized (FRED will display this information on startup)
   - Close other resource-intensive applications

## Uninstallation

To uninstall FRED:
1. Delete the FRED folder from where you installed it
2. Delete the desktop shortcut
3. Optionally, remove any data in the following folders:
   - cache
   - voice_cache
   - conversation_history 