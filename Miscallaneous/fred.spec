# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Chat.py'],  # Main entry point
    pathex=[],
    binaries=[],
    datas=[
        # Include model files
        ('*.pt', '.'),
        ('models/', 'models/'),
        ('*.onnx', '.'),
        ('*.onnx.json', '.'),
        # Include library files
        ('espeak-ng.dll', '.'),
        ('onnxruntime.dll', '.'),
        ('onnxruntime_providers_shared.dll', '.'),
        ('piper_phonemize.dll', '.'),
        # Include directories
        ('espeak-ng-data/', 'espeak-ng-data/'),
        ('piper/', 'piper/'),
        ('ffmpeg/', 'ffmpeg/'),
        # Include data files
        ('data/', 'data/'),
        # Include cache directories (but not the files)
        ('cache/', 'cache/'),
        ('voice_cache/', 'voice_cache/'),
        # Include memory files
        ('Semantic.json', '.'),
        ('Episodic.json', '.'),
        ('Dreaming.json', '.'),
    ],
    hiddenimports=[
        'ollama',
        'duckduckgo_search',
        'PIL',
        'PIL._tkinter_finder',
        'Dreaming',
        'MorningReport',
        'Tools',
        'Voice',
        'Semantic',
        'Procedural',
        'Episodic',
        'Vision',
        'ChatUI',
        'shared_resources',
        'EmotionIntegration',
        'EmotionDetector',
        'FaceRecognition',
        'FaceDB',
        'Transcribe',
        'Task',
        'Research',
        'torch',
        'torchvision',
        'transformers',
        'ultralytics',
        'numpy',
        'opencv-cv2',
        'face_recognition',
        'dlib',
        'memory_format_utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FRED',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for a GUI-only application, but True is useful for debugging
    icon='assets/fred_icon.ico',  # Assuming you have this icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FRED',
) 