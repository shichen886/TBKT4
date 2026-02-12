# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app.py', '.'),
        ('model_sakt.py', '.'),
        ('model_tsakt.py', '.'),
        ('data', 'data'),
        ('save', 'save'),
    ],
    hiddenimports=[
        'streamlit',
        'streamlit.cli',
        'streamlit.runtime',
        'streamlit.web',
        'streamlit.components',
        'pandas',
        'numpy',
        'torch',
        'plotly',
        'cv2',
        'pytesseract',
        'PIL',
        'sklearn',
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

executable = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='知识追踪系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    executable,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='知识追踪系统',
)
