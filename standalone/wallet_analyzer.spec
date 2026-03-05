# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Bitcoin Wallet Risk Analyzer
======================================================
Build with:
    pip install pyinstaller
    pyinstaller wallet_analyzer.spec

Output: dist/wallet_analyzer  (single exe, ~1.5-3 GB)
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

# Collect all torch_geometric submodules (many are dynamically imported)
torch_geo_hiddenimports = collect_submodules('torch_geometric')

# Collect torch submodules needed at runtime (exclude dynamo/inductor - not needed for inference)
torch_hiddenimports = [
    m for m in collect_submodules('torch')
    if not any(x in m for x in ['_dynamo', '_inductor', '_numpy', 'torch.testing', 'torch.utils.tensorboard'])
]

# Collect data files
torch_geo_datas = collect_data_files('torch_geometric', include_py_files=True)
torch_datas = collect_data_files('torch', include_py_files=True)

block_cipher = None

a = Analysis(
    ['wallet_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Bundle the trained model
        (os.path.join('..', 'outputs', 'gnn_model.pt'), '.'),
    ] + torch_geo_datas + torch_datas,
    hiddenimports=[
        # torch_geometric core
        'torch_geometric',
        'torch_geometric.nn',
        'torch_geometric.nn.conv',
        'torch_geometric.nn.conv.gatv2_conv',
        'torch_geometric.nn.norm',
        'torch_geometric.nn.norm.layer_norm',
        'torch_geometric.data',
        'torch_geometric.data.data',
        'torch_geometric.data.batch',
        'torch_geometric.utils',
        'torch_geometric.typing',
        # torch_scatter / torch_sparse (backends for PyG)
        'torch_scatter',
        'torch_sparse',
        'torch_cluster',
        'torch_spline_conv',
        # Standard libs
        'numpy',
        'requests',
        'matplotlib',
        'matplotlib.backends.backend_agg',
    ] + torch_geo_hiddenimports + torch_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages not needed at runtime
        'sklearn',
        'scikit-learn',
        'pandas',
        'scipy',
        'jupyter',
        'notebook',
        'ipykernel',
        'ipython',
        'fastapi',
        'uvicorn',
        'starlette',
        'pydantic',
        'tensorboard',
        'cv2',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'wx',
    ],
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
    name='wallet_analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='wallet_analyzer',
)
