# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect pyproj data files (includes proj.db)
pyproj_datas = collect_data_files('pyproj')

# Collect all submodules from eo_flood_ops
eo_flood_ops_hiddenimports = collect_submodules('eo_flood_ops')

a = Analysis(
    ['scripts\\run_model.py'],
    pathex=['src'],  # Add src to path so it can find eo_flood_ops
    datas=pyproj_datas,
    hiddenimports=[
        'rasterio.sample',
        'rasterio.vrt',
        'rasterio._shim',
        'rasterio.crs',
        'rasterio._features',
        'rasterio._env',
        'pyogrio._geometry',
        'pyogrio._io',
        'pandas._libs.window.aggregations',
        'pyproj',
        'pyproj.datadir',
        'eo_flood_ops',
        'eo_flood_ops.general_utils',
        'eo_flood_ops.thresholding_model',
        'eo_flood_ops.model_utils',
        'eo_flood_ops.manifold_model',
        'numpy.testing',
    ] + eo_flood_ops_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_hooks\\proj_lib_hook.py'],
    excludes=['tkinter', 'matplotlib.tests'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='eo_flood_ops_model',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
