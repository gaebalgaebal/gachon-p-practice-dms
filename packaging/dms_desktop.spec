# PyInstaller specification for the offline Windows desktop application.

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).parent
datas = [
    (str(root / "models" / "face_landmarker.task"), "models"),
    (str(root / "models" / "open_closed_eye.onnx"), "models"),
    (str(root / "configs" / "default.json"), "configs"),
    (str(root / "configs" / "model_manifest.json"), "configs"),
    (str(root / "docs" / "THIRD_PARTY_NOTICES.md"), "."),
]
binaries = collect_dynamic_libs("mediapipe") + collect_dynamic_libs("openvino")
datas += collect_data_files("mediapipe") + collect_data_files("openvino")
hiddenimports = ["dms.eye", "openvino"]

a = Analysis(
    [str(root / "src" / "dms_desktop_main.py")],
    pathex=[str(root / "src")], binaries=binaries, datas=datas,
    hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
    excludes=["torch", "torchvision", "ultralytics", "pandas", "scipy"],
    noarchive=False,
)
a.datas = [item for item in a.datas if "matplotlib/mpl-data/sample_data" not in item[0].replace("\\", "/")]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="GachonDMS",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="GachonDMS")
