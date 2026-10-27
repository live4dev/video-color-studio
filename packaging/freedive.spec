# PyInstaller standalone directory: shared libraries remain replaceable for LGPL compliance.
import sys
from pathlib import Path

root = Path(SPECPATH).parent
data = [(str(root / filename), "licenses") for filename in
        ("LICENSE", "THIRD_PARTY_LICENSES.md", "README.md")]
data.append((str(root / "build" / "licenses"), "licenses/dependencies"))
analysis = Analysis(
    [str(root / "packaging" / "launcher.py")],
    pathex=[str(root / "src")],
    binaries=[], datas=data, hiddenimports=[],
    excludes=["tkinter", "matplotlib", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtQml",
              "PySide6.QtQuick", "PySide6.QtNetwork", "PySide6.QtTest"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True,
    name="Freedive Color Studio", debug=False, strip=False, upx=False,
    console=False, argv_emulation=False,
)
collected = COLLECT(executable, analysis.binaries, analysis.datas,
                    strip=False, upx=False, name="Freedive Color Studio")
if sys.platform == "darwin":
    bundle = BUNDLE(
        collected, name="Freedive Color Studio.app",
        bundle_identifier="dev.live4.freedive-color-studio",
        info_plist={
            "CFBundleShortVersionString": "0.1.0", "CFBundleVersion": "0.1.0",
            "LSMinimumSystemVersion": "13.0", "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.photography",
        },
    )
