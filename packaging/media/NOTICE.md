# Media libraries and corresponding source

Freedive Color Studio's own source files retain their MIT license. The combined
standalone distribution includes GPL codec components and is distributed under
GPLv3, without warranty.
PyAV itself is BSD licensed. The linked FFmpeg community build includes x264
and x265; their licenses remain applicable even though the vendor's FFmpeg patch
removes those libraries from its configure-time GPL guard lists. The vendor's
reported LGPL library string does not replace those obligations.

macOS uses PyAV 15.1.0 / FFmpeg 7.1.1, producer tag `7.1.1-6`, commit
`f665a9654da201e8d5c9ec2b7597ae90fbdedba5`. Windows uses PyAV 16.1.0 /
FFmpeg 8.0.1, producer tag `8.0.1-3`, commit
`9e63121cf6ef9bc1744956c7a4cb772589c80806`. The Windows media build contains
no SRT, GMP or old OpenSSL libraries. Original build scripts, full repositories,
FFmpeg patches and dependency archives are retained without editing.
Every collected source archive has a pinned SHA-256 record.
The archive also includes matching Qt, PySide and Shiboken 6.8.3 sources and
their LGPL/GPL notices, and the native image-only OpenCV source/build recipe.

The separate `Freedive_Color_Studio_Corresponding_Source.zip` (macOS) and
`Freedive_Color_Studio_Windows_Corresponding_Source.zip` are provided beside
their application distributions. They contain source archives, licenses, exact
upstream recipes/patches, native library inventory and the application source
snapshot. Its adjacent JSON record gives the archive SHA-256. Rebuild
instructions are in `application/docs/CORRESPONDING_SOURCE.md` inside that ZIP.
No source downloads are performed by the application.

Both platforms use genuinely compiled IPP-free OpenCV. The Windows extension
also omits its unused FFmpeg plugin and uses a minimal native loader; no opaque
Intel IPP binary is part of that package. Windows source records include exact
GCC/MSYS2 runtime sources and the byte-matched winpthread package provenance,
plus target Python/NumPy interfaces and the Zig SDK's statically linked runtime
sources. Microsoft runtime components retain their separate Microsoft terms
and notices; they are not represented as GPL source files.

The generated source marker states its platform and exact Windows payload hash.
Windows binaries have been cross-compiled and statically audited. Actual
Windows execution has not been performed; source availability and dependency
inspection do not constitute a Windows runtime test.

Primary provenance:

- [PyAV 15.1.0 wheel vendor selection](https://github.com/PyAV-Org/PyAV/blob/v15.1.0/scripts/ffmpeg-7.1.json)
- [Exact vendor build recipe](https://github.com/PyAV-Org/pyav-ffmpeg/blob/7.1.1-6/scripts/build-ffmpeg.py)
- [Exact upstream FFmpeg patch](https://github.com/PyAV-Org/pyav-ffmpeg/blob/7.1.1-6/patches/ffmpeg.patch)
- [Windows producer and patch](https://github.com/PyAV-Org/pyav-ffmpeg/tree/8.0.1-3)
- [FFmpeg 7.1.1 licensing](https://github.com/FFmpeg/FFmpeg/blob/n7.1.1/LICENSE.md)
- [Qt 6.8.3 source archives](https://download.qt.io/archive/qt/6.8/6.8.3/submodules/)
- [PySide/Shiboken 6.8.3 sources](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.8.3-src/)
