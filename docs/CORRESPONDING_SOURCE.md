# Corresponding-source release recipe

This developer tool collects sources for redistribution; the desktop app does
not import or execute it. Run with the locked project Python environment:

```sh
.venv/bin/python tools/collect_media_sources.py
```

The collector verifies every archive against `packaging/media/sources-lock.json`.
It saves downloads in `build/media-sources/archives`, full extracted license
texts in `build/media-sources/notices`, and native inventory/source records in
`build/media-sources/source-manifest.json`. An incomplete download or checksum
mismatch prevents creation of a new release archive. Archives are never executed.
License extraction reads only regular members into generated safe filenames.

After the final app sources and macOS bundle are ready, refresh the source
snapshot using the cache and audit the exact Qt frameworks in that bundle:

```sh
.venv/bin/python tools/collect_media_sources.py --offline --bundle "dist/Freedive Color Studio.app"
```

Output is `dist/Freedive_Color_Studio_Corresponding_Source.zip` plus its adjacent
JSON checksum record. Distribute both beside the app. It includes the complete
application source snapshot, dependency source archives, full original vendor
repositories/build workflows, patches, notices and native inventory. It excludes
the environment, Git internals and user media outside the source/test tree.

For the app, create Python 3.12+ environment, install the lock, build/install
image-only OpenCV using `docs/NATIVE_OPENCV.md`, then run the supplied platform
packaging script. Qt/PySide 6.8.3 archives retain their own build instructions;
PySide's `pyside-setup` source includes Shiboken. The app keeps Qt dynamically
loaded so its libraries remain replaceable.

For the exact media libraries, extract the pinned `pyav-ffmpeg` repository and
the original dependency archives. Follow that repository's platform build
workflow and run its `scripts/build-ffmpeg.py <destination> --community` recipe.
The full repository includes `cibuildpkg.py` and the original patch files. The
upstream recipe downloads sources during a developer rebuild; the collector
provides verified local copies so download steps can use those cached archives.
Build PyAV 15.1.0 against the resulting headers/libraries using its included
source/build instructions. Toolchains are recorded by the upstream workflows;
this is corresponding source, not a claim of identical binary hashes after a
different compiler build.

Keep the vendor patch intact for provenance. It moves x264/x265 out of GPL guard
lists without changing their actual licenses; the combined release's GPLv3
notice and full corresponding sources remain necessary. Full original license
texts are available in `notices/` and inside each source archive.

The macOS source set targets the tested PyAV 15.1.0 / FFmpeg 7.1.1 community
wheel. Windows uses the separate official PyAV 16.1.0 / FFmpeg 8.0.1 producer
without SRT, GMP or old OpenSSL, and a genuine IPP-free OpenCV extension built
from the same 4.14.0 source. Its target Python 3.13.17 interface headers/import
library and NumPy 2.5.4 headers are pinned. The Zig 0.14.1 SDK preserves the
actual libc++/MinGW/compiler-runtime sources compiled statically into cv2.

For the Windows source archive, first build and audit the exact portable stage,
then run:

```sh
.venv/bin/python tools/collect_media_sources.py --offline \
  --windows-bundle "build/windows-shipping/Freedive Color Studio" \
  --work-dir build/windows-media-sources \
  --output dist/Freedive_Color_Studio_Windows_Corresponding_Source.zip
```

Its adjacent JSON marker explicitly contains `platforms: ["windows-x64"]` and
`windows_payload_sha256`. The collector recomputes the complete payload hash,
checks the staged clean OpenCV files against their native build manifest and
rejects the stock IPP-bearing package. The final ZIP builder requires the same
payload marker and checks every staged application source/asset against the
source archive. A subsequent app or loader change requires a new snapshot.

Windows runtime sources include exact MSYS2 GCC 15.2.0-8 (with the GCC runtime
exception), libiconv 1.18-1 and zlib 1.3.1-1 packages, including original source
archives, patches and PKGBUILD recipes. Winpthread provenance identifies the
exact 13.0.0.r354.g40ab95d18-1 package by reproducing the vendor's stripping and
matching the entire shipped DLL. Its original Git repository, commit, recipe
and MIT/BSD notices are retained. Microsoft runtime notices and applicable
runtime terms remain separate; they do not turn those system/compiler binaries
into GPL source code.

The source manifest distinguishes bundled source bytes from `source_access`
records. Exact locked NumPy 2.5.4, tifffile 2026.10.15 and PyInstaller 6.22.3
sdists are included. Pillow 13.0.0's exact published sdist URL and SHA-256, plus
its immutable producer and retained codec/font dependencies, are explicitly
recorded. Cached codec sources are included; the remaining unmodified permissive
components have exact upstream release/commit URLs and original build recipes.
This third-party source-access list is part of the source offer, not a claim
that every linked archive has also been copied into the ZIP.

Both NumPy wheels identify scipy-openblas64 0.3.34.106.0. Its full producer
commit is `78fc0eaf6de71d92fd98f74f7e5bfa356a2f83e3`; its exact OpenBLAS/LAPACK
source is `446c436e10450c348169808f1a6b3fae0925c9f7`. That producer and its
original notices/scripts are bundled. Full source access is pinned to the
immutable upstream commit. The producer selects macOS gfortran release
11.3.0-2; complete GNU runtime/quadmath source access is identified at GCC
commit `2d280e7eafc086e9df85f50ed1a6526d6a3a204d`, with the original GCC runtime
exception and LGPL notices retained. These source-access records explicitly
identify externally hosted source; they do not invent unavailable source or
promise identical binaries from a different compiler.

The unmodified Python interpreter and operating-system/compiler runtimes retain
their own notices and source/redistribution terms. macOS system frameworks and
Windows system DLLs are platform/system components, outside the application's
own source. Microsoft runtime binaries are supplied unchanged by the locked
Python/Qt/NumPy distributions, with their applicable separate notices/terms.

The source archive includes the original platform-specific vendor repositories
and native compiler-driver source. Native Windows CMake/Ninja invokes real PE
compiler drivers rather than relying on direct execution of command files.
Windows binaries have been compiled and inspected as PE; actual Windows
execution remains **NOT RUN**. The collector fails on a newly bundled Qt module
without matching sources or changed/unverified payload, and never substitutes a
different source version silently.
