# Offline skin candidate evaluation

These are developer evaluation tools. The desktop still uses the bundled
SelfieMulticlass model; neither candidate is selected by an environment variable,
downloaded at startup, or silently used as a fallback. A numerically correct ONNX
conversion does not establish underwater segmentation accuracy.

## Fixed candidates and provenance

| Name | Official source | Checkpoint |
| --- | --- | --- |
| `selfie-v2` | Existing bundled SelfieMulticlass with shared local view consensus | Bundled, SHA recorded in core |
| `schp-atr` | [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing), commit `eb84c432cc697f494d99662a05f2335eb2f26095` | Official linked `exp-schp-201908301523-atr.pth` |
| `fcn-skin` | [SemanticSegmentation](https://github.com/WillBrennan/SemanticSegmentation), commit `7646661aa0e05f5c5b087f2aeda3faff42c34e00` | Repository Git-LFS `pretrained/model_segmentation_skin_30.pth` |

FCN checkpoint SHA-256 is pinned by the official Git-LFS pointer:
`a2739556d28995ad1ceb42dca83f292a45f0ecbae3f602f0d1b70f742b7821eb`.
SCHP checkpoint SHA-256 was computed from the repository-linked official Drive
download and is now pinned for every repeat download:
`e9d7c91ce3b4e7133df56b599fc817b533e3439c5e8d282a59126d2fda339a2a`.
Source-archive and converted model hashes live in `tools/skin_candidates.py` and
the conversion JSON. The loader requires the code-pinned ONNX hash, matching
manifest provenance, and successful recorded numerical parity.

Both upstream repositories provide MIT notices, retained with converted artifacts.
The original SCHP checkpoint is published through the repository's official Drive
link; this is recorded separately from the source-code license rather than claimed
to have a separately verified weight-specific license. Before public distribution,
retain all applicable source/checkpoint notices and resolve any contradictory
terms. Public model availability does not authorize publishing user media.

## Conversion

Create a separate environment; do not synchronize or replace the app's custom
native OpenCV/ONNX Runtime environment:

```sh
.venv/bin/python -m venv build/skin-candidates/venv
UV_CACHE_DIR=build/skin-candidates/uv-cache uv pip install \
  --python build/skin-candidates/venv/bin/python \
  torch==2.7.1 torchvision==0.22.1 onnx==1.17.0 onnxruntime==1.23.2 \
  numpy==2.2.6 opencv-python-headless==4.14.0.94

PYTHONPATH=src .venv/bin/python -m tools.convert_skin_candidates fetch fcn-skin
PYTHONPATH=src .venv/bin/python -m tools.convert_skin_candidates fetch schp-atr
PYTHONPATH=src build/skin-candidates/venv/bin/python \
  -m tools.convert_skin_candidates convert fcn-skin
PYTHONPATH=src build/skin-candidates/venv/bin/python \
  -m tools.convert_skin_candidates convert schp-atr
```

Only `fetch` accesses the network. `convert` verifies the archive, every extracted
source file and checkpoint hash before reading the checkpoint with
`torch.load(weights_only=True)`. It disables implicit torchvision downloads, loads
the complete original checkpoint, runs CPU inference, exports fixed 512×512 ONNX
opset17, and compares three deterministic synthetic inputs against the original
  source architecture and one licensed public-domain freediver image. No training
  occurs. Required maximum/mean logit error is
0.002/0.0001, with at least0.999 pixel decision agreement.

Checkpoint downloads use eight bounded anonymous HTTPS ranges, checking exact
Content-Range/length and the full pinned SHA before atomic publication. The SCHP
URL is the observed official repository-linked Drive download, after its standard
large-file confirmation. No account, authentication, existing cookies or cookie
file is used; the tool never attempts to solve a CAPTCHA or bypass an access gate.

For SCHP, the obsolete native `InPlaceABNSync` implementation is represented using
its exact upstream CPU inference formula, including `abs(weight)+epsilon` and the
configured activation. This is not an assertion that the old native extension was
compiled on the current OS. PSP adaptive pooling is replaced by mathematically
equivalent fixed bins for ONNX; original-vs-rewritten logit errors are recorded.
The decoder is upsampled with upstream `align_corners=True` before inverse affine.

To check reproducibility, run each conversion in a fresh process with an additional
`--output-dir build/skin-candidates/onnx-repeat`, then compare both ONNX SHA-256
values. Manifests include measured time, so compare the ONNX files rather than the
entire JSON file. PyTorch/torchvision/ONNX are conversion dependencies only.

The current local converter reuses a byte copy of the already audited image-only
OpenCV4.14.0.94 package and its matching metadata, retaining the main environment
unchanged. Its native hash is checked against
`build/skin-candidates/native-copy-provenance.json` and included in conversion
manifests. This avoids re-downloading the larger stock OpenCV wheel; the same API
and pinned package version are used. This copy is not included in application
packages by these tools.

## Input, output and evaluation API

```python
from tools.skin_candidates import make_predictor

predict = make_predictor("schp-atr", "build/skin-candidates/onnx")
prediction = predict(rgb_float32_0_to_1, person_probability)
```

The returned `SkinPrediction` uses the same core orientation, contextual crops,
consensus, confidence gates and final mask builder as `selfie-v2`. Only the model's
documented input conversion and semantic class mapping differ. It includes model
ID, algorithm version, checkpoint/ONNX hashes and source commit. Evaluation tools
must select this callable explicitly in their own process; app state and installed
MCP remain unchanged.

- SCHP-ATR uses BGR ImageNet normalization and the upstream centered square affine.
  Classes12–15 are body skin,11 face,2 hair,4–10/17 clothing including belt/shoes,
  and1/3/16 accessories. Logits are inverse-warped before softmax. The model has no
  explicit bare-torso class and its sunglasses class does not prove correct diving
  mask exclusion.
- FCN uses RGB ImageNet normalization and a 512-square analysis letterbox. The
  source demo's truncation to multiples32 is replaced by padding so all source
  edges are retained. Its one logit uses sigmoid, not softmax. It supplies binary
  skin/background evidence and **cannot claim separate hair/clothing/gear classes**.
  Zero maps for those absent classes mean unavailable classification, not proven
  absence of those objects.

Neither model was developed for underwater skin. The FCN author reports only150
annotated COCO images. Model scores remain uncalibrated; sparse ROI measurements
and dense independent holdout annotations determine whether a candidate improves
the app. Failed exclusions must not be repaired by blind mask union or looser
acceptance thresholds. If no candidate passes, retain the prior default and report
the failed acceptance honestly.

## Executed conversion checks

On macOS arm64, both candidates were converted twice in independent fresh CPU
processes. Each pair produced byte-identical ONNX files and passed all four
numerical fixtures: black, cyan, fixed-seed noise and the licensed public-domain
freediver photo already shipped in the test fixtures.

| Candidate | ONNX bytes | Worst maximum logit error | Pixel decisions matching reference |
| --- | ---: | ---: | ---: |
| `fcn-skin` | 207,611,264 | 0.000022889 | 100% on each parity fixture |
| `schp-atr` | 267,693,588 | 0.000159264 | 100% on each parity fixture |

Pinned ONNX SHA-256:

- FCN: `adb8680d68faea1fda4729bf1c4335e99c0442ed0b8336e3e5369375d59b0ed9`.
- SCHP: `fc33b8a8709427cce17434f1c99137bc7595ac6b4e6e2184fa08e5dd02f8b868`.

The SCHP reference uses the pinned original network/checkpoint with a pure PyTorch
implementation of the literal upstream CPU normalization operations; the obsolete
native extension was **not compiled or executed**. Parity separately measures the
original adaptive pooling against the fixed-bin export rewrite, then compares
ONNX against the CPU reference. Thus these results establish reproducible graph
conversion against that documented CPU reference, not execution parity with the
old compiled extension or underwater segmentation ground truth.

The main native CPU ONNX Runtime1.23.2 loaded and executed each wrapper with the
expected `[1,1,512,512]` / `[1,18,512,512]` outputs and finite six-class maps.
`artifacts/skin-v2-candidates/conversion-reproducibility.json` aggregates local
software checks; individual manifests/notices are in `build/skin-candidates/onnx`.
The isolated source folder retains each original MIT notice. Candidate inference
on Windows and on minimum supported OS versions is not established by these
macOS checks. Neither candidate weight was added to the desktop bundle.

## Fixed local comparison

Supply explicit local media and frozen sample manifests; the tools contain no
private media paths or sample coordinates. Each run reserves a fresh output
directory and writes only scalar reports/paths. Example:

```sh
.venv/bin/python -m tools.evaluate_skin_candidate \
  --candidate fcn-skin --model-dir build/skin-candidates/onnx \
  --photo /local/input/photo.png --photo-samples /local/labels/photo.json \
  --video /local/input/dive.mp4 --video-samples /local/labels/video.json \
  --output-directory artifacts/new-fcn-run --require-pass
```

Repeat with `selfie-v2` and `schp-atr` into separate fresh directories. Use
`--static-only` for an initial bounded diagnosis, then run the full command to
include source-timestamp optical-flow tracking. A static-only report cannot be
used to select the product default. `--require-pass` returns status2 when the
quality criteria fail; a completed process alone is not acceptance.

Compare complete runs using `tools.compare_skin_runs`, passing the same frozen
photo/video manifests and the v1 baseline reports:

```sh
.venv/bin/python -m tools.compare_skin_runs \
  --baseline-photo /local/baseline/photo.json \
  --baseline-video /local/baseline/video.json \
  --run selfie-v2 artifacts/new-selfie-run/photo.json artifacts/new-selfie-run/video.json \
  --run fcn-skin artifacts/new-fcn-run/photo.json artifacts/new-fcn-run/video.json \
  --run schp-atr artifacts/new-schp-run/photo.json artifacts/new-schp-run/video.json \
  --output artifacts/new-comparison.json
```

The comparison applies the fixed gate and verifies matching inputs, sample
manifest hashes and complete tracked coverage before ranking candidates. Reports
derived from private media remain in ignored local artifacts. Public documentation
may include software/model conversion and licensed-public-fixture evidence, but
must not publish private sample coordinates, ROI measurements, media hashes or
derived acceptance counts without separate authorization.
