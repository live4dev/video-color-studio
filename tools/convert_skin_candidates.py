"""Reproducible developer conversion of two explicitly selected official skin candidates.

Run in the isolated environment documented in docs/SKIN_CANDIDATES.md. No part of
this module is imported by the desktop. Downloading requires the explicit fetch
subcommand; convert and verify operate offline on SHA-pinned local inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import types
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.skin_candidates import UPSTREAM, prepare_candidate_input  # noqa: E402

VERSIONS = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "onnx": "1.17.0",
    "onnxruntime": "1.23.2",
    "numpy": "2.2.6",
    "opencv-python-headless": "4.14.0.94",
}
CHECKPOINT_NAMES = {
    "fcn-skin": "model_segmentation_skin_30.pth",
    "schp-atr": "exp-schp-201908301523-atr.pth",
}


def download_public_checkpoint(name: str, target: Path) -> None:
    """Bounded public HTTP ranges; exact range/length and full SHA before publication."""
    source = UPSTREAM[name]
    if source["checkpoint_sha256"] is None:
        raise ValueError("Official checkpoint SHA must be pinned before fetching")
    size, chunk_size = source["checkpoint_bytes"], 8 * 1024 * 1024
    parts_dir = target.parent / (
        "fcn-download-parts" if name == "fcn-skin" else "schp-download-parts"
    )
    parts_dir.mkdir(exist_ok=True)
    count = (size + chunk_size - 1) // chunk_size

    def part(index):
        start, stop = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        path = parts_dir / f"{index:03d}.part"
        if path.is_file() and path.stat().st_size == stop - start + 1:
            return path
        temporary = path.with_suffix(".downloading")
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    source.get("checkpoint_download_url", source["checkpoint_url"]),
                    headers={"Range": f"bytes={start}-{stop}"},
                )
                with urllib.request.urlopen(request, timeout=240) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{size}"
                    ):
                        raise ValueError("Official checkpoint server did not honor exact range")
                    with temporary.open("wb") as out:
                        shutil.copyfileobj(response, out, 1024 * 1024)
                if temporary.stat().st_size != stop - start + 1:
                    raise ValueError("Official checkpoint range is incomplete")
                temporary.replace(path)
                return path
            except (OSError, ValueError):
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=8) as executor:
        parts = list(executor.map(part, range(count)))
    temporary = target.with_suffix(".part")
    try:
        with temporary.open("wb") as out:
            for path in parts:
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, out, 1024 * 1024)
        if temporary.stat().st_size != size or sha256(temporary) != source["checkpoint_sha256"]:
            for path in parts:
                path.unlink(missing_ok=True)
            raise ValueError("Full official checkpoint SHA mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download_public_fcn(target: Path) -> None:
    """Compatibility helper for the fixed FCN artifact and integrity fixtures."""
    download_public_checkpoint("fcn-skin", target)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_directory(base: Path, name: str) -> Path:
    source = UPSTREAM[name]
    return (
        base
        / "upstream"
        / ("fcn" if name == "fcn-skin" else "schp")
        / (source["repository"].rsplit("/", 1)[1] + "-" + source["commit"])
    )


def verify_inputs(base: Path, name: str) -> tuple[Path, Path]:
    source = UPSTREAM[name]
    source_name = "fcn" if name == "fcn-skin" else "schp"
    archive = base / "upstream" / f"{source_name}-{source['commit']}.zip"
    if sha256(archive) != source["source_zip_sha256"]:
        raise ValueError("Official source archive SHA mismatch")
    directory = source_directory(base, name)
    if not directory.exists():
        with zipfile.ZipFile(archive) as files:
            for entry in files.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source archive")
            files.extractall(directory.parent)
    # Verify extracted source byte-for-byte, including the upstream MIT notice.
    with zipfile.ZipFile(archive) as files:
        for entry in files.infolist():
            if not entry.is_dir():
                path = directory.parent / entry.filename
                if not path.is_file() or path.read_bytes() != files.read(entry):
                    raise ValueError(f"Extracted source changed: {entry.filename}")
    checkpoint = base / "upstream" / CHECKPOINT_NAMES[name]
    expected = source["checkpoint_sha256"]
    if expected is None or sha256(checkpoint) != expected:
        raise ValueError("Official checkpoint SHA mismatch or not yet independently pinned")
    return directory, checkpoint


def fetch(base: Path, name: str) -> None:
    """Explicit official-source download only; no tokens, authentication, or runtime hook."""
    source = UPSTREAM[name]
    directory = base / "upstream"
    directory.mkdir(parents=True, exist_ok=True)
    source_name = "fcn" if name == "fcn-skin" else "schp"
    archive = directory / f"{source_name}-{source['commit']}.zip"
    sources = [
        (
            source["repository"].replace("https://github.com/", "https://codeload.github.com/")
            + "/zip/"
            + source["commit"],
            archive,
            source["source_zip_sha256"],
        )
    ]
    if name == "fcn-skin":
        target = directory / CHECKPOINT_NAMES[name]
        if not target.is_file() or sha256(target) != source["checkpoint_sha256"]:
            download_public_fcn(target)
    for url, target, expected in sources:
        if target.is_file() and sha256(target) == expected:
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
            if sha256(temporary) != expected:
                raise ValueError("Downloaded official source SHA mismatch")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if name == "schp-atr":
        target = directory / CHECKPOINT_NAMES[name]
        if not target.is_file() or sha256(target) != source["checkpoint_sha256"]:
            download_public_checkpoint("schp-atr", target)
    verify_inputs(base, name)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(base: Path, name: str):
    """Build original source architecture on CPU without implicit pretrained downloads."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    directory, checkpoint = verify_inputs(base, name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if name == "fcn-skin":
        import torchvision

        upstream = import_file(
            "freedive_upstream_fcn", directory / "semantic_segmentation/models/fcn.py"
        )
        original = torchvision.models.segmentation.fcn_resnet101

        def no_download(*_args, **_kwargs):
            return original(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)

        categories = [
            key.removeprefix("_categories.") for key in state if key.startswith("_categories.")
        ]
        if categories != ["skin"]:
            raise ValueError(f"Unexpected FCN checkpoint categories: {categories}")
        with patch.object(torchvision.models.segmentation, "fcn_resnet101", no_download):
            model = upstream.FCNResNet101(categories)
        model.load_state_dict(state, strict=True)

        class Output(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, rgb):
                return self.wrapped(rgb)["out"]

    else:
        # Preserve the EXACT inference formula from upstream modules/src/
        # inplace_abn_cpu.cpp: gamma=abs(weight)+eps. Replacing it with ordinary
        # BatchNorm2d or dropping its leaky-ReLU would silently change the model.
        class CPUActivatedBN(nn.Module):
            def __init__(
                self,
                channels,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                activation="leaky_relu",
                slope=0.01,
            ):
                super().__init__()
                self.eps, self.activation, self.slope = eps, activation, slope
                self.weight = nn.Parameter(torch.ones(channels)) if affine else None
                self.bias = nn.Parameter(torch.zeros(channels)) if affine else None
                self.register_buffer("running_mean", torch.zeros(channels))
                self.register_buffer("running_var", torch.ones(channels))

            def forward(self, tensor):
                gamma = (
                    self.weight.abs() + self.eps
                    if self.weight is not None
                    else torch.ones_like(self.running_var)
                )
                # Preserve the upstream ATen operation order, including the
                # subtraction before multiplication, rather than relying on a
                # fused ordinary BatchNorm implementation with different rounding.
                multiplier = torch.rsqrt(self.running_var + self.eps) * gamma
                tensor = (tensor - self.running_mean.view(1, -1, 1, 1)) * multiplier.view(
                    1, -1, 1, 1
                )
                if self.bias is not None:
                    tensor = tensor + self.bias.view(1, -1, 1, 1)
                if self.activation == "leaky_relu":
                    return functional.leaky_relu(tensor, self.slope)
                if self.activation == "relu":
                    return functional.relu(tensor)
                if self.activation == "elu":
                    return functional.elu(tensor)
                if self.activation != "none":
                    raise ValueError("Unexpected ABN activation")
                return tensor

        facade = types.ModuleType("modules")
        facade.InPlaceABNSync = CPUActivatedBN
        with patch.dict(sys.modules, {"modules": facade}):
            upstream = import_file("freedive_upstream_schp", directory / "networks/AugmentCE2P.py")
            model = upstream.resnet101(num_classes=18, pretrained=None)
        state = {key.removeprefix("module."): value for key, value in state["state_dict"].items()}
        model.load_state_dict(state, strict=True)

        class Output(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, rgb):
                return functional.interpolate(
                    self.wrapped(rgb)[0][-1], size=(512, 512), mode="bilinear", align_corners=True
                )

    return Output(model).eval().cpu()


def exportable_pools(model):
    """Exact fixed-bin adaptive pooling for legacy ONNX's non-divisible PSP bins."""
    import torch
    from torch import nn

    class FixedAdaptive(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.size = (size, size) if isinstance(size, int) else size

        def forward(self, value):
            height, width = value.shape[-2:]
            rows = []
            for row in range(self.size[0]):
                y0, y1 = (
                    row * height // self.size[0],
                    ((row + 1) * height + self.size[0] - 1) // self.size[0],
                )
                columns = []
                for column in range(self.size[1]):
                    x0 = column * width // self.size[1]
                    x1 = ((column + 1) * width + self.size[1] - 1) // self.size[1]
                    columns.append(value[..., y0:y1, x0:x1].mean(dim=(-2, -1), keepdim=True))
                rows.append(torch.cat(columns, dim=-1))
            return torch.cat(rows, dim=-2)

    for key, child in model.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(model, key, FixedAdaptive(child.output_size))
        else:
            exportable_pools(child)


def convert(base: Path, name: str, *, output_dir: Path | None = None) -> dict:
    import copy
    import importlib.metadata

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    for package, version in VERSIONS.items():
        if importlib.metadata.version(package).split("+")[0] != version:
            raise ValueError(f"Use pinned converter dependency {package}=={version}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    reference = load_reference(base, name)
    model = copy.deepcopy(reference)
    if name == "schp-atr":
        exportable_pools(model)
    model.eval()
    output = output_dir or base / "onnx"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.onnx"
    example = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
    started = time.monotonic()
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["rgb"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    graph = onnx.load(path)
    graph.producer_name, graph.producer_version = "Freedive Color Studio candidate conversion", "1"
    graph.doc_string, graph.graph.doc_string = "", ""
    for node in graph.graph.node:
        node.doc_string = ""
    source = UPSTREAM[name]
    onnx.helper.set_model_props(
        graph,
        {
            "model_id": source["model_id"],
            "source_commit": source["commit"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "input": "NCHW normalized float32 1x3x512x512",
            "output": "NCHW logits, no sigmoid/softmax",
        },
    )
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.use_deterministic_compute = True
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(20261101)
    fixtures = [
        ("black", np.zeros((320, 480, 3), np.float32)),
        ("cyan", np.full((480, 320, 3), (0.1, 0.45, 0.55), np.float32)),
        ("deterministic-noise", rng.random((173, 287, 3), dtype=np.float32)),
    ]
    public_fixture = ROOT / "tests/fixtures/freediver_moalboal.jpg"
    if sha256(public_fixture) != "771ad767283bb15f5fd47c7c5a3960219b802601f73193ab22f9ad0ec5b2005b":
        raise ValueError("Licensed public parity fixture SHA mismatch")
    import cv2

    public_rgb = cv2.cvtColor(cv2.imread(str(public_fixture)), cv2.COLOR_BGR2RGB)
    fixtures.append(("public-domain-freediver", public_rgb.astype(np.float32) / 255))
    checks = []
    for fixture_name, fixture in fixtures:
        inputs, _geometry = prepare_candidate_input(name, fixture)
        with torch.inference_mode():
            before = reference(torch.from_numpy(inputs)).numpy()
            after = model(torch.from_numpy(inputs)).numpy()
        actual = session.run(["logits"], {"rgb": inputs})[0]
        maximum = float(np.abs(actual - before).max())
        mean = float(np.abs(actual - before).mean())
        pool_maximum = float(np.abs(after - before).max())
        if name == "fcn-skin":
            agreement = float(((actual >= 0) == (before >= 0)).mean())
        else:
            agreement = float((actual.argmax(axis=1) == before.argmax(axis=1)).mean())
        passed = maximum <= 0.002 and mean <= 0.0001 and agreement >= 0.999
        checks.append(
            {
                "fixture": fixture_name,
                "shape": list(fixture.shape),
                "maximum_logit_error": maximum,
                "mean_logit_error": mean,
                "pixel_decision_agreement": agreement,
                "pool_rewrite_maximum_error": pool_maximum,
                "passed": passed,
            }
        )
    manifest = {
        **source,
        "name": name,
        "onnx_sha256": sha256(path),
        "onnx_bytes": path.stat().st_size,
        "dependencies": VERSIONS,
        "opset": 17,
        "input_shape": [1, 3, 512, 512],
        "output_channels": source["channels"],
        "parity": checks,
        "parity_passed": all(check["passed"] for check in checks),
        "conversion_seconds": round(time.monotonic() - started, 3),
        "cpu_only": True,
        "telemetry": False,
        "training": False,
        "reference": (
            "Pinned upstream architecture and checkpoint on CPU; SCHP ABN uses exact "
            "upstream CPU inference formula without the obsolete native extension."
        ),
        "license": "MIT",
        "runtime_downloads": False,
        "underwater_accuracy_claim": False,
    }
    native_copy = base / "native-copy-provenance.json"
    if native_copy.is_file():
        provenance = json.loads(native_copy.read_text())
        package_root = Path(cv2.__file__).resolve().parents[1]
        for native in provenance["copied_native_files"]:
            relative = Path(native["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid isolated native copy provenance")
            if sha256(package_root / relative) != native["sha256"]:
                raise ValueError("Isolated native OpenCV copy changed")
        manifest["conversion_native_opencv"] = provenance
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    directory = source_directory(base, name)
    shutil.copyfile(directory / "LICENSE", output / f"{name}-LICENSE.txt")
    print(json.dumps(manifest, indent=2), flush=True)
    if not manifest["parity_passed"]:
        raise ValueError("Converted model failed CPU numerical parity")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "convert"))
    parser.add_argument("name", choices=("fcn-skin", "schp-atr"))
    parser.add_argument("--base", type=Path, default=ROOT / "build/skin-candidates")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.base, args.name)
    else:
        convert(args.base, args.name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
