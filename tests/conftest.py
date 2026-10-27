import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def rgb() -> np.ndarray:
    y, x = np.mgrid[0:96, 0:128].astype(np.float32)
    out = np.stack((0.05 + x / 128 * 0.25, 0.2 + y / 96 * 0.5, 0.3 + x / 128 * 0.4), axis=2)
    out[20:42, 40:65] = [0.52, 0.42, 0.31]
    out[50:70, 70:100] = [0.02, 0.02, 0.02]
    return out.astype(np.float32)


@pytest.fixture
def photo_path(tmp_path, rgb):
    path = tmp_path / "погружение.png"
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)
    return path
