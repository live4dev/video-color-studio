"""Russian source strings; add gettext catalogs in locale/ for further languages."""

import gettext
from pathlib import Path

_translation = gettext.translation(
    "freedive_color", localedir=Path(__file__).parent / "locale", languages=["ru"], fallback=True
)
_ = _translation.gettext
