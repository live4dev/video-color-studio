import sys
from pathlib import Path

from freedive_color.__main__ import main

if __name__ == "__main__":
    if Path(sys.executable).stem == "freedive-color-mcp":
        from freedive_color.control.mcp_server import main

    raise SystemExit(main())
