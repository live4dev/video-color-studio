import sys


def main() -> int:
    arguments = sys.argv[1:]
    flags = {argument.split("=", 1)[0] for argument in arguments}
    if "--mcp" in flags:
        from .control.mcp_server import main

        return main()
    if "--smoke-video" in flags:
        from .video_smoke import run

        return run(arguments)
    if "--smoke-test" in flags:
        from .smoke import run

        return run(arguments)
    if flags.intersection(("--input", "--project", "--output", "--help", "-h")):
        from .cli import run

        return run(arguments)
    from .ui.app import run

    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
