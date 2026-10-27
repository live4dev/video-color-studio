import sys


def main() -> int:
    arguments = sys.argv[1:]
    if "--smoke-test" in arguments:
        from .smoke import run

        return run(arguments)
    if any(arg in arguments for arg in ("--input", "--output", "--help", "-h")):
        from .cli import run

        return run(arguments)
    from .ui.app import run

    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
