"""Allow ``python -m atlasai``."""

from atlasai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
