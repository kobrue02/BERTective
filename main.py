"""Entry point shim — delegates to bertective.cli.

Kept for backwards compatibility.  Prefer running:
    bertective <command> [options]
or:
    python -m bertective.cli <command> [options]
"""
from bertective.cli import main

if __name__ == "__main__":
    main()
