"""`python -I -m tradebot.options`: zie `tradebot.options.cli`."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - CLI-ingang, getest via cli.main()
    raise SystemExit(main())
