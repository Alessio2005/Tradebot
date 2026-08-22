"""Phase 0 helper - unwrap `except Exception` handlers that swallow errors.

Transformeert het canonieke stille-degradatiepatroon

    try:
        <body>
    except Exception as exc:
        logger.warning(...)
        return <default>

naar

    <body>

zodat de fout propageert in plaats van te worden vervangen door een default.

Het gereedschap is bewust CONSERVATIEF. Het weigert te transformeren wanneer:
  * de Try een `else:` of `finally:` heeft (semantiek verandert dan),
  * er meerdere handlers zijn (bijv. een nauwe handler naast de brede),
  * de handler onvoorwaardelijk raist (dan is er geen probleem),
  * de handler `continue` / `break` bevat (retry- of skip-lus: handmatig werk),
  * de try-body een `yield` bevat.

Alles wat het weigert, wordt gerapporteerd zodat het met de hand kan worden
afgehandeld. Draai daarna altijd ruff + de testsuite.

Gebruik:
    python scripts/unwrap_broad_except.py --list  src/tradebot/risk
    python scripts/unwrap_broad_except.py --apply src/tradebot/risk/var.py
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROAD = {"Exception", "BaseException"}
SKIP_PARTS = {"__pycache__", ".git", ".venv", "build", "dist"}


def exc_names(node: ast.expr | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Tuple):
        out: list[str] = []
        for e in node.elts:
            out.extend(exc_names(e))
        return out
    return []


def always_raises(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and always_raises(stmt.body) and always_raises(stmt.orelse):
                return True
        elif isinstance(stmt, (ast.Return, ast.Break, ast.Continue, ast.Pass)):
            return False
    return False


def contains(body: list[ast.stmt], types: tuple[type, ...]) -> bool:
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, types):
                return True
    return False


def analyse(path: Path):
    """Yield (try_node, verdict, reason) for every broad-except Try in `path`."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        broad = [h for h in node.handlers if set(exc_names(h.type)) & BROAD]
        if not broad:
            continue
        h = broad[0]
        if always_raises(h.body):
            continue  # not a fallback
        if node.orelse or node.finalbody:
            yield node, "MANUAL", "heeft else/finally"
            continue
        if len(node.handlers) > 1:
            yield node, "MANUAL", f"{len(node.handlers)} handlers"
            continue
        if contains(h.body, (ast.Continue, ast.Break)):
            yield node, "MANUAL", "handler bevat continue/break (retry-lus)"
            continue
        if contains(node.body, (ast.Yield, ast.YieldFrom)):
            yield node, "MANUAL", "try-body bevat yield"
            continue
        # Geneste Try-nodes: de bottom-up tekstherschrijving hieronder gaat
        # ervan uit dat regelbereiken elkaar niet overlappen. Een Try binnen een
        # Try schendt die aanname en produceert kapotte syntax.
        if any(isinstance(n, ast.Try) for st in node.body for n in ast.walk(st)):
            yield node, "MANUAL", "try-body bevat een geneste try (overlappende bereiken)"
            continue
        yield node, "AUTO", "unwrap: body behouden, handler verwijderen"


def unwrap_file(path: Path, apply: bool) -> tuple[int, int]:
    """Unwrap every AUTO-eligible Try in `path`. Returns (auto, manual)."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    targets = []
    manual = 0
    for node, verdict, reason in analyse(path):
        if verdict == "AUTO":
            targets.append(node)
        else:
            manual += 1
            print(f"  MANUAL {path.name}:{node.lineno}  {reason}")

    if not targets:
        return 0, manual

    # process bottom-up so line numbers stay valid
    targets.sort(key=lambda n: n.lineno, reverse=True)
    for node in targets:
        try_line = node.lineno - 1                    # 0-based: the `try:` line
        body_start = node.body[0].lineno - 1
        body_end = max(
            getattr(s, "end_lineno", s.lineno) for s in node.body
        )                                             # 1-based inclusive
        handler = node.handlers[0]
        h_end = max(
            getattr(s, "end_lineno", s.lineno) for s in handler.body
        )

        try_indent = len(lines[try_line]) - len(lines[try_line].lstrip())
        body_indent = len(lines[body_start]) - len(lines[body_start].lstrip())
        dedent = body_indent - try_indent

        new_body = []
        for ln in lines[body_start:body_end]:
            if ln.strip() == "":
                new_body.append(ln)
            elif dedent > 0 and ln[:dedent].strip() == "":
                new_body.append(ln[dedent:])
            else:
                new_body.append(ln)

        lines[try_line:h_end] = new_body

    if apply:
        path.write_text("".join(lines), encoding="utf-8")
        # re-parse to guarantee we did not produce broken syntax
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return len(targets), manual


def iter_py(target: Path):
    if target.is_file():
        yield target
        return
    for p in sorted(target.rglob("*.py")):
        if not any(part in SKIP_PARTS for part in p.parts):
            yield p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    target = Path(args.target)
    if not target.is_absolute():
        target = ROOT / target

    tot_auto = tot_manual = 0
    for p in iter_py(target):
        if args.list:
            for node, verdict, reason in analyse(p):
                print(f"{p.relative_to(ROOT).as_posix()}:{node.lineno}: {verdict}  {reason}")
                tot_auto += verdict == "AUTO"
                tot_manual += verdict == "MANUAL"
        else:
            a, m = unwrap_file(p, args.apply)
            if a:
                print(f"  {'unwrapped' if args.apply else 'would unwrap'} {a} in "
                      f"{p.relative_to(ROOT).as_posix()}")
            tot_auto += a
            tot_manual += m
    print(f"\nAUTO={tot_auto}  MANUAL={tot_manual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
