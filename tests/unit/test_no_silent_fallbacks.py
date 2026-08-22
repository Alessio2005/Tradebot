"""Phase 0, deliverable 9 - pytest-wrapper rond de fallback-scanner.

Breekt de build zodra er een nieuwe stille fallback in `src/` verschijnt.
Dit is de test-zijde van exit criterium 1; de CI-zijde staat in
`.github/workflows/hygiene.yml`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "scripts" / "audit_fallbacks.py"
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT / "scripts"))
from audit_fallbacks import BLOCKING_KINDS, scan


@pytest.fixture(scope="module")
def findings():
    return scan(SRC)


class TestNoBlockingFallbacks:
    def test_strict_scan_exits_zero(self) -> None:
        """Exit criterium 1: `audit_fallbacks.py --strict` retourneert 0."""
        r = subprocess.run(
            [sys.executable, str(SCANNER), "--strict", "--quiet"],
            capture_output=True, text=True, cwd=str(ROOT), check=False,
        )
        assert r.returncode == 0, (
            "Er zijn blokkerende stille fallbacks in src/:\n"
            + r.stdout[-4000:] + r.stderr[-2000:]
        )

    def test_zero_import_fallbacks(self, findings) -> None:
        """Nul `try/except ImportError` in src/."""
        hits = [f for f in findings if f.kind == "IMPORT_FALLBACK"]
        assert hits == [], "\n".join(f"{f.rel}:{f.line} -> {f.degraded_to}" for f in hits)

    def test_zero_bare_excepts(self, findings) -> None:
        hits = [f for f in findings if f.kind == "BARE_EXCEPT"]
        assert hits == [], "\n".join(f"{f.rel}:{f.line}" for f in hits)

    def test_zero_broad_excepts_without_reraise(self, findings) -> None:
        """Nul `except Exception` zonder onvoorwaardelijke re-raise."""
        hits = [f for f in findings if f.kind == "BROAD_EXCEPT"]
        assert hits == [], "\n".join(f"{f.rel}:{f.line} -> {f.degraded_to}" for f in hits)

    def test_zero_warn_and_degrade(self, findings) -> None:
        hits = [f for f in findings if f.kind == "WARN_AND_DEGRADE"]
        assert hits == [], "\n".join(f"{f.rel}:{f.line} -> {f.degraded_to}" for f in hits)

    def test_no_blocking_kind_at_all(self, findings) -> None:
        hits = [f for f in findings if f.kind in BLOCKING_KINDS]
        assert hits == [], f"{len(hits)} blokkerende bevinding(en)"


class TestScannerItselfWorks:
    """Groen zonder bewezen rood is waardeloos: de scanner moet vangen."""

    def _scan_snippet(self, tmp_path: Path, code: str):
        p = tmp_path / "sample.py"
        p.write_text(code, encoding="utf-8")
        return scan(p)

    def test_detects_import_fallback(self, tmp_path: Path) -> None:
        f = self._scan_snippet(tmp_path, (
            "try:\n"
            "    import hmmlearn\n"
            "    OK = True\n"
            "except ImportError:\n"
            "    OK = False\n"
        ))
        assert [x.kind for x in f] == ["IMPORT_FALLBACK"]

    def test_detects_bare_except(self, tmp_path: Path) -> None:
        f = self._scan_snippet(tmp_path, "try:\n    x = 1\nexcept:\n    x = 0\n")
        assert [x.kind for x in f] == ["BARE_EXCEPT"]

    def test_detects_broad_except(self, tmp_path: Path) -> None:
        f = self._scan_snippet(tmp_path, (
            "def g():\n"
            "    try:\n"
            "        return risky()\n"
            "    except Exception:\n"
            "        return 0.0\n"
        ))
        assert [x.kind for x in f] == ["BROAD_EXCEPT"]

    def test_accepts_broad_except_that_reraises(self, tmp_path: Path) -> None:
        f = self._scan_snippet(tmp_path, (
            "def g():\n"
            "    try:\n"
            "        return risky()\n"
            "    except Exception as exc:\n"
            "        raise RuntimeError('boem') from exc\n"
        ))
        assert f == []

    def test_conditional_raise_is_not_a_reraise(self, tmp_path: Path) -> None:
        """Een raise binnen een `if` zonder `else` laat het andere pad degraderen."""
        f = self._scan_snippet(tmp_path, (
            "def g():\n"
            "    try:\n"
            "        return risky()\n"
            "    except Exception as exc:\n"
            "        if strict:\n"
            "            raise\n"
            "        return 0.0\n"
        ))
        assert [x.kind for x in f] == ["BROAD_EXCEPT"]

    def test_narrow_except_is_advisory_not_blocking(self, tmp_path: Path) -> None:
        f = self._scan_snippet(tmp_path, (
            "def g():\n"
            "    try:\n"
            "        return chol(a)\n"
            "    except LinAlgError:\n"
            "        return jitter(a)\n"
        ))
        assert [x.kind for x in f] == ["SWALLOWED_EXCEPT"]
        assert f[0].kind not in BLOCKING_KINDS


class TestFailFastModuleIsClean:
    def test_failfast_has_no_except_handlers(self) -> None:
        import ast

        src = (ROOT / "src" / "tradebot" / "utils" / "failfast.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        assert [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)] == []


class TestQuantArchitectIsGone:
    def test_no_quant_architect_imports(self) -> None:
        """Exit criterium 4."""
        import ast

        hits: list[str] = []
        for p in SRC.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    hits += [f"{p}:{n.lineno}" for a in n.names
                             if a.name.split(".")[0] == "quant_architect"]
                elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                    if n.module.split(".")[0] == "quant_architect":
                        hits.append(f"{p}:{n.lineno}")
        assert hits == [], "\n".join(hits)
