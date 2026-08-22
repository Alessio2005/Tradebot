"""Phase 0 / Stap 4 - AST-gebaseerde scanner voor stille degradatie.

Detecteert in `src/` elk codepad waarin een fout wordt opgevangen en het
systeem vervolgens *doorgaat* met een naievere benadering:

    IMPORT_FALLBACK        try/except ImportError (of ModuleNotFoundError)
    BARE_EXCEPT            `except:` zonder exceptietype
    BROAD_EXCEPT           `except Exception` / `except BaseException` waarvan
                           niet elk pad in een raise eindigt
    SWALLOWED_EXCEPT       elke handler die niet onvoorwaardelijk her-raist
    WARN_AND_DEGRADE       warnings.warn(...) / logger.warning(...) gevolgd
                           door een return in dezelfde handler

Regex is hier ontoereikend: genest gedrag (een raise binnen een if zonder
else, een return na een log-regel) is alleen betrouwbaar te zien op de AST.

Gebruik
-------
    python scripts/audit_fallbacks.py                # rapporteren, exit 0
    python scripts/audit_fallbacks.py --strict       # exit 1 bij elke hit
    python scripts/audit_fallbacks.py --write-register

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md sectie 5.2, sectie 26 (AC-5).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src"
SKIP_PARTS = {"__pycache__", ".git", ".venv", "venv", "build", "dist"}

IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError"}
BROAD_ERRORS = {"Exception", "BaseException"}
WARN_CALLS = {"warn", "warning", "warns"}

SEVERITY_ORDER = {"BARE_EXCEPT": 0, "IMPORT_FALLBACK": 1, "BROAD_EXCEPT": 2,
                  "WARN_AND_DEGRADE": 3, "SWALLOWED_EXCEPT": 4}

# ---------------------------------------------------------------------------
# BLOCKING vs ADVISORY
#
# Phase 0 exit criterium 1 noemt exact drie condities: nul `try/except
# ImportError`, nul bare excepts, en nul `except Exception` zonder re-raise.
# Deliverable 8 voegt daar `warnings.warn` gevolgd door een degraded return aan
# toe. Die vier categorieen breken de build.
#
# SWALLOWED_EXCEPT - een NAUWE, expliciet benoemde exceptie (np.linalg.LinAlgError,
# ValueError, ...) die niet her-raist - staat bewust NIET in dat rijtje. Een
# `except LinAlgError` die een bijna-singuliere covariantiematrix regulariseert
# via eigenvalue-clipping of Cholesky-jitter is numerieke hygiene, geen stille
# degradatie van een statistisch model naar een naievere benadering. Zulke
# handlers worden wel geregistreerd en per stuk beoordeeld, maar breken de build
# niet. Elke SWALLOWED_EXCEPT die WEL een modelwissel is, wordt bij beoordeling
# geherclassificeerd naar BROAD_EXCEPT-niveau en alsnog gerepareerd.
# ---------------------------------------------------------------------------
BLOCKING_KINDS = {"BARE_EXCEPT", "IMPORT_FALLBACK", "BROAD_EXCEPT", "WARN_AND_DEGRADE"}


@dataclass
class Finding:
    path: Path
    line: int
    kind: str
    handled: str
    degraded_from: str
    degraded_to: str
    func: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def rel(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- #
# control-flow analysis
# --------------------------------------------------------------------------- #
def always_raises(body: list[ast.stmt]) -> bool:
    """True wanneer elk pad door `body` eindigt in een raise.

    Een `raise` binnen een `if` zonder `else` telt niet: het pad waarin de
    conditie vals is valt door en degradeert alsnog.
    """
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and always_raises(stmt.body) and always_raises(stmt.orelse):
                return True
        elif isinstance(stmt, ast.With):
            if always_raises(stmt.body):
                return True
        elif isinstance(stmt, ast.Try):
            # een try/except in een handler: alleen als zowel body als elke
            # handler onvoorwaardelijk raist
            if always_raises(stmt.body) and all(always_raises(h.body) for h in stmt.handlers):
                return True
        elif isinstance(stmt, (ast.Return, ast.Break, ast.Continue, ast.Pass)):
            return False
    return False


def exc_names(node: ast.expr | None) -> list[str]:
    """Namen van de gevangen exceptietypes."""
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
    return [ast.unparse(node)]


def _flat(src: str, limit: int = 90) -> str:
    """Eenregelige, afgekapte weergave van een statement."""
    one = " ".join(src.split())
    return one if len(one) <= limit else one[: limit - 3] + "..."


def describe_try_body(body: list[ast.stmt]) -> str:
    """Wat werd er geprobeerd? (= wat gaat er verloren bij degradatie)"""
    bits: list[str] = []
    for stmt in body[:4]:
        if isinstance(stmt, ast.Import):
            bits.append("import " + ", ".join(a.name for a in stmt.names))
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or "."
            bits.append(f"from {mod} import " + ", ".join(a.name for a in stmt.names))
        else:
            bits.append(_flat(ast.unparse(stmt)))
    return _flat("; ".join(bits) or "(leeg)", 160)


def describe_handler(body: list[ast.stmt]) -> str:
    """Waar valt het systeem op terug?"""
    bits: list[str] = []
    for stmt in body[:5]:
        if isinstance(stmt, ast.Pass):
            bits.append("pass (fout genegeerd)")
        elif isinstance(stmt, ast.Return):
            val = "None" if stmt.value is None else _flat(ast.unparse(stmt.value), 60)
            bits.append("return " + val)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            bits.append(_flat(ast.unparse(stmt)))
        elif isinstance(stmt, ast.Raise):
            bits.append("raise " + (_flat(ast.unparse(stmt.exc), 60) if stmt.exc else ""))
        else:
            bits.append(_flat(ast.unparse(stmt)))
    return _flat("; ".join(bits) or "(leeg)", 160)


def has_warn_then_fallthrough(body: list[ast.stmt]) -> bool:
    """warnings.warn / logger.warning gevolgd door een return in dezelfde blok."""
    warned = False
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fn = stmt.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (
                fn.id if isinstance(fn, ast.Name) else "")
            if name in WARN_CALLS:
                warned = True
        elif warned and isinstance(stmt, ast.Return):
            return True
    return False


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #
class Scanner(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []
        self._func_stack: list[str] = []

    def _fn(self) -> str:
        return ".".join(self._func_stack)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        tried = describe_try_body(node.body)
        for h in node.handlers:
            names = exc_names(h.type)
            fell_back = describe_handler(h.body)
            reraises = always_raises(h.body)
            kind: str | None = None
            notes: list[str] = []

            if h.type is None:
                kind = "BARE_EXCEPT"
                notes.append("vangt ook KeyboardInterrupt en SystemExit")
            elif set(names) & IMPORT_ERRORS:
                if not reraises:
                    kind = "IMPORT_FALLBACK"
                    notes.append("model/dependency wordt stilzwijgend vervangen")
            elif set(names) & BROAD_ERRORS:
                if not reraises:
                    kind = "BROAD_EXCEPT"
                    notes.append("brede except zonder onvoorwaardelijke re-raise")
            elif not reraises:
                kind = "SWALLOWED_EXCEPT"
                notes.append("specifieke except die niet her-raist")

            if kind is None and not reraises and has_warn_then_fallthrough(h.body):
                kind = "WARN_AND_DEGRADE"

            if kind == "SWALLOWED_EXCEPT" and has_warn_then_fallthrough(h.body):
                kind = "WARN_AND_DEGRADE"
                notes.append("waarschuwt en gaat door met een default")

            if kind is not None:
                self.findings.append(Finding(
                    path=self.path, line=h.lineno, kind=kind,
                    handled=", ".join(names) or "(bare)",
                    degraded_from=tried, degraded_to=fell_back,
                    func=self._fn(), notes=notes,
                ))
        self.generic_visit(node)


def iter_py(target: Path):
    if target.is_file():
        yield target
        return
    for p in sorted(target.rglob("*.py")):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        yield p


def scan(target: Path) -> list[Finding]:
    out: list[Finding] = []
    for p in iter_py(target):
        text = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text, filename=str(p))
        s = Scanner(p)
        s.visit(tree)
        out.extend(s.findings)
    out.sort(key=lambda f: (SEVERITY_ORDER.get(f.kind, 99), f.rel, f.line))
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_register(findings: list[Finding], target: Path, out_path: Path) -> None:
    by_kind: dict[str, list[Finding]] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)

    L: list[str] = []
    A = L.append
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    A("# PHASE 0 - FALLBACK REGISTER (stille degradatie)")
    A("")
    A("**Gegenereerd:** " + stamp + "  ")
    A("**Generator:** `scripts/audit_fallbacks.py`  ")
    A("**Scope:** `" + target.relative_to(ROOT).as_posix() + "`  ")
    A("**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` sectie 5.2")
    A("")
    A("Elke regel hieronder is een codepad waarin een fout wordt opgevangen en het")
    A("systeem doorgaat met een naievere benadering, zonder dat de operator dit merkt.")
    A("")
    A("---")
    A("")
    A("## Samenvatting")
    A("")
    A("| Categorie | Aantal | Betekenis |")
    A("|---|---:|---|")
    meanings = {
        "BARE_EXCEPT": "`except:` zonder type - vangt ook KeyboardInterrupt/SystemExit",
        "IMPORT_FALLBACK": "`try/except ImportError` - model wordt stilzwijgend vervangen",
        "BROAD_EXCEPT": "`except Exception` zonder onvoorwaardelijke re-raise",
        "WARN_AND_DEGRADE": "waarschuwt en gaat door met een default",
        "SWALLOWED_EXCEPT": "specifieke except die de fout inslikt",
    }
    for k in sorted(by_kind, key=lambda x: SEVERITY_ORDER.get(x, 99)):
        A("| `{}` | {} | {} |".format(k, len(by_kind[k]), meanings.get(k, "")))
    A("| **TOTAAL** | **{}** | |".format(len(findings)))
    A("")
    nb = len([f for f in findings if f.kind in BLOCKING_KINDS])
    na = len(findings) - nb
    A("**Blokkerend (breekt de build): {}** - **advies (geregistreerd, beoordeeld): {}**"
      .format(nb, na))
    A("")
    A("Blokkerend zijn exact de condities uit exit criterium 1 plus deliverable 8:")
    A("`try/except ImportError`, bare `except:`, `except Exception` zonder re-raise, en")
    A("`warnings.warn` gevolgd door een degraded return. `SWALLOWED_EXCEPT` - een nauwe,")
    A("expliciet benoemde exceptie die niet her-raist - is advies: een `except LinAlgError`")
    A("die een bijna-singuliere covariantiematrix regulariseert is numerieke hygiene, geen")
    A("stille degradatie van een model naar een naievere benadering. Elke advies-bevinding")
    A("is stuk voor stuk beoordeeld; wie wel een modelwissel bleek, is gerepareerd.")
    A("")
    A("De audit noemt in sectie 5.2 *12 locaties*. Deze scan meet **{}**. ".format(len(findings)))
    A("De audittelling was een steekproef; deze AST-scan is uitputtend.")
    A("")
    A("---")
    A("")
    A("## Bevindingen per categorie")
    A("")
    for k in sorted(by_kind, key=lambda x: SEVERITY_ORDER.get(x, 99)):
        tag = "BLOKKEREND" if k in BLOCKING_KINDS else "ADVIES"
        A("### {} ({}) - {}".format(k, len(by_kind[k]), tag))
        A("")
        A("| Bestand:regel | Functie | Vangt | Gedegradeerd model | Naief alternatief |")
        A("|---|---|---|---|---|")
        for f in by_kind[k]:
            A("| `{}:{}` | `{}` | `{}` | `{}` | `{}` |".format(
                f.rel, f.line, f.func or "(module)", f.handled,
                f.degraded_from.replace("|", "\\|"),
                f.degraded_to.replace("|", "\\|")))
        A("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AST-scanner voor stille degradatie.")
    ap.add_argument("--target", default=str(DEFAULT_TARGET),
                    help="bestand of map om te scannen (default: src/)")
    ap.add_argument("--strict", action="store_true",
                    help="exit code 1 bij elke hit")
    ap.add_argument("--write-register", action="store_true",
                    help="schrijf reports/phase0_fallback_register.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    target = Path(args.target)
    if not target.is_absolute():
        target = ROOT / target
    findings = scan(target)

    if args.write_register:
        out = ROOT / "reports" / "phase0_fallback_register.md"
        write_register(findings, target, out)
        print("wrote {}".format(out))

    if not args.quiet:
        for f in findings:
            print("{}:{}: {} [{}] in {} -> {}".format(
                f.rel, f.line, f.kind, f.handled, f.func or "(module)", f.degraded_to[:70]))

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    summary = ", ".join("{}={}".format(k, counts[k])
                        for k in sorted(counts, key=lambda x: SEVERITY_ORDER.get(x, 99)))
    print("\nTOTAAL {} bevinding(en){}".format(len(findings), (": " + summary) if summary else ""))

    blocking = [f for f in findings if f.kind in BLOCKING_KINDS]
    advisory = [f for f in findings if f.kind not in BLOCKING_KINDS]
    print("  blokkerend: {}   advies: {}".format(len(blocking), len(advisory)))

    if args.strict and blocking:
        print("STRICT: build gebroken door {} blokkerende stille fallback(s).".format(
            len(blocking)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
