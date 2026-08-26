"""Markdown-rendering van de econometrische diagnose — deliverable 9.

Gescheiden van de meting in `validation/diagnostics_report.py` om dezelfde reden
als overal in deze codebase: een module die meet én opmaakt, verleidt tot het
aanpassen van de meting omdat de tabel er anders beter uitziet.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..validation.diagnostics_report import SymbolDiagnostics
from ..validation.econometrics import arch_gate_verdict

__all__ = ["render_diagnostics_report"]


def _yes(flag: bool) -> str:
    return "ja" if flag else "nee"


def render_diagnostics_report(
    payload: Mapping[str, Any],
    diagnostics: Mapping[str, SymbolDiagnostics],
) -> str:
    universe = payload["universe"]
    frac_cfg = payload["fracdiff_config"]
    lines: list[str] = []
    add = lines.append

    add("# PHASE 6 — ECONOMETRISCHE DIAGNOSE VAN DE VOL-PIJPLIJN")
    add("")
    add("> **Deliverable 9** · Phase 6, stappen 3 en 4 · §8.1, §8.2")
    add(f"> **git_sha:** `{payload['git_sha']}`")
    add(f"> **Universum:** {len(universe['symbols'])} gecertificeerde reeksen · "
        f"{universe['n_bars']} bars · {universe['period_start']} t/m "
        f"{universe['period_end']}")
    add("")
    add("Elke toets hieronder is gedraaid VOORDAT er één model is gefit. Dat is "
        "de volgorde die §3 voorschrijft en zij is niet omkeerbaar: een "
        "ARCH-toets die je draait nadat een GARCH-fit is mislukt, is geen poort "
        "meer maar een verklaring achteraf.")
    add("")

    # ------------------------------------------------------------------ #
    add("## 0. Waarop is getoetst, en waarom dat drie reeksen zijn")
    add("")
    add("Een toets op de verkeerde reeks levert een keurige p-waarde die niets "
        "betekent. Per symbool zijn drie transformaties getoetst:")
    add("")
    add("| Reeks | Wat je verwacht | Waarvoor de uitkomst telt |")
    add("|---|---|---|")
    add("| `log_price` | eenheidswortel | controle op de data; **niet** het "
        "ARCH-poortoordeel |")
    add("| `log_return` | stationair, ARCH-effecten aanwezig | **hier geldt het "
        "ARCH-poortoordeel** |")
    add("| `fracdiff(d*)` | stationair mét behouden geheugen | de reeks die een "
        "feature-pijplijn zou gebruiken |")
    add("")
    add("De ARCH-toets op `log_price` staat in de tabellen maar telt niet als "
        "poort. Op een niet-stationaire reeks autocorreleren de gekwadrateerde "
        "residuen vanzelf, dus de toets verwerpt daar bijna altijd — een "
        "artefact van de niet-stationariteit, geen conditionele "
        "heteroskedasticiteit.")
    add("")

    # ------------------------------------------------------------------ #
    add("## 1. Het ARCH-poortoordeel per reeks")
    add("")
    add("Stap 3: *\"De ARCH-test is de poortwachter: is er geen aantoonbare "
        "conditionele heteroskedasticiteit, dan is een GARCH-structuur niet "
        "gerechtvaardigd en stopt het spoor daar met een gedocumenteerd "
        "oordeel.\"*")
    add("")
    add("| Symbool | Engle ARCH LM | p-waarde | Poort | Gevolg voor H1 |")
    add("|---|---:|---:|---|---|")
    for symbol, diag in diagnostics.items():
        test = diag.log_return.engle_arch
        gate = "**OPEN**" if diag.arch_gate_open else "**DICHT**"
        consequence = ("GARCH-familie mag worden gefit"
                       if diag.arch_gate_open else
                       "reeks wordt uit de competitie gede-scoped")
        add(f"| `{symbol}` | {test.statistic:.2f} | {test.p_value:.3g} | "
            f"{gate} | {consequence} |")
    add("")
    n_open = sum(1 for d in diagnostics.values() if d.arch_gate_open)
    add(f"**{n_open} van {len(diagnostics)} reeksen** passeren de ARCH-poort.")
    add("")
    for symbol, diag in diagnostics.items():
        add(f"- `{symbol}` — {arch_gate_verdict(diag.log_return)}")
    add("")

    # ------------------------------------------------------------------ #
    add("## 2. Stationariteit: ADF en KPSS samen")
    add("")
    add("ADF en KPSS hebben TEGENGESTELDE nulhypotheses. Dat is geen redundantie "
        "maar de reden dat zij samen informatiever zijn dan elk apart:")
    add("")
    add("| ADF | KPSS | Betekenis |")
    add("|---|---|---|")
    add("| verwerpt | verwerpt niet | stationair — eenduidig |")
    add("| verwerpt niet | verwerpt | eenheidswortel — eenduidig |")
    add("| verwerpt | verwerpt | tegenstrijdig; lange geheugen of structurele breuk |")
    add("| verwerpt niet | verwerpt niet | onbeslist — de data kan het niet zeggen |")
    add("")
    add("Een pijplijn die alleen ADF draait, leest de laatste twee gevallen als "
        "\"prima\" en gaat door.")
    add("")
    for series_name, attr in (("log_price", "log_price"),
                              ("log_return", "log_return"),
                              ("fracdiff", "fracdiff")):
        add(f"### 2.{'123'[('log_price','log_return','fracdiff').index(attr)]} "
            f"`{series_name}`")
        add("")
        add("| Symbool | ADF stat | ADF p | KPSS stat | KPSS p | Oordeel |")
        add("|---|---:|---:|---:|---:|---|")
        for symbol, diag in diagnostics.items():
            s = getattr(diag, attr)
            clipped = " *" if s.kpss.detail["p_value_clipped_at_table_edge"] else ""
            add(f"| `{symbol}` | {s.adf.statistic:.3f} | {s.adf.p_value:.3g} | "
                f"{s.kpss.statistic:.4f} | {s.kpss.p_value:.3g}{clipped} | "
                f"{s.stationarity_verdict} |")
        add("")
    add("`*` = de KPSS-p-waarde is door `statsmodels` afgekapt op de rand van "
        "zijn tabel. Een `p = 0,01` betekent daar `<= 0,01` en een `p = 0,10` "
        "betekent `>= 0,10`.")
    add("")

    # ------------------------------------------------------------------ #
    add("## 3. Autocorrelatie en structurele breuken")
    add("")
    add("| Symbool | Ljung-Box p (returns) | autocorrelatie? | CUSUM stat | "
        "kritiek | breuk? |")
    add("|---|---:|---|---:|---:|---|")
    for symbol, diag in diagnostics.items():
        lb = diag.log_return.ljung_box
        cs = diag.log_return.cusum
        add(f"| `{symbol}` | {lb.p_value:.3g} | {_yes(lb.rejected)} | "
            f"{cs.statistic:.3f} | {cs.detail['critical_value']:.3f} | "
            f"{_yes(cs.rejected)} |")
    add("")
    n_break = sum(1 for d in diagnostics.values() if d.log_return.cusum.rejected)
    add(f"**{n_break} van {len(diagnostics)} reeksen** vertonen een structurele "
        "breuk in het gemiddelde over dit venster. Dat is relevant voor elk "
        "model in deze fase: een model dat over één venster wordt gefit en "
        "beoordeeld terwijl er een breuk in zit, levert een OOS-prestatie op die "
        "een gemiddelde is over twee verschillende regimes en die dus over geen "
        "van beide iets zegt.")
    add("")

    # ------------------------------------------------------------------ #
    add("## 4. Fractionele differentiëring — de gekozen `d` per reeks")
    add("")
    add(f"Zoekbereik uit `conf/model/fracdiff.yaml`: `d ∈ "
        f"[{frac_cfg['d_lo']}, {frac_cfg['d_hi']}]`, gewichtsdrempel "
        f"`{frac_cfg['weight_threshold']}`, ADF-doel `p < "
        f"{frac_cfg['adf_p_target']}`.")
    add("")
    add("Het punt van FFD is de afruil tussen stationariteit en geheugen. Het "
        "CRITERIUM van §8.1 is `d*` zelf — lager is meer geheugen, want de "
        "FFD-gewichten dalen monotoon in `d`. De kolommen **geheugen** en "
        "**bij d = 1** zijn AFML's correlatiediagnostiek; §4.2 laat zien dat "
        "die maat op dit venster niet monotoon is en soms negatief, en dat zij "
        "hier dus niet als criterium kan dienen. Zij staat in de tabel omdat "
        "haar instabiliteit zelf een resultaat is.")
    add("")
    add("| Symbool | `d*` | ADF p | geheugen | bij `d = 1` | winst | venster | "
        "bars weg | bars over |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for symbol, diag in diagnostics.items():
        f = diag.frac
        gain = f.memory_retained - f.memory_retained_at_d_one
        add(f"| `{symbol}` | {f.d:.4f} | {f.adf_p_value:.3g} | "
            f"{f.memory_retained:.4f} | {f.memory_retained_at_d_one:.4f} | "
            f"{gain:+.4f} | {f.ffd_window_bars} | {f.n_bars_consumed} | "
            f"{f.n_obs_after_ffd} |")
    add("")
    add("`d` staat NIET als constante in code. Hij is per reeks gezocht binnen "
        "de bovenstaande grenzen en wordt hier per reeks geregistreerd; de "
        "zoekruimte komt uit `conf/model/fracdiff.yaml`.")
    add("")
    add("### 4.1 Wat FFD op DEZE reekslengte kost")
    add("")
    add("De kolommen **venster** en **bars weg** zijn de reden dat de "
        "gewichtsdrempel in `conf/` op `1e-4` staat en niet op de AFML-waarde "
        "`1e-5`. Gemeten op `d ≈ 0,45`:")
    add("")
    add("| drempel | venster | truncatiemassa | bars over van 1.743 |")
    add("|---:|---:|---:|---:|")
    add("| `1e-2` | 11 | 0,1216 | 1.733 |")
    add("| `1e-3` | 49 | 0,0571 | 1.695 |")
    add("| `1e-4` | 238 | 0,0271 | **1.506** |")
    add("| `1e-5` | 1.163 | 0,0131 | 581 |")
    add("")
    add("AFML gebruikt `1e-5`, maar dat is geschreven voor reeksen van "
        "tienduizenden bars. Op 1.743 bars kost `1e-5` er 1.162 — twee derde "
        "van de steekproef — en de resterende 581 bars beslaan alleen het "
        "laatste deel van het venster. Elke toets op die reeks meet dan een "
        "ander tijdvak dan de rest van deze fase.")
    add("")
    add("**Defect in mijn eigen werk, gevonden tijdens deze stap.** De eerste "
        "versie van deze tabel had een kolom *warm-up* die consequent `0 bars` "
        "rapporteerde. `frac_diff_ffd` zet de eerste rijen niet op NaN maar "
        "LAAT ZE VALLEN, en mijn meting telde NaN's. Daardoor zag een "
        "transformatie die twee derde van de steekproef opat, eruit alsof zij "
        "gratis was. De correctie is een lengteverschil in plaats van een "
        "NaN-telling, en zij is de directe aanleiding voor de drempelkeuze "
        "hierboven.")
    add("")
    add("### 4.2 De geheugenmaat is op dit venster NIET monotoon in `d`")
    add("")
    add("AFML §5.5 meet behouden geheugen als `corr(FFD_d(x), x)` en toont een "
        "curve die netjes daalt van 1 naar 0 als `d` van 0 naar 1 loopt. Op "
        "DEZE reeks doet zij dat niet. Gemeten op hetzelfde raster:")
    add("")
    header = sorted(next(iter(diagnostics.values())).frac.memory_curve,
                    key=float)
    add("| Symbool | " + " | ".join(f"`d={h}`" for h in header) + " |")
    add("|---" * (len(header) + 1) + "|")
    for symbol, diag in diagnostics.items():
        row = " | ".join(f"{diag.frac.memory_curve[h]:+.3f}" for h in header)
        add(f"| `{symbol}` | {row} |")
    add("")
    add("De maat springt van teken en is niet monotoon. De reden is geen "
        "rekenfout maar de vorm van dit venster: AFML's curve wordt gemeten op "
        "reeksen waarin het NIVEAU door één sterke trend wordt gedomineerd, en "
        "2021-11 t/m 2026-08 is dat niet — piek, instorting, herstel. De "
        "FFD-reeks is in essentie een lang gewogen gemiddelde van verleden "
        "returns, en dat heeft op een niet-monotoon prijspad geen systematisch "
        "verband met het niveau.")
    add("")
    add("**Gevolg voor de interpretatie.** De kolom *geheugen* in tabel 4 is "
        "DIAGNOSTIEK en geen criterium. Het criterium van §8.1 is `d` zelf: de "
        "minimale `d` die ADF-stationariteit haalt, IS per constructie het "
        "maximale behoud van geheugen, want de FFD-gewichten dalen monotoon in "
        "`d`. Wie de correlatiekolom als criterium zou gebruiken, zou op deze "
        "data een willekeurige `d` kiezen.")
    add("")
    add("### 4.3 De ADF/KPSS-tegenspraak op de FFD-reeks")
    add("")
    add("`min_frac_diff` zoekt de kleinste `d` die ADF onder `p = 0,05` brengt "
        "en optimaliseert dus UITSLUITEND tegen ADF. Tabel 2.3 laat zien wat "
        "dat oplevert: de ADF-p ligt per constructie net onder 0,05, terwijl "
        "KPSS op elke reeks verwerpt. De twee toetsen zijn het oneens, en dat "
        "is geen bug maar precies wat het paar hoort te laten zien.")
    add("")
    add("De interpretatie: de minimale `d` in ADF-zin is voor KPSS "
        "ONDERGEDIFFERENTIEERD. Wie een reeks nodig heeft die BEIDE toetsen "
        "doorstaat, moet een hogere `d` accepteren en dus geheugen inleveren. "
        "Deze fase kiest dat niet, omdat §8.1 expliciet de minimale `d` bij "
        "ADF-stationariteit voorschrijft — maar het feit hoort in het rapport "
        "en niet in een voetnoot, want een feature-pijplijn die deze reeks "
        "gebruikt, gebruikt een reeks die KPSS niet-stationair noemt.")
    add("")

    # ------------------------------------------------------------------ #
    add("## 5. Wat deze diagnose NIET zegt")
    add("")
    add("- Zij zegt niet dat een model dat door de ARCH-poort komt, ook zal "
        "winnen. De poort is een noodzakelijke voorwaarde, geen aanwijzing.")
    add("- Zij zegt niets over de RV-proxy. QLIKE vergelijkt een forecast met "
        "een proxy, en de gecertificeerde store bevat geen intraday-data; die "
        "beperking staat in `reports/GARCH_VS_EWMA_COMPETITION.md` en in de "
        "Data Adequacy Gate, niet hier.")
    add("- Zij zegt niets over de economische waarde van een model. Elke "
        "statistische winst moet nog door de authoritative engine.")
    add("")
    return "\n".join(lines) + "\n"
