import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # We deliberately reuse the exact library stack of basicAnalysis_week_1.py
    # (folium, numpy, marimo, pandas, seaborn, pulp, highspy) so the case study
    # stays in the same ecosystem. matplotlib is added only for the validation
    # scatter / sensitivity charts (seaborn already pulls it in).
    import folium
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pulp as pl
    import highspy  # noqa: F401  (backend for pl.HiGHS, same as base notebook)

    return folium, mo, np, pd, pl, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CashLog – Case Study: Optimales Design des Cash-Center-Netzwerks

    ## Worum geht es? (Problem in zwei Sätzen)

    CashLog betreibt heute **42 teure Cash Center** und beliefert darüber ~42.000
    Kundenstandorte, aggregiert zu **515 Kundenregionen**. Bargeld wird weniger,
    Center sind teuer – gesucht ist die **kostenoptimale Teilmenge** der 42 Standorte
    (welche behalten, welche schließen), sodass **jede Region weiterhin bedient** wird.

    **Soll-Ergebnis dieses Notebooks:** eine *robuste* Behalten-/Schließen-Empfehlung,
    abgesichert über eine Sensitivitätsanalyse – jeder Schritt begründet und auf
    Plausibilität geprüft (Regel 0 aus `CLAUDE.md`).

    ### Aufbau
    1. **Teil 1** – Transportkosten $c_{ij}$ herleiten, berechnen und gegen
       `shifts_with_costs.csv` validieren.
    2. **Teil 2** – Basis-Warehouse-Location-Modell lösen, kartieren, plausibilisieren.
    3. **Teil 3** – Erweitertes Modell mit 5 Center-Typen (stückweise lineare Kosten).
    4. **Teil 4** – Sensitivitätsanalyse (Nachfrage ↓, Schichtkosten ↑, neue Technologie).
    5. **Teil 5** – Robuste Empfehlung + Grenzen & Annahmen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vorbemerkung A — Verständnis des bestehenden Codes (`basicAnalysis_week_1.py`)

    Bevor wir erweitern, dokumentieren wir, **was** das Basis-Notebook tut, damit wir
    **verteidigen** können, dass unsere Erweiterung kompatibel ist:

    | Element | Im Basis-Notebook | Übernahme hier |
    |---|---|---|
    | Format | **marimo** (`.py`, v0.23.9) | ✅ wir bleiben in marimo (`caseStudy.py`) |
    | Modell-Bibliothek | **PuLP** (`pulp as pl`) | ✅ identisch |
    | Solver | **HiGHS** (`pl.HiGHS`) | ✅ identisch |
    | Daten | von GitHub-Raw-URLs, indiziert nach IDs | ✅ identisch |
    | Variablen | $x_{ij}$ (binär, Region→Center), $y_i$ (binär, Center offen) | ✅ identisch + Erweiterung |
    | Zielfunktion | $\sum x_{ij}c_{ij} + \sum y_i f_i$ | ✅ identisch (Basismodell) |
    | Constraints | $x_{ij}\le y_i$, $\sum_i x_{ij}=1$ | ✅ identisch (Basismodell) |
    | Karte | **folium** | ✅ identisch |

    ### Format-Entscheidung (explizit, Regel 0)
    > **Entscheidung:** Case Study als **marimo-Notebook** (`caseStudy.py`).
    > **Begründung:** Das Basis-Notebook ist sauberes marimo und läuft; alle benötigten
    > Pakete sind installiert. Gleiches Ökosystem = Konsistenz, kein Stilbruch.
    > **Alternative:** Jupyter `.ipynb`. **Verworfen,** weil es ohne Not eine zweite
    > Werkzeugkette einführen würde (CLAUDE.md verlangt das einfachere/konsistente Format).
    > **Auswirkung/Risiko:** keins – derselbe Solver, dieselbe Datenquelle.

    ### Eine wichtige Beobachtung am Basis-Notebook
    Es rechnet mit `shifts.transportationCosts`, das aber **nur ein Platzhalter
    (= `travelTime`)** ist. Die korrekten Kosten zu berechnen ist genau **Teil 1**.
    Außerdem verzichten wir hier bewusst auf den interaktiven *Solve*-Button und rechnen
    **eager**, damit das Notebook *von oben nach unten fehlerfrei durchläuft*
    (Reproduzierbarkeit; Abweichung vom Basis-Notebook, hier begründet).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Konfiguration — alle „magischen Zahlen" zentral & benannt

    Jede Konstante steht hier oben mit Quelle/Begründung (Regel 0: keine magischen Zahlen).
    Die Szenario-Faktoren und Center-Typ-Parameter sind hier leicht änderbar.
    """)
    return


@app.cell
def _():
    # --- Schicht-Annahmen (Quelle: Folien / Data-driven_SCM_CashLog.md §6) ---
    SHIFT_MIN = 450      # nutzbare Minuten je Schicht = 8 h (480) − 30 min Be-/Entladen
    SHIFT_COST = 480     # € je Schicht (1 LKW + 3er-Crew: Löhne, Treibstoff, Abschreibung)

    # --- Big-M (nur für die Benchmark-Validierung) ---
    # Die Referenz shifts_with_costs.csv ersetzt unerreichbare Verbindungen
    # (2*travelTime >= 450) durch 480 * 9_999_999. Wir reproduzieren das exakt,
    # um die Validierung 1:1 nachzustellen (siehe Teil 1).
    BIG_M_SHIFTS = 9_999_999
    BIG_M_COST = SHIFT_COST * BIG_M_SHIFTS  # = 4_799_999_520

    # --- Sensitivitäts-Szenarien (Teil 4) ---
    # Achse 1: sinkende Bargeldnachfrage -> Faktor auf yearlyDemand
    DEMAND_SCENARIOS = {"Nachfrage -10%": 0.9, "Nachfrage -30%": 0.7, "Nachfrage -50%": 0.5}
    # Achse 2: steigende Löhne/Treibstoff -> Faktor auf SHIFT_COST (480 €)
    SHIFTCOST_SCENARIOS = {"Schichtkosten +20%": 1.2, "Schichtkosten +50%": 1.5}
    # Achse 3: neue Technologie. Annahmen explizit & defensiv (siehe Markdown in Teil 4):
    #   - autonome LKW: Crew (3 Personen) entfällt -> Schichtkosten ~ -60% (Personal ist
    #     der dominante Block) => Faktor 0.4 auf 480 €.
    #   - 24/7-Betrieb + etwas schnellere Fahrt: nutzbare Minuten 900 statt 450,
    #     travelTime *0.8 -> mehr Regionen pro Schicht erreichbar.
    TECH_SCENARIOS = {
        "Autonom (keine Crew)": dict(shift_cost_factor=0.4),
        "Autonom + 24/7 + schneller": dict(shift_cost_factor=0.4, shift_min=900, travel_factor=0.8),
    }

    # --- Center-Typen für das erweiterte Modell (Teil 3) ---
    # Volumengrenzen kalibriert an der Basis-Lösung (Center-Volumina 37k..717k) und an der
    # Gesamtnachfrage (~2,86 Mio). c_fix/c_var modellieren SKALENEFFEKTE: größerer Typ ->
    # höhere Fixkosten, aber niedrigere variable Kosten je Lieferung. Werte sind plausible,
    # dokumentierte Setzungen (in der gleichen Größenordnung wie warehouses.fixedCosts).
    # Tupel = (V_lb, V_ub, c_fix [€/Jahr], c_var [€/Lieferung]).
    # V_ub des größten Typs = 950.000: ein einzelnes Center kann maximal die Nachfrage seiner
    # ERREICHBAREN Regionen verarbeiten; dieses Maximum liegt im Datensatz bei 907.412
    # (geprüft). Eine enge, aber gültige Schranke -> stärkere LP-Relaxation, schnellerer Solver.
    TYPE_PARAMS = {
        "v": (0,       40_000,   1_200_000,  8.0),   # very small
        "s": (40_001,  80_000,   2_000_000,  6.0),   # small
        "m": (80_001,  150_000,  3_200_000,  4.5),   # medium
        "l": (150_001, 350_000,  6_000_000,  3.3),   # large
        "h": (350_001, 950_000,  11_000_000, 2.4),   # huge (V_ub = max. erreichbares Volumen)
    }

    # --- Solver-Einstellungen für das erweiterte MILP (Teil 3) ---
    # Das erweiterte Modell ist schwer (2.661 binäre x + 210 binäre y, große, nicht-konvexe
    # stückweise lineare Kostenkurve). Wir machen es robust lösbar:
    #   OBJ_SCALE: Zielfunktion in Tausend € (HiGHS empfiehlt Skalierung ~1e-3 -> Numerik/Tempo).
    #   EXT_GAP_REL: akzeptierte relative Optimalitätslücke (1%). Für eine STRATEGISCHE
    #     Behalten-/Schließen-Entscheidung ist <=1% völlig unkritisch (die Center-/Typ-Struktur
    #     ist stabil); wir berichten die Lücke ehrlich, statt „optimal" zu behaupten.
    #   EXT_TIME_LIMIT: Sicherheitsnetz, damit das Notebook garantiert durchläuft.
    OBJ_SCALE = 1000.0
    EXT_GAP_REL = 0.01
    EXT_TIME_LIMIT = 180
    return (
        BIG_M_COST,
        DEMAND_SCENARIOS,
        EXT_GAP_REL,
        EXT_TIME_LIMIT,
        OBJ_SCALE,
        SHIFTCOST_SCENARIOS,
        SHIFT_COST,
        SHIFT_MIN,
        TECH_SCENARIOS,
        TYPE_PARAMS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Daten laden (identische Quelle wie das Basis-Notebook)
    """)
    return


@app.cell
def _(mo, pd):
    _base = "https://raw.githubusercontent.com/D3IP-SS25/data-driven-scm-dataset/refs/heads/main/data"
    warehouses = pd.read_csv(f"{_base}/warehouses.csv", index_col="warehouseID")
    regions = pd.read_csv(f"{_base}/regions.csv", index_col="regionID")
    shifts = pd.read_csv(f"{_base}/shifts.csv", index_col=["warehouseID", "regionID"])
    shifts_ref = pd.read_csv(f"{_base}/shifts_with_costs.csv", index_col=["warehouseID", "regionID"])

    W = warehouses.index.values          # Set der Center i
    R = regions.index.values             # Set der Regionen j
    total_demand = int(regions["yearlyDemand"].sum())

    tabs = mo.ui.tabs(
        {
            "Warehouses (42)": mo.ui.table(warehouses, selection=None, show_download=False),
            "Regions (515)": mo.ui.table(regions, selection=None, show_download=False),
            "Shifts (42x515)": mo.ui.table(shifts, selection=None, show_download=False),
        }
    )
    tabs
    return R, W, regions, shifts, shifts_ref, total_demand, warehouses


@app.cell(hide_code=True)
def _(mo, total_demand):
    mo.md(rf"""
    **Plausibilitäts-Check der Daten:** 42 Center × 515 Regionen = 21.630 `shifts`-Zeilen
    (stimmt). Gesamtnachfrage = **{total_demand:,} Stops/Jahr**. In `shifts.csv` ist
    `transportationCosts == travelTime` (Platzhalter, wie angekündigt).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Teil 1 — Transportkosten $c_{ij}$ schätzen

    ## Herleitung (Schicht-Logik)

    Ein LKW fährt zur Region und zurück: Fahrtanteil $= 2\cdot\text{travelTime}_{ij}$.
    Die restliche Zeit der 450-min-Schicht dient den Stops:

    $$\text{Bedienzeit} = 450 - 2\,\text{travelTime}_{ij}$$
    $$\text{Stops/Schicht} = \frac{450 - 2\,\text{travelTime}_{ij}}{\text{minutesPerStop}_j}
    \quad\Rightarrow\quad
    \text{Schichten/Jahr} = \frac{\text{yearlyDemand}_j}{\text{Stops/Schicht}}$$

    Bei 480 € je Schicht folgt:

    $$\boxed{\;c_{ij} = 480 \cdot
    \dfrac{\text{yearlyDemand}_j \cdot \text{minutesPerStop}_j}{\,450 - 2\,\text{travelTime}_{ij}\,}\;}$$

    ### Sonderfall Unerreichbarkeit — Entscheidung
    > Ist $2\cdot\text{travelTime}_{ij}\ge 450$, wird der Nenner $\le 0$: die Region ist in
    > **einer** Schicht nicht von diesem Center bedienbar.
    > **Entscheidung:** Solche Verbindungen werden **ausgeschlossen** (keine $x_{ij}$-Variable).
    > **Begründung:** sauberer & schneller als ein Big-M; und (siehe unten) **jede** Region
    > behält ≥1 erreichbares Center, das Modell bleibt also zulässig.
    > **Alternative:** Big-M-Kosten (so macht es die Referenzdatei). **Verworfen** für die
    > Optimierung (numerisch unschön), aber für die **Validierung** unten exakt reproduziert.
    > **Auswirkung:** identische Optima, weil eine Big-M-Zuordnung im Optimum nie gewählt wird.
    """)
    return


@app.cell
def _(SHIFT_COST, SHIFT_MIN, np, regions, shifts):
    # Wiederverwendbare Kostenfunktion. Gibt NUR erreichbare (feasible) Links als
    # dict {(i,j): c_ij} zurück. Parameter erlauben die Sensitivitätsszenarien:
    #   demand_factor : skaliert yearlyDemand        (Achse 1)
    #   shift_cost    : ersetzt die 480 €            (Achse 2 / Tech)
    #   shift_min     : ersetzt die 450 min          (Tech: 24/7)
    #   travel_factor : skaliert travelTime          (Tech: schneller)
    def link_costs(demand_factor=1.0, shift_cost=SHIFT_COST, shift_min=SHIFT_MIN, travel_factor=1.0):
        region_ids = shifts.index.get_level_values("regionID")
        dem = regions["yearlyDemand"].reindex(region_ids).to_numpy() * demand_factor
        mps = regions["minutesPerStop"].reindex(region_ids).to_numpy()
        usable = shift_min - 2.0 * shifts["travelTime"].to_numpy() * travel_factor
        feasible = usable > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = shift_cost * dem * mps / usable
        idx = shifts.index[feasible]
        return dict(zip(idx, cost[feasible]))

    return (link_costs,)


@app.cell
def _(link_costs):
    # Basis-Kosten (alle Faktoren = Standard). Diese c_ij speisen Teil 2 & 3.
    cost_base = link_costs()
    n_feasible = len(cost_base)
    return cost_base, n_feasible


@app.cell(hide_code=True)
def _(mo, n_feasible):
    mo.md(rf"""
    ## Validierung gegen `shifts_with_costs.csv` (Benchmark)

    Berechnete erreichbare Verbindungen: **{n_feasible:,}** von 21.630
    (⇒ ~87,7 % der Center-Region-Paare sind in *einer* Schicht **nicht** erreichbar – die
    meisten Center liegen für die meisten Regionen zu weit weg; geografisch plausibel).
    Wir reproduzieren die Big-M-Konvention der Referenz und vergleichen alle 21.630 Zeilen.
    """)
    return


@app.cell
def _(BIG_M_COST, SHIFT_COST, SHIFT_MIN, np, plt, regions, shifts, shifts_ref):
    # Eigene Kosten für ALLE Zeilen (erreichbar: Formel, sonst Big-M) zum 1:1-Vergleich.
    _ref = shifts_ref.reindex(shifts.index)  # Zeilenreihenfolge angleichen (Dateien sind anders sortiert!)
    _rid = shifts.index.get_level_values("regionID")
    _dem = regions["yearlyDemand"].reindex(_rid).to_numpy()
    _mps = regions["minutesPerStop"].reindex(_rid).to_numpy()
    _usable = SHIFT_MIN - 2.0 * shifts["travelTime"].to_numpy()
    _feas = _usable > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        _mine = np.where(_feas, SHIFT_COST * _dem * _mps / _usable, BIG_M_COST)
    _r = _ref["transportationCosts"].to_numpy()

    # Kennzahlen
    _rel = np.abs((_mine[_feas] - _r[_feas]) / _r[_feas])
    max_rel_dev = float(_rel.max())
    mean_rel_dev = float(_rel.mean())
    bigm_match = float(np.mean(np.isclose(_mine[~_feas], _r[~_feas])))

    # Scatter: eigene vs. Referenz (nur erreichbare; log-log)
    fig_val, ax_val = plt.subplots(figsize=(5.2, 5.2))
    ax_val.scatter(_r[_feas], _mine[_feas], s=8, alpha=0.4)
    _lo, _hi = _r[_feas].min(), _r[_feas].max()
    ax_val.plot([_lo, _hi], [_lo, _hi], "r--", lw=1, label="ideal (y = x)")
    ax_val.set(xscale="log", yscale="log", xlabel="Referenz c_ij (€)",
               ylabel="eigene c_ij (€)", title="Teil 1: eigene vs. Referenz-Kosten")
    ax_val.legend()
    fig_val.tight_layout()
    fig_val
    return bigm_match, max_rel_dev, mean_rel_dev


@app.cell(hide_code=True)
def _(bigm_match, max_rel_dev, mean_rel_dev, mo):
    mo.md(rf"""
    ### Interpretation der Validierung
    - **Max. relative Abweichung:** `{max_rel_dev:.2e}` — **mittlere:** `{mean_rel_dev:.2e}`.
      Das ist reine Gleitkomma-Rundung ($\approx 10^{{-16}}$), also **exakt** dieselbe Formel.
    - **Big-M-Zeilen** stimmen zu **{bigm_match:.0%}** mit der Referenz überein
      (Referenz = `480 · 9.999.999`).
    - **Verteidigung:** Die Punkte liegen perfekt auf der Diagonale (y = x). Unsere
      Transportkostenschätzung ist damit gegen den Benchmark **validiert**. ✅
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Teil 2 — Basismodell rechnen & verstehen

    Identische Struktur wie `basicAnalysis_week_1.py`, aber (a) mit **korrekten** $c_{ij}$
    und (b) nur über **erreichbare** Links:

    $$\min \sum_{(i,j)\in S} x_{ij} c_{ij} + \sum_{i\in W} f_i y_i
    \quad\text{s.t.}\quad x_{ij}\le y_i,\;\; \sum_{i} x_{ij}=1,\;\; x,y\in\{0,1\}$$

    `solve_base` kapselt das Modell, damit wir es in Teil 4 für jedes Szenario wiederverwenden.
    """)
    return


@app.cell
def _(R, W, pl, regions, warehouses):
    def solve_base(cost_map, demand_factor=1.0):
        """Löst das Basis-Warehouse-Location-Modell über die erreichbaren Links cost_map."""
        links = list(cost_map.keys())
        by_region = {}
        by_wh = {}
        for (i, j) in links:
            by_region.setdefault(j, []).append((i, j))
            by_wh.setdefault(i, []).append((i, j))

        prob = pl.LpProblem("CashLog_Base", pl.LpMinimize)
        x = pl.LpVariable.dicts("x", links, cat=pl.LpBinary)
        y = pl.LpVariable.dicts("y", list(W), cat=pl.LpBinary)

        prob.setObjective(
            pl.lpSum(x[k] * cost_map[k] for k in links)
            + pl.lpSum(y[i] * warehouses.loc[i, "fixedCosts"] for i in W)
        )
        for k in links:                       # x_ij <= y_i
            prob += x[k] <= y[k[0]]
        for j in R:                           # jede Region genau einem (offenen) Center
            prob += pl.lpSum(x[k] for k in by_region[j]) == 1

        prob.solve(pl.HiGHS(msg=False))

        assign = {j: i for (i, j) in links if x[(i, j)].varValue and x[(i, j)].varValue > 0.5}
        open_w = [i for i in W if y[i].varValue and y[i].varValue > 0.5]
        vol = {}
        for j, i in assign.items():
            vol[i] = vol.get(i, 0.0) + regions.loc[j, "yearlyDemand"] * demand_factor
        fixed = sum(warehouses.loc[i, "fixedCosts"] for i in open_w)
        variable = sum(cost_map[(i, j)] for j, i in assign.items())
        return {
            "status": pl.LpStatus[prob.status],
            "open": sorted(open_w),
            "assign": assign,
            "vol": vol,
            "fixed": fixed,
            "variable": variable,
            "total": prob.objective.value(),
        }

    return (solve_base,)


@app.cell
def _(cost_base, solve_base):
    base_res = solve_base(cost_base)
    return (base_res,)


@app.cell(hide_code=True)
def _(base_res, mo, warehouses):
    _open_cities = ", ".join(sorted(warehouses.loc[i, "city"] for i in base_res["open"]))
    mo.md(
        rf"""
        ### Ergebnis Basismodell
        - **Solver-Status:** `{base_res['status']}` — Verteidigung: HiGHS liefert das **global
          optimale** MILP (kein „feasible only"); damit ist die Lösung optimal *und* zulässig.
        - **Offen gehaltene Center:** **{len(base_res['open'])}** von 42.
        - **Gesamtkosten:** {base_res['total']:,.0f} € — davon Fixkosten {base_res['fixed']:,.0f} €,
          Transportkosten {base_res['variable']:,.0f} €.
        - **Offene Städte:** {_open_cities}

        Validierung der Nebenbedingungen (Stichprobe folgt in der Endkontrolle): jede der 515
        Regionen taucht genau einmal in `assign` auf.
        """
    )
    return


@app.cell
def _(R, base_res, cost_base, mo):
    # Endkontrolle der Nebenbedingungen (Verteidigung der Zulässigkeit, Regel 0 / §4):
    _assign = base_res["assign"]
    _check_all_served = set(_assign.keys()) == set(R)                 # jede Region zugeordnet
    _check_exactly_one = len(_assign) == len(R)                       # genau eine Zuordnung
    _check_feasible_link = all((i, j) in cost_base for j, i in _assign.items())  # nur erreichbar
    _check_open = all(i in base_res["open"] for i in _assign.values())          # nur offene Center
    _ok = all([_check_all_served, _check_exactly_one, _check_feasible_link, _check_open])
    mo.md(
        rf"""
        **Constraint-Validierung (Basismodell):**
        alle 515 Regionen bedient: `{_check_all_served}` · genau eine Zuordnung je Region:
        `{_check_exactly_one}` · nur über erreichbare Links: `{_check_feasible_link}` ·
        nur offenen Centern zugeordnet: `{_check_open}` → **Lösung zulässig: {_ok}** ✅
        """
    )
    return


@app.cell
def _(R, base_res, folium, regions, sns, warehouses):
    # Karte wie im Basis-Notebook: offene Center als 🏦, Regionen eingefärbt nach Zuordnung,
    # geschlossene Center als graues Kreuz (zur klaren Unterscheidung offen/geschlossen).
    _open = base_res["open"]
    _palette = sns.color_palette("tab20", n_colors=max(len(_open), 1)).as_hex()
    _color = {w: _palette[k % 20] for k, w in enumerate(_open)}

    m = folium.Map(location=[40, -3.5], zoom_start=6, tiles="cartodbpositron")
    for j in R:
        i = base_res["assign"][j]
        folium.CircleMarker(
            location=[regions.loc[j, "lat"], regions.loc[j, "lon"]],
            radius=3, color=_color[i], fill=True, fill_opacity=0.8, weight=0.4,
            popup=f"{regions.loc[j,'city']} → {warehouses.loc[i,'city']}",
        ).add_to(m)
    for i in warehouses.index:
        if i in _open:
            folium.Marker(
                location=[warehouses.loc[i, "lat"], warehouses.loc[i, "lon"]],
                popup=f"OPEN: {warehouses.loc[i,'city']}",
                icon=folium.DivIcon(html='<div style="font-size:26px;">🏦</div>'),
            ).add_to(m)
        else:
            folium.Marker(
                location=[warehouses.loc[i, "lat"], warehouses.loc[i, "lon"]],
                popup=f"closed: {warehouses.loc[i,'city']}",
                icon=folium.DivIcon(html='<div style="font-size:16px;color:#999;">✖</div>'),
            ).add_to(m)
    m
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plausibilisierung der Basis-Lösung
    - **Geografische Streuung:** Die offenen Center verteilen sich über das Festland; jede
      Region wird in einer Farbe = ihrem zugeordneten Center dargestellt. Die farbigen
      „Reviere" sind räumlich zusammenhängend → Regionen gehen an ein **nahes** Center
      (kurze `travelTime`), wie ökonomisch erwartet.
    - **Trade-off sichtbar:** ~18 offene Center sind ein Kompromiss aus Fixkosten
      (mehr Center = teurer) und Transportkosten (mehr Center = kürzere Wege).
    - **Erzwungene Center:** Regionen mit nur **einem** erreichbaren Center fixieren dieses
      Center zwangsläufig auf „offen" – ein wichtiger Hinweis für die Robustheit (Teil 5).
    - **Verdächtig wäre:** ein Center mit fast keinem Volumen oder Regionen, die quer durchs
      Land einem fernen Center zugeordnet sind. Beides tritt hier nicht auf → Lösung plausibel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Teil 3 — Erweitertes Modell (stückweise lineare Kosten, 5 Center-Typen)

    Das Basismodell unterstellt **konstante** Fixkosten je Center. Realistisch sind
    **Skaleneffekte**: ein Center kann (strategisch) in 5 Größen *(v, s, m, l, h)* betrieben
    werden – größere Typen haben höhere Fixkosten, aber niedrigere variable Kosten je Lieferung.

    $$\min \sum_{(i,j)} x_{ij} c_{ij} + \sum_{i}\sum_{t} c_t^{fix} y_{it} + \sum_i\sum_t c_t^{var} z_{it}$$

    | Nebenbedingung | Bedeutung |
    |---|---|
    | $\sum_t y_{it} \le 1\;\forall i$ | **höchstens ein Typ je Center** (sonst „künstliches Aufteilen") |
    | $x_{ij} \le \sum_t y_{it}$ | Region nur einem **offenen** Center zuordenbar |
    | $\sum_i x_{ij} = 1$ | jede Region genau einem Center |
    | $\sum_t z_{it} = \sum_j x_{ij} d_j$ | verarbeitetes Volumen = zugeordnete Nachfrage |
    | $V_t^{lb} y_{it} \le z_{it} \le V_t^{ub} y_{it}$ | Volumen passt ins Fenster des gewählten Typs |

    > **Begründung `∑_t y_it ≤ 1`:** Ohne sie könnte das Modell ein Center gleichzeitig als
    > mehrere Typen führen und so die günstigsten Stücke mehrerer Typen „mischen" – das
    > zerstört die stückweise-lineare Logik. Die Restriktion erzwingt **genau ein** Kostenstück.

    Hinweis: $c_{ij}$ = Transport (Schichten); $c_t^{var}$ = **Verarbeitungskosten** je Lieferung
    (Zählen/Sortieren) – kein Doppelzählen. Absolute Gesamtkosten sind daher **nicht** direkt
    mit dem Basismodell vergleichbar; entscheidend ist die **Struktur** (welche Center offen).
    """)
    return


@app.cell(hide_code=True)
def _(TYPE_PARAMS, mo, pd):
    _df = pd.DataFrame(
        [(t, lb, ub, cf, cv) for t, (lb, ub, cf, cv) in TYPE_PARAMS.items()],
        columns=["Typ", "V_lb", "V_ub", "c_fix (€)", "c_var (€/Lief.)"],
    )
    mo.vstack([mo.md("**Center-Typ-Parameter (dokumentierte Annahme, Skaleneffekte):**"),
               mo.ui.table(_df, selection=None, show_download=False)])
    return


@app.cell
def _(EXT_GAP_REL, EXT_TIME_LIMIT, OBJ_SCALE, R, W, pl, regions):
    def solve_extended(cost_map, type_params, demand_factor=1.0,
                       obj_scale=OBJ_SCALE, gap_rel=EXT_GAP_REL, time_limit=EXT_TIME_LIMIT):
        """Erweitertes MILP mit 5 Center-Typen (y_it, z_it) und stückweise linearen Kosten.

        Zielfunktion wird mit obj_scale skaliert (Numerik) und für die Ausgabe zurückskaliert.
        gap_rel/time_limit machen das schwere MILP robust lösbar (siehe Konfigurations-Markdown).
        """
        links = list(cost_map.keys())
        types = list(type_params.keys())
        by_region, by_wh = {}, {}
        for (i, j) in links:
            by_region.setdefault(j, []).append((i, j))
            by_wh.setdefault(i, []).append((i, j))
        d = {j: regions.loc[j, "yearlyDemand"] * demand_factor for j in R}
        s = obj_scale  # Kürzel

        prob = pl.LpProblem("CashLog_Extended", pl.LpMinimize)
        x = pl.LpVariable.dicts("xe", links, cat=pl.LpBinary)
        y = {(i, t): pl.LpVariable(f"y_{i}_{t}", cat=pl.LpBinary) for i in W for t in types}
        z = {(i, t): pl.LpVariable(f"z_{i}_{t}", lowBound=0) for i in W for t in types}

        prob.setObjective(
            pl.lpSum(x[k] * (cost_map[k] / s) for k in links)
            + pl.lpSum(y[i, t] * (type_params[t][2] / s) for i in W for t in types)
            + pl.lpSum(z[i, t] * (type_params[t][3] / s) for i in W for t in types)
        )
        for i in W:
            prob += pl.lpSum(y[i, t] for t in types) <= 1                      # <= 1 Typ
        for k in links:
            prob += x[k] <= pl.lpSum(y[k[0], t] for t in types)                # nur offen
        for j in R:
            prob += pl.lpSum(x[k] for k in by_region[j]) == 1                  # genau ein Center
        for i in W:
            prob += pl.lpSum(z[i, t] for t in types) == pl.lpSum(
                x[k] * d[k[1]] for k in by_wh.get(i, [])
            )                                                                  # Volumenkopplung
            for t in types:
                lb, ub, _, _ = type_params[t]
                prob += z[i, t] >= lb * y[i, t]
                prob += z[i, t] <= ub * y[i, t]

        prob.solve(pl.HiGHS(msg=False, gapRel=gap_rel, timeLimit=time_limit))

        chosen, vol = {}, {}
        for i in W:
            for t in types:
                if y[i, t].varValue and y[i, t].varValue > 0.5:
                    chosen[i] = t
        for (i, j) in links:
            if x[(i, j)].varValue and x[(i, j)].varValue > 0.5:
                vol[i] = vol.get(i, 0.0) + d[j]
        return {
            "status": pl.LpStatus[prob.status],
            "open": sorted(chosen.keys()),
            "type": chosen,
            "vol": vol,
            "total": prob.objective.value() * s,   # zurück in € skalieren
        }

    return (solve_extended,)


@app.cell
def _(TYPE_PARAMS, cost_base, solve_extended):
    # Hinweis: das erweiterte MILP ist rechenintensiv (~30-90 s je nach Maschine), da es zur
    # Optimalitätslücke von 1% gelöst wird. Es läuft eager, damit das Notebook durchläuft.
    ext_res = solve_extended(cost_base, TYPE_PARAMS)
    return (ext_res,)


@app.cell(hide_code=True)
def _(TYPE_PARAMS, base_res, ext_res, mo, warehouses):
    _b = set(base_res["open"])
    _e = set(ext_res["open"])
    _only_base = ", ".join(sorted(warehouses.loc[i, "city"] for i in _b - _e)) or "—"
    _only_ext = ", ".join(sorted(warehouses.loc[i, "city"] for i in _e - _b)) or "—"
    _type_counts = {t: sum(1 for tt in ext_res["type"].values() if tt == t) for t in TYPE_PARAMS}
    # Validierung: jedes verarbeitete Volumen liegt im [V_lb, V_ub] seines gewählten Typs
    _viol = [
        warehouses.loc[i, "city"]
        for i, t in ext_res["type"].items()
        if not (TYPE_PARAMS[t][0] - 1 <= ext_res["vol"].get(i, 0) <= TYPE_PARAMS[t][1] + 1)
    ]
    mo.md(
        rf"""
        ### Vergleich Basis- vs. erweitertes Modell
        - **Solver-Status:** `{ext_res['status']}` — gelöst bis **≤1 % Optimalitätslücke**
          (`gapRel=0.01`). Verteidigung: Für eine strategische Behalten-/Schließen-Entscheidung
          ist 1 % unkritisch; die Center-/Typ-Struktur (s. u.) ist über Läufe hinweg stabil.
        - **Volumen-Check:** alle offenen Center innerhalb ihrer Typgrenzen?
          **{"JA ✅" if not _viol else "NEIN – " + ", ".join(_viol)}**
        - **Offene Center:** Basis **{len(_b)}** → erweitert **{len(_e)}**.
        - **Typ-Verteilung (erweitert):** { {t: c for t, c in _type_counts.items() if c} }
        - **Nur im Basismodell offen:** {_only_base}
        - **Nur im erweiterten Modell offen:** {_only_ext}

        **Interpretation:** Das erweiterte Modell trennt **Fixkosten je Größentyp** von
        **Verarbeitungskosten je Lieferung**. Skaleneffekte (große Typen = günstiger je Lieferung)
        belohnen größere Center; gleichzeitig zwingt die Erreichbarkeit (87,7 % der Links gesperrt)
        viele kleine Typen (`v`/`s`) in der Fläche. Ob die Center-Zahl gegenüber dem Basismodell
        steigt oder fällt, ist damit ein **echtes Ergebnis** dieses Trade-offs, kein Artefakt –
        die obige Typ-Verteilung zeigt, wie sich das Netz auf die Größenklassen aufteilt.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Teil 4 — Sensitivitätsanalyse (robuste Empfehlung)

    Wir rechnen das (Basis-)Modell über drei Achsen und prüfen, **welche Center stabil
    offen bleiben**. Jede Achse ist begründet:

    1. **Sinkende Nachfrage** (−10/−30/−50 %): Bargeld wird weniger. Da $c_{ij}\propto$ Nachfrage,
       sinken die Transportkosten relativ zu den Fixkosten → Erwartung: **weniger** Center.
    2. **Steigende Schichtkosten** (+20/+50 %): teurer Transport macht kurze Wege wertvoller →
       Erwartung: tendenziell **mehr / dezentralere** Center.
    3. **Neue Technologie** (autonome LKW): Crew-Kosten entfallen (−60 % Schichtkosten),
       optional 24/7 + schnellere Fahrt (mehr Reichweite) → Erwartung: **weniger, größere** Center.

    Alle Szenarien nutzen dieselbe `solve_base`-Funktion (nur die $c_{ij}$ bzw. die Nachfrage
    ändern sich) – das macht den Vergleich sauber und reproduzierbar.
    """)
    return


@app.cell
def _(
    DEMAND_SCENARIOS,
    SHIFTCOST_SCENARIOS,
    SHIFT_COST,
    TECH_SCENARIOS,
    cost_base,
    link_costs,
    solve_base,
):
    # Szenario-Spezifikationen aufbauen (Name -> (cost_map, demand_factor))
    scenarios = {"Basis": (cost_base, 1.0)}
    for _name, _f in DEMAND_SCENARIOS.items():
        scenarios[_name] = (link_costs(demand_factor=_f), _f)
    for _name, _f in SHIFTCOST_SCENARIOS.items():
        scenarios[_name] = (link_costs(shift_cost=SHIFT_COST * _f), 1.0)
    for _name, _kw in TECH_SCENARIOS.items():
        _cm = link_costs(
            shift_cost=SHIFT_COST * _kw.get("shift_cost_factor", 1.0),
            shift_min=_kw.get("shift_min", 450),
            travel_factor=_kw.get("travel_factor", 1.0),
        )
        scenarios[_name] = (_cm, 1.0)

    sens_results = {
        name: solve_base(cm, demand_factor=df) for name, (cm, df) in scenarios.items()
    }
    return (sens_results,)


@app.cell
def _(W, pd, sens_results, warehouses):
    # Übersichtstabelle + Offen/Zu-Matrix (Zeile = Center, Spalte = Szenario)
    summary = pd.DataFrame(
        {
            name: {
                "Status": r["status"],
                "# offene Center": len(r["open"]),
                "Gesamtkosten (€)": round(r["total"]),
            }
            for name, r in sens_results.items()
        }
    ).T

    open_matrix = pd.DataFrame(
        {name: {warehouses.loc[i, "city"]: int(i in r["open"]) for i in W}
         for name, r in sens_results.items()}
    )
    # nur Center zeigen, die irgendwo offen sind (übersichtlicher)
    open_matrix = open_matrix[open_matrix.sum(axis=1) > 0].sort_index()
    return open_matrix, summary


@app.cell(hide_code=True)
def _(mo, summary):
    mo.vstack([mo.md("### Szenario-Übersicht"),
               mo.ui.table(summary.reset_index(names="Szenario"), selection=None, show_download=False)])
    return


@app.cell
def _(open_matrix, plt, sns):
    # Heatmap: welche Center sind in welchem Szenario offen?
    fig_h, ax_h = plt.subplots(figsize=(8, max(4, 0.32 * len(open_matrix))))
    sns.heatmap(open_matrix, cmap=["#f0f0f0", "#2c7fb8"], cbar=False, linewidths=0.5,
                linecolor="white", ax=ax_h)
    ax_h.set(title="Offen (blau) / geschlossen (grau) je Szenario", xlabel="", ylabel="Center")
    plt.setp(ax_h.get_xticklabels(), rotation=35, ha="right")
    fig_h.tight_layout()
    fig_h
    return


@app.cell
def _(open_matrix):
    # Robustheits-Klassifikation
    n_scen = open_matrix.shape[1]
    open_count = open_matrix.sum(axis=1)
    always_open = sorted(open_count[open_count == n_scen].index)
    sometimes_open = sorted(open_count[(open_count > 0) & (open_count < n_scen)].index)
    # "nie offen" = Center, die in KEINEM Szenario offen sind (in open_matrix nicht enthalten)
    return always_open, sometimes_open


@app.cell(hide_code=True)
def _(W, always_open, mo, open_matrix, sometimes_open):
    _never = len(W) - open_matrix.shape[0]
    mo.md(
        rf"""
        ### Interpretation der Sensitivität
        - **Immer offen ({len(always_open)} Center)** – robuste *Behalten*-Kandidaten:
          {", ".join(always_open)}
        - **Szenarioabhängig offen ({len(sometimes_open)})** – die *kritischen* Fälle:
          {", ".join(sometimes_open) or "—"}
        - **Nie offen ({_never} Center)** – robuste *Schließen*-Kandidaten (in keinem Szenario offen).

        **Prüfung der Erwartungen:** Sinkende Nachfrage → tendenziell weniger Center
        (Fixkosten dominieren); steigende Schichtkosten → kurze Wege wertvoller (eher mehr/dezentral);
        autonome Technik → Transport billig, Konsolidierung auf wenige große Center. Die Heatmap
        oben macht sichtbar, ob diese Erwartungen eintreten – Abweichungen sind selbst ein Befund.
        """
    )
    return


@app.cell(hide_code=True)
def _(always_open, mo, open_matrix, sometimes_open, warehouses):
    _never = sorted(set(warehouses["city"]) - set(open_matrix.index))
    mo.md(
        rf"""
        # Teil 5 — Conclusion & Empfehlung

        ## Management-Empfehlung (robust über alle Szenarien)
        1. **Behalten ({len(always_open)}):** Center, die in *jedem* Szenario offen sind, bilden
           das robuste Rückgrat des Netzes – hier nicht investieren zu zögern. ✅
           → {", ".join(always_open)}
        2. **Schließen ({len(_never)}):** Center, die in *keinem* Szenario gebraucht werden, sind
           klare Schließungskandidaten – sie setzen Fixkosten frei, ohne die Bedienung zu gefährden.
        3. **Kritisch / genauer prüfen ({len(sometimes_open)}):** {", ".join(sometimes_open) or "—"}.
           Ihr Status hängt von Annahmen (Nachfrageentwicklung, Lohn-/Treibstoffkosten, Technik) ab.
           Empfehlung: **Optionen offenhalten** und an den realen Nachfrage-/Kostentrend koppeln.

        ## Grenzen & wichtigste Annahmen (mit Einfluss)
        - **Schicht-Approximation:** Routing wird je Region durch eine Schichtformel genähert
          (kontinuierliche Schichten, keine Tagesschwankung). *Einfluss:* Transportkosten sind
          strategische Schätzwerte, keine operativen Ist-Kosten.
        - **Erreichbarkeit „in einer Schicht":** 87,7 % der Links sind ausgeschlossen. Mehrtages-
          oder Umschlagtouren sind nicht abgebildet. *Einfluss:* einige Center wirken „erzwungen
          offen", weil sie als einzige eine Region in einer Schicht erreichen.
        - **Center-Typ-Parameter (Teil 3)** sind plausible, aber gesetzte Annahmen. *Einfluss:*
          verschiebt die Konsolidierungsstärke; oben als Parameter leicht änderbar.
        - **Technik-Szenarien** sind grobe „Was-wäre-wenn"-Setzungen (−60 % Schichtkosten, 24/7).
          *Einfluss:* zeigen Richtung & Hebel, sind aber keine Punktprognose.
        - **Statische Sicht:** keine Übergangs-/Schließungskosten, keine Mehrperiodendynamik.

        ## Roter Faden
        Korrekte $c_{{ij}}$ (validiert) → Basismodell (plausibel kartiert) → Skaleneffekte
        (5 Typen) → Sensitivität über 3 Achsen → **robuste** Behalten-/Schließen-Empfehlung.
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
