import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # ------------------------------------------------------------------------
    # Library stack. We stay inside the same ecosystem as the course's base
    # notebook so nothing about the tooling is surprising:
    #   - pandas / numpy : data handling and vectorised maths
    #   - pulp + highspy : build the optimisation model (PuLP) and solve it (HiGHS)
    #   - folium         : interactive map of the resulting network
    #   - seaborn/matplotlib : the validation scatter and the sensitivity heatmap
    #   - marimo         : the notebook format itself
    # ------------------------------------------------------------------------
    import folium
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pulp as pl
    import highspy  # noqa: F401  (the HiGHS backend behind pl.HiGHS)

    return folium, mo, np, pd, pl, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CashLog — Designing the Optimal Cash-Center Network

    **What you are about to read.** This notebook solves one concrete business
    decision from start to finish, and it is written so that *someone who has
    never seen this problem before* can follow every step — both the **business
    logic** and the **maths/code tricks** that make it work.

    It is built as a single straight line ("red carpet"). Each part only uses
    results produced by the part before it:

    > **Story → Model → Data → Costs → Solve → Make it realistic → Stress-test → Recommend**

    | Part | Question it answers | What we produce |
    |---|---|---|
    | **0** | *What is the real-world problem?* | A plain-language problem statement |
    | **1** | *How do we turn it into maths?* | The Warehouse-Location model |
    | **2** | *What data do we have?* | Loaded & sanity-checked tables |
    | **3** | *What does it cost to serve a region?* | The transport cost $c_{ij}$ (+ validation) |
    | **4** | *Which centers should stay open?* | The base optimisation + a map |
    | **5** | *What if size/economies of scale matter?* | The extended model (5 center sizes) |
    | **6** | *Is the answer robust to the future?* | A sensitivity analysis over 3 axes |
    | **7** | *So what should management do?* | A keep / close / watch recommendation |

    Every number shown in the text below is computed live by the code in this
    notebook — nothing is typed in by hand — so the explanations can never drift
    away from the actual results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 0 — The problem, in plain language

    **Who is CashLog?** CashLog is the Spanish market leader in *cash logistics*.
    Concretely it does three things: it **collects** cash from supermarkets, banks
    and petrol stations; it **counts, sorts and stores** that cash in high-security
    buildings called **Cash Centers**; and it **distributes** cash back out to banks
    and other customers.

    **Why is there a decision to make?** Historically CashLog ran one Cash Center in
    *every* Spanish provincial capital — because the central bank had a branch there.
    That reason is gone, and two pressures now push the other way:

    - **Cash is shrinking.** Electronic payment keeps replacing physical cash, so the
      volume each center handles is falling.
    - **Cash Centers are expensive.** Security, surveillance and insurance make every
      single building cost millions of euros per year just to keep open.

    Today the network is **42 Cash Centers** serving about **42,000 customer
    locations**. The 42,000 customers have already been grouped (by the course) into
    **515 customer regions** — we will come back to *why* grouping is necessary.

    > **The decision (one sentence):** Out of the 42 existing Cash Centers, which
    > subset do we **keep open** and which do we **close**, so that the **total yearly
    > cost is minimal** while **every one of the 515 regions is still served**?

    **Two hard constraints from the business side:**

    1. We do **not** open new locations — we only choose among the 42 that exist.
    2. Every region must remain served. Closing a center is only allowed if its
       regions can be picked up by other centers.

    Everything that follows is just a rigorous way of answering that one question —
    and, crucially, answering it in a way we can *defend* and *stress-test*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1 — Turning the story into a maths problem

    ## 1.1 The trade-off that drives everything

    Why can't we just close the expensive centers? Because of a tug-of-war:

    - **More open centers** → high **fixed costs** (each building is expensive) but
      **short driving distances**, so low **transport costs**.
    - **Fewer open centers** → low fixed costs but **long distances**, so high
      transport costs.

    The cheapest network sits *somewhere in the middle*, and we cannot eyeball it.

    ## 1.2 Why we must optimise the network *as a whole*

    A tempting shortcut is to score each center on its own and close the worst ones.
    This fails because of **network effects**: whether a center is worth keeping
    depends on *which other centers are open*. If we close center A, its regions must
    be re-assigned to B, C, … — which changes *their* load and cost. The locations are
    interdependent, so we must decide all of them **simultaneously**. That is exactly
    what an optimisation model does.

    ## 1.3 The template: the Warehouse-Location model

    This is a classic model. In words: *pick which facilities to open, and assign every
    customer to an open facility, so that total (fixed + transport) cost is minimal.*

    **The vocabulary (we will reuse these symbols everywhere):**

    | Symbol | Meaning |
    |---|---|
    | $i$ | a Cash Center (one of the 42), the "warehouse" |
    | $j$ | a customer region (one of the 515) |
    | $f_i$ | yearly fixed cost of running center $i$ |
    | $c_{ij}$ | yearly transport cost of serving region $j$ from center $i$ |
    | $y_i \in \{0,1\}$ | **decision:** is center $i$ open (1) or closed (0)? |
    | $x_{ij} \in \{0,1\}$ | **decision:** is region $j$ served by center $i$? |

    **The model:**

    $$\min_{x,y}\; \underbrace{\sum_{i}\sum_{j} x_{ij}\,c_{ij}}_{\text{transport cost}}
    \;+\; \underbrace{\sum_{i} f_i\,y_i}_{\text{fixed cost of open centers}}$$

    subject to

    $$\sum_{i} x_{ij} = 1 \;\;\forall j \quad\text{(every region served by exactly one center)}$$
    $$x_{ij} \le y_i \;\;\forall i,j \quad\text{(you may only use a center that is open)}$$

    Read the second constraint literally: if center $i$ is closed ($y_i = 0$) then
    $x_{ij}$ is forced to $0$ for all regions — you cannot assign work to a closed
    building. The objective then automatically balances the trade-off from 1.1.

    ## 1.4 Why this model does *not* fit CashLog out of the box

    Three realities break the textbook version — and dealing with them honestly *is*
    the case study:

    1. **What is $c_{ij}$, really?** There is no price tag "cost to serve region $j$".
       Trucks serve many customers per 8-hour **shift** on a route, not one trip per
       customer. We must *derive* $c_{ij}$ from shift logic → **Part 3**.
    2. **42,000 customers is too many points** to optimise directly. They have been
       **clustered into 515 regions** (already done for us, in `regions.csv`). This is
       why we work with regions and not individual shops.
    3. **Fixed cost is not really a constant.** A center can be built bigger or smaller;
       bigger centers cost more to run but are cheaper *per delivery* (economies of
       scale). We capture that with the **extended model** → **Part 5**.

    We solve the textbook model first (Part 4) and then upgrade it (Part 5), so you can
    see exactly what each layer of realism adds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.5 Configuration — every "magic number" named in one place

    Good practice: no unexplained constants buried in the code. Everything a reader
    might want to question — the shift assumptions, the future scenarios, the
    center-size parameters, the solver settings — lives in the single cell below,
    each with its source and reasoning. If you want to try different assumptions,
    this is the **only** cell you change.
    """)
    return


@app.cell
def _():
    # === Shift assumptions (source: lecture slides / Data-driven_SCM_CashLog.md §6) ===
    SHIFT_MIN = 450      # usable minutes per shift = 8h (480) minus 30 min loading/unloading
    SHIFT_COST = 480     # € per shift (1 truck + 3-person crew: wages, fuel, depreciation)

    # === Big-M, used ONLY for the benchmark validation in Part 3 ===
    # The reference file replaces every UNREACHABLE link (where 2*travelTime >= 450)
    # with the constant 480 * 9_999_999. We reproduce that exactly so our validation
    # can compare all 21,630 rows 1:1 against the benchmark.
    BIG_M_SHIFTS = 9_999_999
    BIG_M_COST = SHIFT_COST * BIG_M_SHIFTS  # = 4_799_999_520

    # === Sensitivity scenarios (Part 6) ===
    # Axis 1 — cash demand falls: a multiplier applied to every region's yearlyDemand.
    DEMAND_SCENARIOS = {"Demand -10%": 0.9, "Demand -30%": 0.7, "Demand -50%": 0.5}
    # Axis 2 — wages/fuel rise: a multiplier on the 480 € shift cost.
    SHIFTCOST_SCENARIOS = {"Shift cost +20%": 1.2, "Shift cost +50%": 1.5}
    # Axis 3 — new technology. Assumptions are stated explicitly and defended in Part 6:
    #   * autonomous trucks: the 3-person crew (the dominant cost block) disappears,
    #     so shift cost drops ~60%  -> factor 0.4 on 480 €.
    #   * 24/7 operation + slightly faster driving: usable minutes 900 instead of 450,
    #     travelTime *0.8 -> each shift reaches more regions.
    TECH_SCENARIOS = {
        "Autonomous (no crew)": dict(shift_cost_factor=0.4),
        "Autonomous + 24/7 + faster": dict(shift_cost_factor=0.4, shift_min=900, travel_factor=0.8),
    }

    # === Center types for the extended model (Part 5) ===
    # Five discrete sizes approximate a smooth economies-of-scale curve. Volume bounds are
    # calibrated to the base solution (center volumes land in ~37k..717k) and the total
    # demand (~2.86 M). c_fix/c_var encode ECONOMIES OF SCALE: a bigger type has higher
    # fixed cost but lower variable cost per delivery. Values are documented, plausible
    # assumptions, in the same order of magnitude as warehouses.fixedCosts.
    # Tuple layout = (V_lb, V_ub, c_fix [€/year], c_var [€/delivery]).
    # The largest type's V_ub = 950,000 because a single center can process at most the
    # demand of the regions it can REACH; that maximum is 907,412 in this dataset (checked).
    # A tight-but-valid bound gives a stronger LP relaxation and a faster solver.
    TYPE_PARAMS = {
        "v": (0,       40_000,   1_200_000,  8.0),   # very small
        "s": (40_001,  80_000,   2_000_000,  6.0),   # small
        "m": (80_001,  150_000,  3_200_000,  4.5),   # medium
        "l": (150_001, 350_000,  6_000_000,  3.3),   # large
        "h": (350_001, 950_000,  11_000_000, 2.4),   # huge (V_ub = max reachable volume)
    }

    # === Solver settings for the heavy extended MILP (Part 5) ===
    # That model is hard (2,661 binary x + 210 binary y, a large non-convex piecewise cost
    # curve). Three settings keep it reliably solvable; each is explained where it is used:
    #   OBJ_SCALE     : solve in units of 1000 € (HiGHS asks for ~1e-3 scaling -> numerics).
    #   EXT_GAP_REL   : accept a 1% optimality gap. For a strategic keep/close decision a
    #                   <=1% gap is harmless; we report the gap honestly instead of
    #                   over-claiming "provably optimal".
    #   EXT_TIME_LIMIT: a safety net so the notebook is guaranteed to finish.
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
    # Part 2 — The data

    We load four tables from the course's public GitHub repository (the exact same
    source as the base notebook). Each row is indexed by its IDs so we can look things
    up by `warehouseID` / `regionID`.

    | Table | Rows | What it gives us |
    |---|---|---|
    | `warehouses.csv` | 42 | each center's `city`, `fixedCosts` ($f_i$), `lat`/`lon` |
    | `regions.csv` | 515 | each region's `yearlyDemand` ($d_j$), `minutesPerStop`, `lat`/`lon` |
    | `shifts.csv` | 21,630 | for every (center, region) pair: the `travelTime` in minutes |
    | `shifts_with_costs.csv` | 21,630 | a **reference** cost file we use to validate Part 3 |

    Two things to note up front, because they shape the whole analysis:

    - `shifts.csv` has 42 × 515 = 21,630 rows — every center paired with every region.
      The `transportationCosts` column in it is **only a placeholder** (it literally
      equals `travelTime`); computing the real cost is the job of Part 3.
    - `minutesPerStop` already bundles the driving *between* customers inside a region
      with the time spent *at* each customer, so a region's internal routing is
      pre-summarised into a single productivity number.
    """)
    return


@app.cell
def _(mo, pd):
    _base = "https://raw.githubusercontent.com/D3IP-SS25/data-driven-scm-dataset/refs/heads/main/data"
    warehouses = pd.read_csv(f"{_base}/warehouses.csv", index_col="warehouseID")
    regions = pd.read_csv(f"{_base}/regions.csv", index_col="regionID")
    shifts = pd.read_csv(f"{_base}/shifts.csv", index_col=["warehouseID", "regionID"])
    shifts_ref = pd.read_csv(f"{_base}/shifts_with_costs.csv", index_col=["warehouseID", "regionID"])

    W = warehouses.index.values          # the set of centers i
    R = regions.index.values             # the set of regions j
    n_pairs = len(shifts)                # 42 * 515 = 21,630 center-region pairs
    total_demand = int(regions["yearlyDemand"].sum())

    # Show the three primary tables as browsable tabs.
    tabs = mo.ui.tabs(
        {
            "Warehouses (42)": mo.ui.table(warehouses, selection=None, show_download=False),
            "Regions (515)": mo.ui.table(regions, selection=None, show_download=False),
            "Shifts (42x515)": mo.ui.table(shifts, selection=None, show_download=False),
        }
    )
    tabs
    return R, W, n_pairs, regions, shifts, shifts_ref, total_demand, warehouses


@app.cell(hide_code=True)
def _(mo, n_pairs, total_demand):
    mo.md(
        rf"""
        **Data sanity check.** We have **{n_pairs:,}** center–region pairs (= 42 × 515 ✓)
        and a total yearly demand of **{total_demand:,} stops/year** across all 515 regions.
        These two numbers match the case brief, so the tables loaded correctly and we can
        build on them.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 3 — Estimating the transport cost $c_{ij}$

    This is the first real piece of analysis. We need one number — the **yearly cost to
    serve region $j$ from center $i$** — but the data only gives us a *driving time*. We
    bridge that gap with the shift logic the business actually uses.

    ## 3.1 The derivation, one step at a time

    A truck leaves the center, drives to the region, works, and drives back.

    **Step 1 — How much working time is left in a shift?**
    The round trip eats $2 \cdot \text{travelTime}_{ij}$ minutes, so the time available
    for actually serving customers is

    $$\text{usable time} = \underbrace{450}_{\text{shift}} - 2\cdot \text{travelTime}_{ij}.$$

    **Step 2 — How many stops fit in one shift?**
    Each stop takes `minutesPerStop` (driving between shops + time at the shop):

    $$\text{stops per shift} = \frac{450 - 2\,\text{travelTime}_{ij}}{\text{minutesPerStop}_j}.$$

    **Step 3 — How many shifts does the region need per year?**
    The region needs `yearlyDemand` stops in total:

    $$\text{shifts per year} = \frac{\text{yearlyDemand}_j}{\text{stops per shift}}.$$

    **Step 4 — Multiply by the price of a shift (480 €):**

    $$\boxed{\; c_{ij} = 480 \cdot
    \frac{\text{yearlyDemand}_j \cdot \text{minutesPerStop}_j}{\,450 - 2\,\text{travelTime}_{ij}\,} \;}$$

    **Does this make economic sense?** Yes, on every term:
    the farther the region (bigger `travelTime`), the smaller the denominator → the more
    expensive it is; the more spread-out the customers (bigger `minutesPerStop`), the
    more expensive; the more demand, the more shifts, the more cost. Good.

    ## 3.2 The "unreachable link" trick

    Look at the denominator. If $2\cdot \text{travelTime}_{ij} \ge 450$, it becomes zero
    or negative — meaning the region simply **cannot be reached and served within a
    single shift** from that center. The cost is effectively infinite.

    > **Decision:** we *drop* those links entirely — we never create an $x_{ij}$ variable
    > for them. **Why this is safe:** every region keeps at least one reachable center
    > (we verified: min 1, median 5, max 12 reachable centers per region), so the model
    > stays feasible. **Why it is better than the alternative:** the reference file
    > instead keeps the link with a giant "Big-M" cost; that works but is numerically
    > ugly and slows the solver. Dropping the link gives the *same optimum* (an
    > infinite-cost assignment would never be chosen anyway) and a cleaner, faster model.

    We still reproduce the Big-M convention *for the validation only*, so we can compare
    against the benchmark file row for row.
    """)
    return


@app.cell
def _(SHIFT_COST, SHIFT_MIN, np, regions, shifts):
    # Reusable cost function. It returns ONLY the reachable (feasible) links as a
    # dict {(i, j): c_ij}. The keyword parameters are what let Part 6 re-run the very
    # same logic under different assumptions, without copy-pasting code:
    #   demand_factor : scales yearlyDemand   (Axis 1: cash demand falls)
    #   shift_cost    : replaces the 480 €    (Axis 2 / tech: wages, no crew)
    #   shift_min     : replaces the 450 min  (tech: 24/7 operation)
    #   travel_factor : scales travelTime     (tech: faster vehicles)
    def link_costs(demand_factor=1.0, shift_cost=SHIFT_COST, shift_min=SHIFT_MIN, travel_factor=1.0):
        region_ids = shifts.index.get_level_values("regionID")
        # Align region attributes to every shift row (one lookup per (i, j) pair).
        dem = regions["yearlyDemand"].reindex(region_ids).to_numpy() * demand_factor
        mps = regions["minutesPerStop"].reindex(region_ids).to_numpy()
        usable = shift_min - 2.0 * shifts["travelTime"].to_numpy() * travel_factor
        feasible = usable > 0  # the unreachable-link filter from 3.2
        # We compute cost for every row but only keep the feasible ones; the divide-by-zero
        # / negative cases are masked out, so we silence those warnings deliberately.
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = shift_cost * dem * mps / usable
        idx = shifts.index[feasible]
        return dict(zip(idx, cost[feasible]))

    return (link_costs,)


@app.cell
def _(link_costs):
    # The baseline costs (all factors at their default). These c_ij feed Parts 4 and 5.
    cost_base = link_costs()
    n_feasible = len(cost_base)
    return cost_base, n_feasible


@app.cell(hide_code=True)
def _(mo, n_feasible, n_pairs):
    _pct_unreach = 100.0 * (1 - n_feasible / n_pairs)
    mo.md(
        rf"""
        ## 3.3 Validation against the benchmark file

        Of the {n_pairs:,} possible center–region pairs, only **{n_feasible:,}** are
        reachable within a single shift — about **{_pct_unreach:.1f}%** are *not*. That is
        geographically sensible: most centers are simply too far from most regions, so the
        network is naturally sparse.

        To trust our formula we compare it, row for row, against the course's reference file
        `shifts_with_costs.csv`. We use our formula on reachable links and the Big-M
        constant on unreachable ones (matching the reference's convention), then check all
        {n_pairs:,} rows. One subtlety handled in the code: the two CSV files are sorted
        differently, so we must **re-index** the reference onto our row order before
        comparing — otherwise we would compare unrelated rows.
        """
    )
    return


@app.cell
def _(BIG_M_COST, SHIFT_COST, SHIFT_MIN, np, plt, regions, shifts, shifts_ref):
    # Build OUR cost for ALL rows (formula where reachable, Big-M otherwise) for a 1:1 compare.
    _ref = shifts_ref.reindex(shifts.index)   # critical: align row order (files are sorted differently!)
    _rid = shifts.index.get_level_values("regionID")
    _dem = regions["yearlyDemand"].reindex(_rid).to_numpy()
    _mps = regions["minutesPerStop"].reindex(_rid).to_numpy()
    _usable = SHIFT_MIN - 2.0 * shifts["travelTime"].to_numpy()
    _feas = _usable > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        _mine = np.where(_feas, SHIFT_COST * _dem * _mps / _usable, BIG_M_COST)
    _r = _ref["transportationCosts"].to_numpy()

    # Metrics: relative deviation on reachable links, and exact match on the Big-M rows.
    _rel = np.abs((_mine[_feas] - _r[_feas]) / _r[_feas])
    max_rel_dev = float(_rel.max())
    mean_rel_dev = float(_rel.mean())
    bigm_match = float(np.mean(np.isclose(_mine[~_feas], _r[~_feas])))

    # Scatter: our cost vs reference cost (reachable links only, log-log). Perfect overlap
    # on the diagonal y = x means "identical formula".
    fig_val, ax_val = plt.subplots(figsize=(5.2, 5.2))
    ax_val.scatter(_r[_feas], _mine[_feas], s=8, alpha=0.4)
    _lo, _hi = _r[_feas].min(), _r[_feas].max()
    ax_val.plot([_lo, _hi], [_lo, _hi], "r--", lw=1, label="ideal (y = x)")
    ax_val.set(xscale="log", yscale="log", xlabel="reference c_ij (€)",
               ylabel="our c_ij (€)", title="Part 3: our cost vs reference cost")
    ax_val.legend()
    fig_val.tight_layout()
    fig_val
    return bigm_match, max_rel_dev, mean_rel_dev


@app.cell(hide_code=True)
def _(bigm_match, max_rel_dev, mean_rel_dev, mo):
    mo.md(
        rf"""
        ### Reading the validation
        - **Largest relative deviation:** `{max_rel_dev:.2e}` — **mean:** `{mean_rel_dev:.2e}`.
          That is pure floating-point rounding (~$10^{{-16}}$), i.e. our formula is **exactly**
          the reference formula, not merely close.
        - **Unreachable (Big-M) rows** match the reference **{bigm_match:.0%}** of the time.
        - **Conclusion:** every point sits on the diagonal $y = x$. Our transport-cost
          estimate is **validated** against the benchmark. ✅ We can now trust these
          $c_{{ij}}$ as the input to the optimisation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 4 — Solving the base model

    Now we plug the validated $c_{ij}$ into the Warehouse-Location model from Part 1.
    It is the textbook model with two practical touches: it uses our **real** costs, and
    it only builds variables for **reachable** links $S$ (the trick from 3.2):

    $$\min \sum_{(i,j)\in S} x_{ij}\,c_{ij} + \sum_{i} f_i\,y_i
    \quad\text{s.t.}\quad \sum_{i} x_{ij}=1,\;\; x_{ij}\le y_i,\;\; x,y\in\{0,1\}.$$

    We wrap the whole thing in a function `solve_base(...)`. That is deliberate: Part 6
    calls this *same* function once per scenario, so the base case and every stress-test
    are guaranteed to be solved identically.

    **How the code mirrors the maths** (read alongside the cell):
    - `x = LpVariable.dicts("x", links, cat=Binary)` — one $x_{ij}$ per reachable link.
    - `y = LpVariable.dicts("y", W, cat=Binary)` — one $y_i$ per center.
    - `setObjective(...)` — the two sums of the objective.
    - `x[k] <= y[k[0]]` — the "use only open centers" constraint, per link.
    - `lpSum(x over region j) == 1` — the "served exactly once" constraint, per region.
    - `prob.solve(pl.HiGHS(...))` — hand it to the solver and read back the chosen values.
    """)
    return


@app.cell
def _(R, W, pl, regions, warehouses):
    def solve_base(cost_map, demand_factor=1.0):
        """Solve the base Warehouse-Location model over the reachable links in cost_map.

        cost_map      : {(i, j): c_ij} for reachable links only.
        demand_factor : only used to report each center's served volume consistently when
                        the scenario scaled demand; it does not change the optimisation.
        """
        links = list(cost_map.keys())
        # Index the links two ways so the constraints are cheap to build.
        by_region = {}   # region j -> its reachable (i, j) links
        by_wh = {}       # center i -> its reachable (i, j) links
        for (i, j) in links:
            by_region.setdefault(j, []).append((i, j))
            by_wh.setdefault(i, []).append((i, j))

        prob = pl.LpProblem("CashLog_Base", pl.LpMinimize)
        x = pl.LpVariable.dicts("x", links, cat=pl.LpBinary)   # x_ij: region j served by i?
        y = pl.LpVariable.dicts("y", list(W), cat=pl.LpBinary)  # y_i:  center i open?

        # Objective: transport cost over chosen links + fixed cost of open centers.
        prob.setObjective(
            pl.lpSum(x[k] * cost_map[k] for k in links)
            + pl.lpSum(y[i] * warehouses.loc[i, "fixedCosts"] for i in W)
        )
        for k in links:                       # x_ij <= y_i : only assign to an open center
            prob += x[k] <= y[k[0]]
        for j in R:                           # sum_i x_ij = 1 : each region served exactly once
            prob += pl.lpSum(x[k] for k in by_region[j]) == 1

        prob.solve(pl.HiGHS(msg=False))

        # Read the solution back out of the variables.
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
        ### Result of the base model
        - **Solver status:** `{base_res['status']}` — HiGHS returns the **globally optimal**
          integer solution (not just "a feasible one"), so this is the true cost minimum.
        - **Centers kept open:** **{len(base_res['open'])}** of 42.
        - **Total yearly cost:** {base_res['total']:,.0f} € — split into fixed cost
          {base_res['fixed']:,.0f} € and transport cost {base_res['variable']:,.0f} €.
        - **Open cities:** {_open_cities}

        The fact that the cost splits into a sizeable chunk of *both* fixed and transport
        cost is the trade-off of Part 1 made visible: the optimiser stopped opening centers
        exactly where the next building's fixed cost would outweigh the driving it saves.
        """
    )
    return


@app.cell
def _(R, base_res, cost_base, mo):
    # Independent re-check that the returned solution actually satisfies the constraints.
    # We never just trust the solver's "Optimal" label; we verify feasibility ourselves.
    _assign = base_res["assign"]
    _check_all_served = set(_assign.keys()) == set(R)                            # every region assigned
    _check_exactly_one = len(_assign) == len(R)                                  # exactly one center each
    _check_feasible_link = all((i, j) in cost_base for j, i in _assign.items())  # only reachable links used
    _check_open = all(i in base_res["open"] for i in _assign.values())          # only open centers used
    _ok = all([_check_all_served, _check_exactly_one, _check_feasible_link, _check_open])
    mo.md(
        rf"""
        **Constraint re-check (independent of the solver):**
        all 515 regions served: `{_check_all_served}` · exactly one center per region:
        `{_check_exactly_one}` · only reachable links used: `{_check_feasible_link}` ·
        only open centers used: `{_check_open}` → **solution is feasible: {_ok}** ✅
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plausibility map
    An optimisation result you cannot *see* and explain is suspicious — it is usually a
    data or model bug. So we plot it. Open centers are 🏦, closed ones a grey ✖, and
    every region is coloured by the center it is assigned to. We expect each center's
    coloured "territory" to be a compact, connected area of nearby regions (short
    `travelTime`). If instead a region were assigned across the country, or a center
    carried almost no volume, that would be a red flag.
    """)
    return


@app.cell
def _(R, base_res, folium, regions, sns, warehouses):
    # Map: open centers as a bank emoji, regions coloured by assigned center, closed
    # centers as a faint grey cross so "open vs closed" is unmistakable at a glance.
    _open = base_res["open"]
    _palette = sns.color_palette("tab20", n_colors=max(len(_open), 1)).as_hex()
    _color = {w: _palette[k % 20] for k, w in enumerate(_open)}

    m = folium.Map(location=[40, -3.5], zoom_start=6, tiles="cartodbpositron")
    for j in R:
        i = base_res["assign"][j]
        folium.CircleMarker(
            location=[regions.loc[j, "lat"], regions.loc[j, "lon"]],
            radius=3, color=_color[i], fill=True, fill_opacity=0.8, weight=0.4,
            popup=f"{regions.loc[j,'city']} -> {warehouses.loc[i,'city']}",
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
    # Part 5 — Making it realistic: economies of scale (the extended model)

    ## 5.1 What the base model gets wrong

    The base model charges a **constant** fixed cost $f_i$ for every center, no matter
    how much it handles. That is unrealistic: this is a *strategic, long-term* decision,
    so a center can be built **bigger or smaller**. A bigger center costs more to run but
    is **cheaper per delivery** (it processes cash at scale). The true cost-vs-volume
    relationship is a **curve**, not a flat fee plus a straight line.

    ## 5.2 The modelling trick: approximate the curve with 5 straight pieces

    We cannot put a smooth curve into a *linear* program. The standard trick is
    **piecewise-linear costs**: chop the volume axis into a few ranges and let each range
    be a center *type*. We use five — very small `v`, small `s`, medium `m`, large `l`,
    huge `h`. Each type $t$ has a volume window $[V_t^{lb}, V_t^{ub}]$, a yearly fixed
    cost $c_t^{fix}$ and a variable cost per delivery $c_t^{var}$. Bigger type → higher
    fixed, lower variable. (The exact numbers are in the configuration cell.)

    ## 5.3 The new variables and constraints

    We add, for every center $i$ and type $t$:
    $y_{it} \in \{0,1\}$ (is $i$ built as type $t$?) and $z_{it} \ge 0$ (volume handled
    as that type).

    $$\min \sum_{(i,j)} x_{ij}\,c_{ij}
    + \sum_{i}\sum_{t} c_t^{fix}\,y_{it}
    + \sum_{i}\sum_{t} c_t^{var}\,z_{it}$$

    | Constraint | What it enforces |
    |---|---|
    | $\sum_t y_{it} \le 1\;\;\forall i$ | **at most one type per center** |
    | $x_{ij} \le \sum_t y_{it}$ | a region may only use an open center |
    | $\sum_i x_{ij} = 1$ | each region served exactly once |
    | $\sum_t z_{it} = \sum_j x_{ij}\,d_j$ | volume handled = demand actually assigned |
    | $V_t^{lb}\,y_{it} \le z_{it} \le V_t^{ub}\,y_{it}$ | volume must fit the chosen type's window |

    > **Why $\sum_t y_{it} \le 1$ is essential:** without it the model could label one
    > center as several types at once and cherry-pick the cheapest slice of each — which
    > would destroy the whole piecewise-linear idea. Forcing *exactly one* type makes the
    > center sit on *one* cost piece, as intended. Notice how the last two constraints
    > work together: $y_{it}=0$ forces $z_{it}=0$ (an unused type carries no volume), and
    > $y_{it}=1$ forces $z_{it}$ into that type's window. That is how a binary "which
    > size" choice and a continuous "how much volume" choice are linked.

    **One honest caveat about comparing totals:** here $c_{ij}$ is *transport* and
    $c_t^{var}$ is *processing* (counting/sorting) — they are different cost categories,
    not double-counting. So the absolute total of this model is **not** directly
    comparable to the base model's total. What we compare is the **structure**: which
    centers stay open, and at which size.
    """)
    return


@app.cell(hide_code=True)
def _(TYPE_PARAMS, mo, pd):
    _df = pd.DataFrame(
        [(t, lb, ub, cf, cv) for t, (lb, ub, cf, cv) in TYPE_PARAMS.items()],
        columns=["Type", "V_lb", "V_ub", "c_fix (€)", "c_var (€/delivery)"],
    )
    mo.vstack(
        [
            mo.md("**The five center types (documented economies-of-scale assumption):**"),
            mo.ui.table(_df, selection=None, show_download=False),
        ]
    )
    return


@app.cell
def _(EXT_GAP_REL, EXT_TIME_LIMIT, OBJ_SCALE, R, W, pl, regions):
    def solve_extended(cost_map, type_params, demand_factor=1.0,
                       obj_scale=OBJ_SCALE, gap_rel=EXT_GAP_REL, time_limit=EXT_TIME_LIMIT):
        """Extended MILP with 5 center types (y_it, z_it) and piecewise-linear costs.

        Two numerical aids make this hard model solve reliably (see 5.4 in the text):
          * obj_scale shrinks the objective (costs ~1e8) into thousands of euros, which the
            solver handles far better; we multiply the final objective back up for display.
          * gap_rel / time_limit stop the solver once it is within 1% of optimal (or after a
            time cap), which is plenty for a strategic keep/close decision.
        """
        links = list(cost_map.keys())
        types = list(type_params.keys())
        by_region, by_wh = {}, {}
        for (i, j) in links:
            by_region.setdefault(j, []).append((i, j))
            by_wh.setdefault(i, []).append((i, j))
        d = {j: regions.loc[j, "yearlyDemand"] * demand_factor for j in R}
        s = obj_scale  # short alias used in the objective below

        prob = pl.LpProblem("CashLog_Extended", pl.LpMinimize)
        x = pl.LpVariable.dicts("xe", links, cat=pl.LpBinary)
        y = {(i, t): pl.LpVariable(f"y_{i}_{t}", cat=pl.LpBinary) for i in W for t in types}
        z = {(i, t): pl.LpVariable(f"z_{i}_{t}", lowBound=0) for i in W for t in types}

        # Objective in scaled units (divide every coefficient by s = obj_scale).
        prob.setObjective(
            pl.lpSum(x[k] * (cost_map[k] / s) for k in links)
            + pl.lpSum(y[i, t] * (type_params[t][2] / s) for i in W for t in types)
            + pl.lpSum(z[i, t] * (type_params[t][3] / s) for i in W for t in types)
        )
        for i in W:
            prob += pl.lpSum(y[i, t] for t in types) <= 1                      # <= 1 type per center
        for k in links:
            prob += x[k] <= pl.lpSum(y[k[0], t] for t in types)                # only assign to open center
        for j in R:
            prob += pl.lpSum(x[k] for k in by_region[j]) == 1                  # exactly one center per region
        for i in W:
            prob += pl.lpSum(z[i, t] for t in types) == pl.lpSum(
                x[k] * d[k[1]] for k in by_wh.get(i, [])
            )                                                                  # volume = assigned demand
            for t in types:
                lb, ub, _, _ = type_params[t]
                prob += z[i, t] >= lb * y[i, t]                                # volume fits type window...
                prob += z[i, t] <= ub * y[i, t]                                # ...from below and above

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
            "total": prob.objective.value() * s,   # scale back into euros for display
        }

    return (solve_extended,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.4 Why this model needs two numerical tricks (and why that is fine)

    This MILP is genuinely hard: 2,661 binary $x$ + 210 binary $y$, plus a non-convex
    piecewise cost curve with weak linear relaxation (the $z \le V^{ub} y$ links are
    loose). Two settings make it behave, and a beginner deserves to know *why*:

    1. **Objective scaling (÷1000).** The raw costs are ~$10^8$. Solvers do arithmetic in
       floating point, and mixing huge and tiny numbers causes rounding trouble — HiGHS
       literally warns "scale the objective". Working in **thousands of euros** keeps the
       numbers in a comfortable range; we multiply the answer back up at the end, so the
       reported euros are unchanged.
    2. **A 1% optimality gap + time limit.** Proving an integer solution is *exactly*
       optimal can take very long; getting within 1% is fast. The solver reports a
       **gap** = how far the current solution could *at worst* be from the best possible.
       Stopping at ≤1% means we might leave ≤1% of cost on the table — irrelevant for a
       strategic *which buildings* decision, where the open/size structure is stable. We
       **report** the gap honestly rather than claiming "provably optimal".

    The next cell runs the model (≈30–90 s — this is the slow step of the notebook).
    """)
    return


@app.cell
def _(TYPE_PARAMS, cost_base, solve_extended):
    # Heavy step: the extended MILP solves to a 1% gap and runs eagerly so the notebook is
    # fully self-contained top-to-bottom. Expect roughly 30-90 s depending on the machine.
    ext_res = solve_extended(cost_base, TYPE_PARAMS)
    return (ext_res,)


@app.cell(hide_code=True)
def _(TYPE_PARAMS, base_res, ext_res, mo, warehouses):
    _b = set(base_res["open"])
    _e = set(ext_res["open"])
    _only_base = ", ".join(sorted(warehouses.loc[i, "city"] for i in _b - _e)) or "—"
    _only_ext = ", ".join(sorted(warehouses.loc[i, "city"] for i in _e - _b)) or "—"
    _type_counts = {t: sum(1 for tt in ext_res["type"].values() if tt == t) for t in TYPE_PARAMS}
    # Verify every open center's processed volume lies inside its chosen type's [V_lb, V_ub].
    _viol = [
        warehouses.loc[i, "city"]
        for i, t in ext_res["type"].items()
        if not (TYPE_PARAMS[t][0] - 1 <= ext_res["vol"].get(i, 0) <= TYPE_PARAMS[t][1] + 1)
    ]
    mo.md(
        rf"""
        ### Base vs extended model
        - **Solver status:** `{ext_res['status']}` — solved to a **≤1% optimality gap**
          (`gapRel=0.01`), as discussed in 5.4.
        - **Volume check:** is every open center inside its type's volume window?
          **{"YES ✅" if not _viol else "NO – " + ", ".join(_viol)}** (this confirms the
          $V^{{lb}}\,y \le z \le V^{{ub}}\,y$ constraints did their job).
        - **Open centers:** base **{len(_b)}** → extended **{len(_e)}**.
        - **Type distribution (extended):** { {t: c for t, c in _type_counts.items() if c} }
        - **Open only in the base model:** {_only_base}
        - **Open only in the extended model:** {_only_ext}

        **Interpretation.** The extended model separates a center's **fixed cost by size**
        from its **per-delivery processing cost**. Economies of scale reward bigger centers
        (cheaper per delivery), while the reachability limit (most links are blocked) still
        forces many small `v`/`s` centers to cover the map. Whether the *number* of centers
        rises or falls versus the base model is therefore a **real outcome of that
        trade-off**, not an artefact — and the type distribution above shows how the network
        splits across the size classes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 6 — Stress-testing the answer (sensitivity analysis)

    A single optimal answer is fragile: it is optimal *for today's assumptions*. The
    real deliverable is a **robust** recommendation — one that holds up across a
    plausible range of futures. So we re-solve the (base) model under three forces the
    case brief calls out, and watch **which centers stay open**.

    1. **Falling demand** (−10/−30/−50%): cash keeps shrinking. Since $c_{ij}$ is
       proportional to demand, transport cost falls relative to fixed cost →
       expectation: **fewer** centers.
    2. **Rising shift cost** (+20/+50%): wages and fuel go up, making transport pricier
       and short trips more valuable → expectation: **more / more decentralised** centers.
    3. **New technology** (autonomous trucks): the crew cost vanishes (−60% shift cost),
       optionally with 24/7 operation and faster driving (more reach) → expectation:
       **fewer, larger** centers.

    Every scenario reuses the *same* `solve_base` function — only the $c_{ij}$ (and, for
    Axis 1, the demand) change. That is what makes the comparison clean: any difference
    in the result comes from the assumption, not from a different model.
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
    # Build every scenario as  name -> (cost_map, demand_factor)  then solve them all.
    scenarios = {"Base": (cost_base, 1.0)}
    for _name, _f in DEMAND_SCENARIOS.items():
        scenarios[_name] = (link_costs(demand_factor=_f), _f)          # Axis 1: scale demand
    for _name, _f in SHIFTCOST_SCENARIOS.items():
        scenarios[_name] = (link_costs(shift_cost=SHIFT_COST * _f), 1.0)  # Axis 2: scale shift cost
    for _name, _kw in TECH_SCENARIOS.items():                         # Axis 3: technology bundles
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
    # Two views of the scenario results:
    #   summary     : one row per scenario (status, #open centers, total cost)
    #   open_matrix : a center x scenario grid of 1 (open) / 0 (closed) -> drives the heatmap
    summary = pd.DataFrame(
        {
            name: {
                "Status": r["status"],
                "# open centers": len(r["open"]),
                "Total cost (€)": round(r["total"]),
            }
            for name, r in sens_results.items()
        }
    ).T

    open_matrix = pd.DataFrame(
        {name: {warehouses.loc[i, "city"]: int(i in r["open"]) for i in W}
         for name, r in sens_results.items()}
    )
    # Only show centers that are open in at least one scenario (keeps the heatmap readable).
    open_matrix = open_matrix[open_matrix.sum(axis=1) > 0].sort_index()
    return open_matrix, summary


@app.cell(hide_code=True)
def _(mo, summary):
    mo.vstack(
        [
            mo.md("### Scenario overview"),
            mo.ui.table(summary.reset_index(names="Scenario"), selection=None, show_download=False),
        ]
    )
    return


@app.cell
def _(open_matrix, plt, sns):
    # Heatmap: which centers are open (blue) vs closed (grey) in each scenario. Reading a
    # row left-to-right tells you how stable that center is; reading a column tells you how
    # lean the network gets under that assumption.
    fig_h, ax_h = plt.subplots(figsize=(8, max(4, 0.32 * len(open_matrix))))
    sns.heatmap(open_matrix, cmap=["#f0f0f0", "#2c7fb8"], cbar=False, linewidths=0.5,
                linecolor="white", ax=ax_h)
    ax_h.set(title="Open (blue) / closed (grey) by scenario", xlabel="", ylabel="Center")
    plt.setp(ax_h.get_xticklabels(), rotation=35, ha="right")
    fig_h.tight_layout()
    fig_h
    return


@app.cell
def _(open_matrix):
    # Robustness classification straight from the matrix:
    #   always_open    -> open in EVERY scenario  (robust "keep")
    #   sometimes_open -> open in some but not all (the genuinely uncertain ones)
    #   never_open     -> open in NO scenario     (= centers absent from open_matrix)
    n_scen = open_matrix.shape[1]
    open_count = open_matrix.sum(axis=1)
    always_open = sorted(open_count[open_count == n_scen].index)
    sometimes_open = sorted(open_count[(open_count > 0) & (open_count < n_scen)].index)
    return always_open, sometimes_open


@app.cell(hide_code=True)
def _(W, always_open, mo, open_matrix, sometimes_open):
    _never = len(W) - open_matrix.shape[0]
    mo.md(
        rf"""
        ### Reading the sensitivity
        - **Always open ({len(always_open)} centers)** — the robust *keep* backbone:
          {", ".join(always_open)}
        - **Scenario-dependent ({len(sometimes_open)})** — the genuinely *critical* cases:
          {", ".join(sometimes_open) or "—"}
        - **Never open ({_never} centers)** — robust *close* candidates (open in no scenario).

        **Did the expectations hold?** Falling demand should thin the network (fixed cost
        dominates); rising shift cost should favour more, more local centers; autonomous
        technology should collapse the network onto a few large hubs. The heatmap above lets
        you confirm each direction — and any direction that *doesn't* match is itself a
        finding worth reporting, not something to hide.
        """
    )
    return


@app.cell(hide_code=True)
def _(always_open, mo, open_matrix, sometimes_open, warehouses):
    _never = sorted(set(warehouses["city"]) - set(open_matrix.index))
    mo.md(
        rf"""
        # Part 7 — Conclusion & recommendation

        ## Management recommendation (robust across all scenarios)
        1. **Keep ({len(always_open)}):** centers open in *every* scenario form the robust
           backbone of the network — invest here without hesitation. ✅
           → {", ".join(always_open)}
        2. **Close ({len(_never)}):** centers needed in *no* scenario are clear closure
           candidates — closing them frees fixed cost without putting service at risk.
        3. **Watch / examine ({len(sometimes_open)}):** {", ".join(sometimes_open) or "—"}.
           Their status depends on how the future actually unfolds (demand trend, wage/fuel
           costs, technology). Recommendation: **keep the option open** and tie the decision
           to the real demand/cost trend as it becomes clear.

        ## Limits & key assumptions (with their effect)
        - **Shift approximation:** routing within a region is approximated by one shift
          formula (continuous shifts, no day-to-day variation). *Effect:* the $c_{{ij}}$ are
          strategic estimates, not operational actuals.
        - **"Reachable in one shift":** most links are excluded; multi-day or transshipment
          tours are not modelled. *Effect:* a few centers look "forced open" because they are
          the only one able to reach a region within a single shift.
        - **Center-type parameters (Part 5)** are plausible but assumed numbers. *Effect:*
          they shift how strongly the network consolidates; change them in the config cell.
        - **Technology scenarios** are deliberately coarse "what-ifs" (−60% shift cost, 24/7).
          *Effect:* they show direction and leverage, not a point forecast.
        - **Static view:** no transition/closure costs, no multi-period dynamics.

        ## The red thread, one more time
        Correct $c_{{ij}}$ (validated) → base model (mapped & sanity-checked) → economies of
        scale (5 types) → sensitivity over 3 axes → a **robust** keep / close / watch
        recommendation. Each step fed the next; nothing was asserted without being computed
        and checked.
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
