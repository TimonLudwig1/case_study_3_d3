import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CashLog - Designing the optimal Cash-Center Network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problem Definition and Strategic Context

    ### 1.1 Background and Market Dynamics
    CashLog operates as the market leader in cash logistics within the Spanish financial sector. The core business model comprises three primary operational pillars:
    * **Cash Collection:** collecting cash from customers
    * **Processing and Vaulting:** counting, verification, and high-security storage within dedicated Cash Centers.
    * **Distribution:** redistribution of physical currency to client networks.

    Historically, CashLog's network architecture followed a decentralized structure, maintaining one Cash Center in every Spanish provincial capital. This geographic distribution was strictly coupled with the local branch network of the Spanish Central Bank. Due to institutional restructuring, this regulatory dependency has ceased to exist, forcing a strategic reallocation of assets. Currently, the efficiency of the network is constrained by two countervailing macroeconomic pressures:

    1. **Secular Decline in Cash Volume:** The accelerating substitution of physical currency by digital and electronic payment systems has led to a structural contraction in total processed volume per facility.
    2. **High Asset Specificity and Fixed Costs:** Security, surveillance and insurance make every
      cash center highly expensive just to keep open.

    ### 1.2 Current Network Topology
    The current logistics network consists of a set of existing facilities and demand aggregates:
    * **Facilities:** $M = 42$ active Cash Centers.
    * **Demand Points:** $N = 42,000$ client locations, structurally aggregated into $K = 515$ distinct customer regions.

    ### 1.3 Optimization Objective and Operational Constraints
    The strategic objective is to formulate a network optimization problem aimed at identifying the optimal subset of Cash Centers to retain. The mathematical objective is to minimize total annualized expenditures while ensuring demand fulfillment.

    The optimization model is bound by two deterministic business constraints:

    1. **Asset Exclusivity:** The solution space for potential facility locations is strictly bounded by the current network. The opening of new locations is excluded.
    2. **Guaranteed Service Levels:** Demand fulfillment is non-negotiable. A facility closure is only permissible if the associated regional demand can be completely absorbed by the remaining active network without violating capacity or distance thresholds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formalizing the Problem

    ### 2.1 The Underlying Economic Trade-off

    Centers cannot simply be closed on the basis of cost alone, since an underlying trade-off governs the decision problem. Opening more centers reduces driving time and distance, and thus transport cost, but increases the fixed cost of operating additional Cash Centers. Conversely, opening fewer centers lowers fixed cost but increases transport cost through longer average distances. The cheapest, most stable, and most efficient network therefore lies somewhere between these two extremes, rather than at either boundary.

    ### 2.2 Network Interdependencies and Systemic Optimization

    Scoring each center individually and closing the worst-performing one is not a valid approach, due to network effects. Whether a given center is worth keeping depends on which other centers remain open. If one center is closed, its regions must be reassigned to other centers, which in turn changes their load and cost parameters. Because regions are interdependent, all centers must be decided upon simultaneously, which requires an optimization model rather than a sequence of local decisions.

    ### 2.3 Baseline Formulation: The Warehouse-Location Model

    The classic template for a "which facilities should remain open" problem is the warehouse-location model.

    **Notation**

    | Symbol | Meaning |
    |---|---|
    | $i$ | a Cash Center (one of 42), the "warehouse" |
    | $j$ | a customer region (one of 515) |
    | $f_i$ | yearly fixed cost of operating center $i$ |
    | $c_{ij}$ | yearly transport cost of serving region $j$ from center $i$ |
    | $y_i \in \{0,1\}$ | **decision:** is center $i$ open (1) or closed (0)? |
    | $x_{ij} \in \{0,1\}$ | **decision:** is region $j$ served by center $i$? |

    **Model formulation:**

    $$\min_{x,y}\; \underbrace{\sum_{i}\sum_{j} x_{ij}\,c_{ij}}_{\text{transport cost}}
    \;+\; \underbrace{\sum_{i} f_i\,y_i}_{\text{fixed cost of open centers}}$$

    subject to

    $$\sum_{i} x_{ij} = 1 \;\;\forall j \quad\text{(every region is served by exactly one center)}$$
    $$x_{ij} \le y_i \;\;\forall i,j \quad\text{(a region may only be served by a center that is open)}$$

    This model provides the underlying logic for our approach. However, several of its core assumptions do not hold for the CashLog problem.

    ### Structural Limitations of the Baseline Model

    Three of the model's built-in assumptions cannot be applied directly to the CashLog problem:

    **1. $c_{ij}$ has no obvious value.** The model assumes a known cost-per-link $c_{ij}$. No such "cost to serve region $j$ from center $i$" figure exists in the raw data. Trucks serve many customers per eight-hour shift along a route, rather than making one round trip per customer. $c_{ij}$ must therefore be derived from shift logic.

    **2. Individual customers are the wrong unit of analysis.** The model assumes a manageable, fixed set of customers $j$. Optimizing location choices against roughly 42,000 individual points is computationally impractical and does not reflect how routing works in practice, since trucks do not make isolated trips for single customers. Customers must therefore first be aggregated into cluster regions.

    **3. Fixed cost $f_i$ is treated as a single constant per center.** The model assumes each facility has one fixed-cost figure, independent of the volume it processes. At CashLog, capacity is a strategic, investable choice. Centers can be built at different sizes, and costs follow economies of scale, so a larger center costs more in absolute terms but less per delivery. A single constant $f_i$ cannot capture this relationship.

    **Consequence:** points 1 and 2 are pre-processing problems addressed before the model is run. Point 3 is a structural flaw in the objective function itself. Resolving it requires replacing the single constant $f_i$ with a cost structure that depends on the volume a center actually handles. This extended model is developed later, once the data has been examined closely enough to calibrate it properly.
    """)
    return


@app.cell
def _():
    #shared imports and parameters
    #imports
    import folium
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pulp as pl
    import highspy  # noqa: F401  (backend for pl.HiGHS, same as base notebook)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from IPython.display import display

    #parameters
    shift_min = 450
    shift_cost_base = 480
    return (
        display,
        folium,
        mcolors,
        mo,
        np,
        pd,
        pl,
        plt,
        shift_cost_base,
        shift_min,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Data

    We load four tables from the course's public GitHub repository.
    """)
    return


@app.cell
def _(pd):
    _base = "https://raw.githubusercontent.com/D3IP-SS25/data-driven-scm-dataset/refs/heads/main/data"
    warehouses = pd.read_csv(f"{_base}/warehouses.csv", index_col="warehouseID")
    regions = pd.read_csv(f"{_base}/regions.csv", index_col="regionID")
    shifts = pd.read_csv(f"{_base}/shifts.csv", index_col=["warehouseID", "regionID"])
    shifts_ref = pd.read_csv(f"{_base}/shifts_with_costs.csv", index_col=["warehouseID", "regionID"])
    return regions, shifts, shifts_ref, warehouses


@app.cell
def _(display, pd):
    #helper function for data analysis
    from IPython.display import Markdown

    def quick_look(df: pd.DataFrame, name: str, exclude: list[str] | None = None):
        missing = df.isna().sum()
        missing = missing[missing > 0]

        meta_md = f"**`{name}`** — {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
        meta_md += "| column | dtype | missing |\n|---|---|---|\n"
        meta_md += "\n".join(f"| `{c}` | {df[c].dtype} | {missing.get(c, 0)} |" for c in df.columns)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        id_like = [c for c in numeric_cols if c.lower().endswith(("id", "_id"))]
        drop = set(id_like) | set(exclude or [])
        numeric_cols = [c for c in numeric_cols if c not in drop]

        display(Markdown(meta_md))
        display(df.head(3))
        if numeric_cols:
            display(df[numeric_cols].describe().round(2))
        else:
            display(Markdown("_no non-ID numeric columns to describe_"))

    return (quick_look,)


@app.cell
def _(display, quick_look, warehouses):
    display(quick_look(warehouses, "warehouses"))
    display(warehouses.sort_values("fixedCosts", ascending=False).head(3))
    return


@app.cell
def _(quick_look, regions):
    quick_look(regions, "regions")
    return


@app.cell
def _(quick_look, shifts):
    quick_look(shifts, "shifts")
    return


@app.cell
def _(shifts):
    shifts
    return


@app.cell
def _(quick_look, shifts_ref):
    quick_look(shifts_ref, "shifts_ref")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use four different datasets:

    - `warehouses.csv`
    - `regions.csv`
    - `shifts.csv`
    - `shifts_with_costs.csv`

    The warehouses dataset contains the city, location, and fixed cost of opening and operating each of the 42 candidate warehouses. Fixed costs range from €1,344,000.00 to €35,904,000.00, with the most expensive warehouses located in Madrid and Barcelona. Likely reflecting the size and property price level of these regions.

    The regions dataset contains information for each of the 515 clustered demand regions, and therefore already addresses the second major limitation of the base model discussed above: customers were pre-aggregated into regions by zip code. For each region, the dataset provides a unique regionID, the city name, its location, yearly demand, and average minutesPerStop. Yearly demand ranges from 16.00 to 247,638.00, and minutesPerStop ranges from 6.20 to 32.12 minutes. The minutesPerStop column already combines the driving time between customers within a region with the time spent at each customer, and therefore represents a single, pre-summarized productivity figure.

    The third dataset, shifts.csv, contains the travel time (in minutes) and transportation cost for every (center, region) pair, totaling 21,630 rows. The transportationCosts column in this dataset is a placeholder equal to the travel time; sensible, logic-based proxies for transportation cost are derived from travel time in the following step.

    The final dataset, shifts_with_costs.csv, is our reference file containing precomputed transportation costs. It is used later to validate our own derivation of transportation costs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Estimating the Transport Cost $c_{ij}$

    Our raw dataset only provides driving time as a placeholder for the yearly cost of serving region $j$ from center $i$. An estimate must therefore be derived from the shift logic actually used by the business: a truck leaves the center, drives to the region, works, and drives back to the Cash Center.

    **1. Working time per shift.** The round trip consumes $2 \cdot \text{travelTime}_{ij}$ minutes.$^*$ The time available for actually serving customers is therefore:

    $$\text{usable time} = \underbrace{450}_{\text{shift length}} - 2\cdot \text{travelTime}_{ij}.$$

    This is the time remaining for service after the round trip has been accounted for.

    **2. Number of stops per shift.** Each stop takes minutesPerStop (driving between customers plus time spent at each customer). Dividing the usable time by the average time per stop gives the average number of stops per shift:

    $$\text{stops per shift} = \frac{450 - 2\,\text{travelTime}_{ij}}{\text{minutesPerStop}_j}.$$

    **3. Number of shifts a region requires per year.** Each region requires yearlyDemand stops in total. Dividing the yearly stop demand by the number of stops possible in a single shift gives the total number of shifts a region requires per year for all its customers to be served:

    $$\text{shifts per year} = \frac{\text{yearlyDemand}_j}{\text{stops per shift}}.$$

    **4. Deriving the transport cost.** To obtain a feasible value for the yearly transport cost, the number of shifts per year is multiplied by the cost of a single shift:

    $$c_{ij} = 480 \cdot
    \frac{\text{yearlyDemand}_j \cdot \text{minutesPerStop}_j}{450 - 2\,\text{travelTime}_{ij}}.$$

    $^*$ This assumes travelTime$_{ij}$ represents a one-way trip from center to region, requiring the factor of 2 to capture the full round trip; this is the most plausible reading of the raw data, though the dataset does not state it explicitly.

    ### Interpretation

    The farther away a region (the larger the travel time) the smaller the denominator becomes, which increases $c_{ij}$. The more spread out the customers within a region (the larger minutesPerStop) the more expensive it is to serve them. Finally, the larger the demand, the larger the numerator, and thus the higher the cost.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 Unreachable Links

    Routes for which $2\cdot \text{travelTime}_{ij} \ge 450$ are infeasible. Travel time alone would already exceed the usable time per shift, and the denominator in our cost formula becomes zero or negative.

    Two options exist for handling such links: dropping them entirely (never creating an $x_{ij}$ variable for them), or retaining them with a Big-M cost large enough that the solver would never select them.

    Dropping is the simpler and more robust option, where it is safe to do so. It guarantees that the connection is never chosen, since a variable that does not exist cannot be selected. A Big-M penalty only achieves the same outcome if $M$ happens to be larger than every alternative the model could otherwise choose. A judgment call that would need to be justified separately, and one that cannot yet be made since the model has not been solved and its cost scale is not yet known.

    Our approach is therefore to drop unreachable links, unless doing so would leave some region with no reachable center at all. In that case, dropping would render the model infeasible, and we would fall back to a Big-M penalty instead. This condition is checked in the following step.
    """)
    return


@app.cell
def _(mo, regions, shift_min, shifts, shifts_ref):
    # a link is reachable if 2*travelTime < shift_min (i.e. denominator > 0)
    # transform the shifts and regions dataframes to usable format for easier analysis
    shifts_long = shifts.reset_index()
    shifts_ref_long = shifts_ref.reset_index()
    regions_flat = regions.reset_index()  # regionID von Index zu normaler Spalte

    is_reachable = 2 * shifts_long["travelTime"] < shift_min

    reach_counts = (
        shifts_long[is_reachable]
        .groupby("regionID")["warehouseID"]
        .nunique()
    )

    n_unreachable_regions = (~regions_flat["regionID"].isin(reach_counts.index)).sum()
    _stats = reach_counts.describe()

    mo.md(f"""
    **Regions with zero reachable centers: {n_unreachable_regions}**

    Reachable centers per region:

    | Statistic | Value |
    |---|---:|
    | Count | {_stats['count']:.0f} |
    | Mean | {_stats['mean']:.1f} |
    | Std | {_stats['std']:.1f} |
    | Min | {_stats['min']:.0f} |
    | 25% | {_stats['25%']:.0f} |
    | Median (50%) | {_stats['50%']:.0f} |
    | 75% | {_stats['75%']:.0f} |
    | Max | {_stats['max']:.0f} |
    """)
    return is_reachable, regions_flat, shifts_long, shifts_ref_long


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decision: Drop Unreachable Links

    The check confirms that every region retains at least one reachable center (minimum 1, median 5, maximum 12 reachable centers per region, across all 515 regions). Dropping unreachable links therefore does not risk infeasibility, and we adopt the simpler, more robust option. Unreachable links are dropped entirely rather than handled via a Big-M penalty.
    """)
    return


@app.cell
def _(is_reachable, mo, shifts_long):
    # Nur die als erreichbar geprüften Links behalten wir (siehe Entscheidung oben).
    shifts_reachable = shifts_long[is_reachable].copy()

    n_total = len(shifts_long)
    n_kept = len(shifts_reachable)
    n_dropped = n_total - n_kept

    mo.md(f"""
    - Total possible center–region links: **{n_total:,}**
    - Reachable (kept): **{n_kept:,}** ({n_kept/n_total:.1%})
    - Unreachable (dropped): **{n_dropped:,}** ({n_dropped/n_total:.1%})
    """)
    return (shifts_reachable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dropping unreachable links reduces the model from all 21,630 theoretically possible center–region pairs to 2,661 actually usable ones, removing approximately 87.7% of all pairs. This is consistent with the reachability check above (a median of 5 reachable centers per region, out of 42 in total): most centers are simply too far from most regions, so a sparse network is exactly what would be expected geographically, rather than an indication of a problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Calculating Transport costs

    We can now calculate every transportation cost for every possible (center, region) tupel
    """)
    return


@app.cell
def _(regions_flat, shift_cost_base, shift_min, shifts_reachable):
    #transportation costs per year for each reachable link
    demand_by_region = regions_flat.set_index("regionID")["yearlyDemand"]
    stop_time_by_region = regions_flat.set_index("regionID")["minutesPerStop"]

    shifts_reachable["yearlyDemand"] = shifts_reachable["regionID"].map(demand_by_region)
    shifts_reachable["minutesPerStop"] = shifts_reachable["regionID"].map(stop_time_by_region)

    usable_minutes = shift_min - 2 * shifts_reachable["travelTime"]
    shifts_reachable["transportationCosts"] = (
        shift_cost_base * shifts_reachable["yearlyDemand"] * shifts_reachable["minutesPerStop"] / usable_minutes
    )

    cost_base = shifts_reachable[["warehouseID", "regionID", "transportationCosts"]]
    cost_base.head()
    return (cost_base,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before we continue with further analysis and modeling, we sanity-check the shape of our results:
    """)
    return


@app.cell
def _(cost_base, shifts_long):
    sanity = cost_base.merge(
        shifts_long[["warehouseID", "regionID", "travelTime"]],
        on=["warehouseID", "regionID"],
    )
    sanity.sort_values("travelTime").head(10)
    return (sanity,)


@app.cell
def _(sanity):
    sanity.sort_values("travelTime").tail(10)
    return


@app.cell
def _(plt, sanity, shift_min):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.scatter(sanity['travelTime'], sanity['transportationCosts'], alpha=0.4, s=10)
    _ax.set_xlabel('One-way travel time (minutes)')
    _ax.set_ylabel('Estimated annual transport cost (€)')
    _ax.set_title('Sanity check: cost should rise with travel time, and blow up near the\nfeasibility limit (225 min = half of 450-min shift)')
    _ax.axvline(shift_min / 2, color='red', linestyle='--', linewidth=1, label='feasibility limit (225 min)')
    _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At first glance, the transportation costs appear plausible: transport cost rises with one-way travel time, and rises steeply as travel time approaches the 225-minute feasibility limit. This matches the shape of our formula, in which the denominator shrinks toward zero.

    One irregularity, however, needs to be checked before proceeding. Six (warehouse, region) pairs have a travel time of 0, yet their transportation costs vary considerably. At a travel time of 0, the denominator equals 450 for each of these rows, so differences in cost can only stem from variation in yearly demand and/or minutesPerStop. Mathematically this is correct, but we nonetheless check whether the center is located at, or very close to, the geographic center of the region:
    """)
    return


@app.cell
def _(regions_flat, sanity, warehouses):
    warehouses_flat = warehouses.reset_index()

    sanity[sanity["travelTime"] == 0][["warehouseID", "regionID"]].merge(
        warehouses_flat[["warehouseID", "city"]], on="warehouseID"
    ).merge(
        regions_flat[["regionID", "city", "zipCode"]], on="regionID"
    )
    return (warehouses_flat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The check confirms that this is not a data issue: in every case, the warehouse city is either identical to the region's city (e.g., Jaen–Jaen, Soria–Soria, Toledo–Toledo) or a directly neighbouring town (e.g., Oviedo–Siero, Granada–Ogijares). These are simply regions whose center coincides with the location of the cash center itself, so a travel time of zero is exactly what would be expected. We can therefore conclude that entries with large variations in transportation cost despite similar travel times are driven solely by variance in yearlyDemand $\times$ minutesPerStop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Validating Against the Benchmark File

    To validate our formula, we compare it against the course's reference file, `shifts_with_costs.csv`. This comparison is restricted to reachable links only, since these are the only links our cost formula actually produces and the only ones that feed into the model to be solved. The benchmark's placeholder values for unreachable links are not used by our model, so including them in the comparison would provide no information about the correctness of our formula.
    """)
    return


@app.cell
def _(cost_base, mo, shifts_ref_long):
    benchmark = shifts_ref_long.copy()
    comparison = cost_base.merge(
        benchmark[["warehouseID", "regionID", "transportationCosts"]],
        on=["warehouseID", "regionID"],
        how="left",
        suffixes=("_own", "_benchmark"),
    )

    # Sanity check: hat jede unserer reachable rows eine Entsprechung in der Referenz?
    n_missing = comparison["transportationCosts_benchmark"].isna().sum()

    comparison["rel_diff"] = (
        (comparison["transportationCosts_own"] - comparison["transportationCosts_benchmark"]).abs()
        / comparison["transportationCosts_benchmark"]
    )
    _stats = comparison["rel_diff"].describe()

    mo.md(f"""
    Reachable rows without a benchmark match: **{n_missing}** of {len(comparison):,}

    Relative difference to benchmark (`rel_diff`):

    | Statistic | Value |
    |---|---:|
    | Count | {_stats['count']:.0f} |
    | Mean | {_stats['mean']:.2e} |
    | Std | {_stats['std']:.2e} |
    | Min | {_stats['min']:.2e} |
    | 25% | {_stats['25%']:.2e} |
    | Median (50%) | {_stats['50%']:.2e} |
    | 75% | {_stats['75%']:.2e} |
    | Max | {_stats['max']:.2e} |
    """)
    return (comparison,)


@app.cell
def _(comparison, plt):
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(comparison['transportationCosts_benchmark'], comparison['transportationCosts_own'], alpha=0.4, s=15)
    lims = [comparison['transportationCosts_benchmark'].min(), comparison['transportationCosts_benchmark'].max()]
    _ax.plot(lims, lims, color='red', linestyle='--', linewidth=1, label='perfect agreement (x = y)')
    _ax.set_xlabel('Benchmark cost (€)')
    _ax.set_ylabel('Our cost (€)')
    _ax.set_xscale('log')
    _ax.set_yscale('log')
    _ax.set_title('Validation: our formula vs. the benchmark file\n(2,661 reachable links)')
    # x = y Referenzlinie: perfekte Übereinstimmung läge exakt auf dieser Geraden
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The relative difference between our formula and the benchmark is effectively zero across all 2,661 reachable links. The median, the 25th percentile, and even the minimum are exactly zero, and the largest observed difference (approximately $4 \times 10^{-16}$) is negligible and can reasonably be attributed to floating-point rounding rather than a meaningful discrepancy. In the scatter plot below, every point falls exactly on the $x = y$ line (shown on a log-log scale); no data point deviates visibly from perfect agreement, consistent with the near-zero relative differences reported above. These results indicate that our derived cost formula reproduces the benchmark values exactly for all practical purposes, providing strong evidence that the derivation is correct rather than merely a close approximation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Network Optimization

    A model cannot be solved without its cost data, so this section first examines the fixed-cost structure of the network, then builds two optimization models on top of it.

    The first model is a data-based baseline. It uses the estimated transportation costs and the existing annual fixed costs from the warehouse data, with each center treated as a single constant fixed cost. Because no physical capacity information is available, this model does not impose warehouse-capacity constraints.

    The second model removes the structural flaw that the baseline still carries. It makes the size of every center a decision, so that each location can be closed or operated at one of five size tiers, each with its own fixed cost, its own variable processing cost per delivery, and its own volume range. The cost figures for these tiers are derived directly from CashLog's own warehouse data.

    The comparison of both models shows how strongly the network recommendation depends on making capacity and volume-dependent processing costs part of the decision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.1 The Fixed-Cost Structure

    Before any model is solved, the fixed-cost data is examined, since it drives both formulations that follow and a model cannot be solved without it. The dataset does not contain a direct measure of each center's physical capacity, so the fixed cost of each warehouse is used as a proxy for its size. Fixed cost is typically driven by square footage, electricity, salaries, and regional price levels, all of which are broadly indicators of physical size.
    """)
    return


@app.cell
def _(plt, warehouses_flat):
    _sorted_costs = (
        warehouses_flat["fixedCosts"]
        .sort_values()
        .reset_index(drop=True)
    )

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(
        range(len(_sorted_costs)),
        _sorted_costs,
        marker="o",
    )
    _ax.set_xlabel("Warehouses, sorted by fixed cost")
    _ax.set_ylabel("Fixed cost (€)")
    _ax.set_title(
        "Natural breaks in the fixed-cost distribution"
    )
    plt.show()
    return


@app.cell
def _(mo, warehouses_flat):
    _counts = (
        warehouses_flat["fixedCosts"]
        .value_counts()
        .sort_index()
    )
    _n_unique = warehouses_flat["fixedCosts"].nunique()

    _table_rows = "\n".join(
        f"| €{_cost:,.0f} | {_n} |"
        for _cost, _n in _counts.items()
    )

    mo.md(f"""
    Distinct fixed-cost values: **{_n_unique}**

    | Fixed cost | Number of centers |
    |---:|---:|
    {_table_rows}

    Sum of counts: **{_counts.sum()}** of 42 centers.
    """)
    return


@app.cell
def _(warehouses_flat):
    tier_by_fixed_cost = {
        1_344_000: "v",
        2_580_000: "s",
        4_572_000: "m",
        15_012_000: "l",
        35_904_000: "h",
    }

    warehouses_flat["tier"] = warehouses_flat["fixedCosts"].map(tier_by_fixed_cost)
    warehouses_flat[["warehouseID", "city", "fixedCosts", "tier"]]
    return (tier_by_fixed_cost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gap analysis reveals that all 42 warehouses fall into exactly five distinct fixed-cost levels.

    These results provide strong, data-driven support for using five size tiers. While the two largest tiers each contain only a single warehouse, they are retained as separate categories rather than being merged, since their fixed costs (€15.0M and €35.9M) represent clearly distinct capacity levels rather than minor variation around a common value.

    Each warehouse is mapped to its size tier directly via its fixed-cost value. Since there are exactly five distinct values, this mapping is exact, rather than an approximation or a clustering result with ambiguous boundary cases.
    """)
    return


@app.cell
def _(regions_flat, shifts_long, warehouses_flat):
    BASE_SHIFT_MINUTES = 450
    BASE_SHIFT_COST = 480

    WAREHOUSE_IDS = warehouses_flat["warehouseID"].tolist()
    REGION_IDS = regions_flat["regionID"].tolist()

    DEMAND_BY_REGION = (
        regions_flat
        .set_index("regionID")["yearlyDemand"]
        .to_dict()
    )

    STOP_TIME_BY_REGION = (
        regions_flat
        .set_index("regionID")["minutesPerStop"]
        .to_dict()
    )

    FIXED_COST_BY_WAREHOUSE = (
        warehouses_flat
        .set_index("warehouseID")["fixedCosts"]
        .to_dict()
    )

    CITY_BY_WAREHOUSE = (
        warehouses_flat
        .set_index("warehouseID")["city"]
        .to_dict()
    )


    def build_scenario_links(
        demand_factor=1.0,
        shift_cost=BASE_SHIFT_COST,
        shift_minutes=BASE_SHIFT_MINUTES,
        travel_factor=1.0,
    ):
        """Recalculate reachable links and annual transportation costs."""

        link_data = shifts_long[
            ["warehouseID", "regionID", "travelTime"]
        ].copy()

        link_data["effective_travel_time"] = (
            link_data["travelTime"] * travel_factor
        )

        link_data["usable_minutes"] = (
            shift_minutes
            - 2 * link_data["effective_travel_time"]
        )

        link_data = link_data[
            link_data["usable_minutes"] > 0
        ].copy()

        link_data["scenario_demand"] = (
            link_data["regionID"].map(DEMAND_BY_REGION)
            * demand_factor
        )

        link_data["minutesPerStop"] = (
            link_data["regionID"].map(STOP_TIME_BY_REGION)
        )

        link_data["transportationCosts"] = (
            shift_cost
            * link_data["scenario_demand"]
            * link_data["minutesPerStop"]
            / link_data["usable_minutes"]
        )

        reachable_regions = set(link_data["regionID"])

        missing_regions = [
            region_id
            for region_id in REGION_IDS
            if region_id not in reachable_regions
        ]

        return link_data, missing_regions

    return (
        BASE_SHIFT_COST,
        BASE_SHIFT_MINUTES,
        CITY_BY_WAREHOUSE,
        DEMAND_BY_REGION,
        FIXED_COST_BY_WAREHOUSE,
        REGION_IDS,
        WAREHOUSE_IDS,
        build_scenario_links,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 Data-Based Baseline Model

    The baseline model minimizes annual transportation costs and the
    existing fixed costs of open cash centers.

    The decision variable \(x_{ij}\) equals one if region \(j\) is assigned
    to cash center \(i\). The variable \(y_i\) equals one if location \(i\)
    remains open.

    \[
    \min
    \sum_{(i,j)} c_{ij}x_{ij}
    +
    \sum_i f_i y_i
    \]

    Each region must be assigned to exactly one reachable open location.
    No capacity restriction is imposed because the available warehouse
    data do not contain physical throughput capacities.
    """)
    return


@app.cell
def _(
    BASE_SHIFT_COST,
    BASE_SHIFT_MINUTES,
    FIXED_COST_BY_WAREHOUSE,
    REGION_IDS,
    WAREHOUSE_IDS,
    build_scenario_links,
    pl,
):
    def solve_basic_network(
        demand_factor=1.0,
        shift_cost=BASE_SHIFT_COST,
        shift_minutes=BASE_SHIFT_MINUTES,
        travel_factor=1.0,
        fixed_cost_factor=1.0,
    ):
        link_data, missing_regions = build_scenario_links(
            demand_factor=demand_factor,
            shift_cost=shift_cost,
            shift_minutes=shift_minutes,
            travel_factor=travel_factor,
        )

        if missing_regions:
            return {
                "status": "Infeasible: unreachable regions",
                "missing_regions": missing_regions,
                "total_cost": None,
                "transportation_cost": None,
                "fixed_cost": None,
                "n_open": None,
                "open_centers": [],
                "assignment": [],
            }

        link_keys = list(
            zip(
                link_data["warehouseID"],
                link_data["regionID"],
            )
        )

        links_by_region = {
            region_id: []
            for region_id in REGION_IDS
        }

        for warehouse_id, region_id in link_keys:
            links_by_region[region_id].append(warehouse_id)

        transport_cost_lookup = (
            link_data
            .set_index(["warehouseID", "regionID"])
            ["transportationCosts"]
            .to_dict()
        )

        model = pl.LpProblem(
            "CashLog_Basic_Model",
            pl.LpMinimize,
        )

        x_basic = pl.LpVariable.dicts(
            "x_basic",
            link_keys,
            cat=pl.LpBinary,
        )

        y_basic = pl.LpVariable.dicts(
            "y_basic",
            WAREHOUSE_IDS,
            cat=pl.LpBinary,
        )

        transportation_expression = pl.lpSum(
            transport_cost_lookup[(warehouse_id, region_id)]
            * x_basic[(warehouse_id, region_id)]
            for warehouse_id, region_id in link_keys
        )

        fixed_cost_expression = pl.lpSum(
            FIXED_COST_BY_WAREHOUSE[warehouse_id]
            * fixed_cost_factor
            * y_basic[warehouse_id]
            for warehouse_id in WAREHOUSE_IDS
        )

        model += (
            transportation_expression
            + fixed_cost_expression
        )

        for warehouse_id, region_id in link_keys:
            model += (
                x_basic[(warehouse_id, region_id)]
                <= y_basic[warehouse_id]
            )

        for region_id in REGION_IDS:
            model += pl.lpSum(
                x_basic[(warehouse_id, region_id)]
                for warehouse_id
                in links_by_region[region_id]
            ) == 1

        model.solve(pl.HiGHS(msg=False))

        status_name = pl.LpStatus[model.status]

        if status_name != "Optimal":
            return {
                "status": status_name,
                "total_cost": None,
                "transportation_cost": None,
                "fixed_cost": None,
                "n_open": None,
                "open_centers": [],
                "assignment": [],
            }

        open_centers = [
            warehouse_id
            for warehouse_id in WAREHOUSE_IDS
            if (y_basic[warehouse_id].value() or 0) > 0.5
        ]

        assignment = [
            (warehouse_id, region_id)
            for warehouse_id, region_id in link_keys
            if (
                x_basic[(warehouse_id, region_id)].value()
                or 0
            ) > 0.5
        ]

        return {
            "status": status_name,
            "total_cost": pl.value(model.objective),
            "transportation_cost": pl.value(
                transportation_expression
            ),
            "fixed_cost": pl.value(
                fixed_cost_expression
            ),
            "n_open": len(open_centers),
            "open_centers": open_centers,
            "assignment": assignment,
            "link_data": link_data,
        }

    return (solve_basic_network,)


@app.cell
def _(CITY_BY_WAREHOUSE, REGION_IDS, mo, solve_basic_network):
    basic_result = solve_basic_network()

    mo.stop(
        basic_result["status"] != "Optimal",
        mo.md(
            f"Basic model status: "
            f"**{basic_result['status']}**"
        ),
    )

    basic_open_cities = sorted(
        CITY_BY_WAREHOUSE[warehouse_id]
        for warehouse_id
        in basic_result["open_centers"]
    )

    mo.md(f"""
    ### Baseline Results

    - Solver status: **{basic_result["status"]}**
    - Total annual cost: **€{basic_result["total_cost"]:,.0f}**
    - Transportation cost: **€{basic_result["transportation_cost"]:,.0f}**
    - Existing facility fixed cost: **€{basic_result["fixed_cost"]:,.0f}**
    - Open locations: **{basic_result["n_open"]} of 42**
    - Assigned regions: **{len(basic_result["assignment"])} of {len(REGION_IDS)}**

    Open locations:

    {", ".join(basic_open_cities)}
    """)
    return (basic_result,)


@app.cell
def _(REGION_IDS, basic_result, mo):
    assert basic_result["status"] == "Optimal"
    assert len(basic_result["assignment"]) == len(REGION_IDS)
    assert len(basic_result["open_centers"]) == basic_result["n_open"]

    mo.md(
        "✅ Every customer region is assigned to exactly one open "
        "and reachable cash center."
    )
    return


@app.cell
def _(basic_result, pd, regions_flat, warehouses_flat):
    basic_assignment_df = pd.DataFrame(
        basic_result["assignment"],
        columns=["warehouseID", "regionID"],
    )

    basic_volume_by_center = (
        basic_assignment_df
        .merge(
            regions_flat[
                ["regionID", "yearlyDemand"]
            ],
            on="regionID",
        )
        .groupby("warehouseID")["yearlyDemand"]
        .sum()
        .rename("assigned_volume")
        .reset_index()
        .merge(
            warehouses_flat[
                ["warehouseID", "city", "fixedCosts", "tier"]
            ],
            on="warehouseID",
        )
        .sort_values(
            "assigned_volume",
            ascending=False,
        )
    )

    basic_volume_by_center
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The baseline model identifies the cost-minimizing network under the assumption that every open cash center can process an unlimited amount of demand. Under this assumption, the solver recommends keeping only 18 of the 42 existing locations open while assigning every one of the 515 customer regions to one of these centers.

    The solution achieves this primarily by reducing fixed facility costs. Since no capacity restrictions exist, there is no disadvantage to concentrating demand in a small number of locations as long as the resulting increase in transportation cost is outweighed by the savings from closing additional cash centers.

    This behavior is consistent with the mathematical formulation of the model. The optimizer only trades off transportation costs against fixed facility costs, while implicitly assuming unlimited processing capacity at every open location.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Capacity-Constrained Network Model

    The baseline model assumes that every open cash center can process an unlimited amount of demand. While this assumption simplifies the optimization problem, it allows the optimizer to concentrate arbitrarily large demand volumes in a small number of facilities.

    To obtain an operationally feasible network, the model is extended by introducing explicit capacity limits. Five cash-center tiers are defined, representing increasing processing capacities from very small to huge facilities. Each warehouse location may be assigned to one of these tiers, and the corresponding lower and upper volume bounds ensure that every open facility operates within a realistic capacity range.

    At this stage, the original CashLog fixed costs remain unchanged. The sole purpose of this extension is to isolate the effect of introducing capacity constraints before refining the cost structure in the following section.
    """)
    return


@app.cell
def _(pd, tier_by_fixed_cost, warehouses_flat):
    # Capacity bounds from the course template.
    # Each warehouse keeps the tier inferred from its observed fixed cost.

    CAPACITY_BY_TIER = {
        "v": {
            "lower_bound": 0,
            "upper_bound": 19_348,
        },
        "s": {
            "lower_bound": 19_349,
            "upper_bound": 45_415,
        },
        "m": {
            "lower_bound": 45_416,
            "upper_bound": 107_327,
        },
        "l": {
            "lower_bound": 107_328,
            "upper_bound": 199_999,
        },
        "h": {
            "lower_bound": 200_000,
            "upper_bound": 99_999_999,
        },
    }

    # Tier assigned to each warehouse based on its observed fixed-cost level
    TIER_BY_WAREHOUSE = (
        warehouses_flat
        .set_index("warehouseID")["tier"]
        .to_dict()
    )

    # Capacity limits for each individual warehouse
    LOWER_CAPACITY_BY_WAREHOUSE = {
        warehouse_id: CAPACITY_BY_TIER[tier]["lower_bound"]
        for warehouse_id, tier in TIER_BY_WAREHOUSE.items()
    }

    UPPER_CAPACITY_BY_WAREHOUSE = {
        warehouse_id: CAPACITY_BY_TIER[tier]["upper_bound"]
        for warehouse_id, tier in TIER_BY_WAREHOUSE.items()
    }

    # Table for displaying the tier definitions
    capacity_tier_table = (
        pd.DataFrame(CAPACITY_BY_TIER)
        .T
        .rename_axis("tier")
        .reset_index()
    )

    capacity_tier_table["fixed_cost"] = (
        capacity_tier_table["tier"]
        .map({
            tier: fixed_cost
            for fixed_cost, tier in tier_by_fixed_cost.items()
        })
    )

    capacity_tier_table = capacity_tier_table[
        [
            "tier",
            "fixed_cost",
            "lower_bound",
            "upper_bound",
        ]
    ]

    capacity_tier_table
    return (
        CAPACITY_BY_TIER,
        LOWER_CAPACITY_BY_WAREHOUSE,
        TIER_BY_WAREHOUSE,
        UPPER_CAPACITY_BY_WAREHOUSE,
    )


@app.cell
def _(
    BASE_SHIFT_COST,
    BASE_SHIFT_MINUTES,
    DEMAND_BY_REGION,
    FIXED_COST_BY_WAREHOUSE,
    LOWER_CAPACITY_BY_WAREHOUSE,
    REGION_IDS,
    TIER_BY_WAREHOUSE,
    UPPER_CAPACITY_BY_WAREHOUSE,
    WAREHOUSE_IDS,
    build_scenario_links,
    pl,
):
    def solve_capacity_network(
        demand_factor=1.0,
        shift_cost=BASE_SHIFT_COST,
        shift_minutes=BASE_SHIFT_MINUTES,
        travel_factor=1.0,
        fixed_cost_factor=1.0,
    ):
        """
        Solve the capacity-constrained network model.

        Each warehouse keeps the tier inferred from its observed fixed cost.
        The tier determines its lower and upper volume bounds.
        The objective still contains only transportation costs and the
        original CashLog fixed costs.
        """

        link_data, missing_regions = build_scenario_links(
            demand_factor=demand_factor,
            shift_cost=shift_cost,
            shift_minutes=shift_minutes,
            travel_factor=travel_factor,
        )

        if missing_regions:
            return {
                "status": "Infeasible: unreachable regions",
                "missing_regions": missing_regions,
                "total_cost": None,
                "transportation_cost": None,
                "fixed_cost": None,
                "processing_cost": 0,
                "n_open": None,
                "open_centers": [],
                "selected_type": {},
                "assigned_volume": {},
                "assignment": [],
                "link_data": link_data,
                "mip_gap": float("nan"),
            }

        link_keys = list(
            zip(
                link_data["warehouseID"],
                link_data["regionID"],
            )
        )

        links_by_region = {
            region_id: []
            for region_id in REGION_IDS
        }

        links_by_center = {
            warehouse_id: []
            for warehouse_id in WAREHOUSE_IDS
        }

        for warehouse_id, region_id in link_keys:
            links_by_region[region_id].append(
                warehouse_id
            )
            links_by_center[warehouse_id].append(
                region_id
            )

        transport_cost_lookup = (
            link_data
            .set_index(["warehouseID", "regionID"])[
                "transportationCosts"
            ]
            .to_dict()
        )

        scenario_demand = {
            region_id: (
                DEMAND_BY_REGION[region_id]
                * demand_factor
            )
            for region_id in REGION_IDS
        }

        model = pl.LpProblem(
            "CashLog_Capacity_Model",
            pl.LpMinimize,
        )

        x_capacity = pl.LpVariable.dicts(
            "x_capacity",
            link_keys,
            cat=pl.LpBinary,
        )

        y_capacity = pl.LpVariable.dicts(
            "y_capacity",
            WAREHOUSE_IDS,
            cat=pl.LpBinary,
        )

        transportation_expression = pl.lpSum(
            transport_cost_lookup[
                (warehouse_id, region_id)
            ]
            * x_capacity[
                (warehouse_id, region_id)
            ]
            for warehouse_id, region_id in link_keys
        )

        fixed_cost_expression = pl.lpSum(
            FIXED_COST_BY_WAREHOUSE[warehouse_id]
            * fixed_cost_factor
            * y_capacity[warehouse_id]
            for warehouse_id in WAREHOUSE_IDS
        )

        model += (
            transportation_expression
            + fixed_cost_expression
        )

        # A region may only be assigned to an open warehouse
        for warehouse_id, region_id in link_keys:
            model += (
                x_capacity[
                    (warehouse_id, region_id)
                ]
                <= y_capacity[warehouse_id]
            )

        # Every region must be assigned to exactly one reachable warehouse
        for region_id in REGION_IDS:
            model += pl.lpSum(
                x_capacity[
                    (warehouse_id, region_id)
                ]
                for warehouse_id
                in links_by_region[region_id]
            ) == 1

        # Assigned volume must remain within the bounds of the
        # warehouse's fixed, data-derived tier
        for warehouse_id in WAREHOUSE_IDS:

            assigned_volume_expression = pl.lpSum(
                scenario_demand[region_id]
                * x_capacity[
                    (warehouse_id, region_id)
                ]
                for region_id
                in links_by_center[warehouse_id]
            )

            model += (
                assigned_volume_expression
                >= LOWER_CAPACITY_BY_WAREHOUSE[warehouse_id]
                * y_capacity[warehouse_id]
            )

            model += (
                assigned_volume_expression
                <= UPPER_CAPACITY_BY_WAREHOUSE[warehouse_id]
                * y_capacity[warehouse_id]
            )

        available_solvers = pl.listSolvers(
            onlyAvailable=True
        )

        if "HiGHS" not in available_solvers:
            raise RuntimeError(
                "HiGHS is not available. "
                "Install it with: pip install highspy"
            )

        model.solve(
            pl.HiGHS(msg=False)
        )

        status_name = pl.LpStatus[
            model.status
        ]

        if status_name != "Optimal":
            return {
                "status": status_name,
                "missing_regions": [],
                "total_cost": None,
                "transportation_cost": None,
                "fixed_cost": None,
                "processing_cost": 0,
                "n_open": None,
                "open_centers": [],
                "selected_type": {},
                "assigned_volume": {},
                "assignment": [],
                "link_data": link_data,
            }

        open_centers = [
            warehouse_id
            for warehouse_id in WAREHOUSE_IDS
            if (
                y_capacity[warehouse_id].value()
                or 0
            ) > 0.5
        ]

        selected_type = {
            warehouse_id: TIER_BY_WAREHOUSE[warehouse_id]
            for warehouse_id in open_centers
        }

        assigned_volume = {
            warehouse_id: sum(
                scenario_demand[region_id]
                * (
                    x_capacity[
                        (warehouse_id, region_id)
                    ].value()
                    or 0
                )
                for region_id
                in links_by_center[warehouse_id]
            )
            for warehouse_id in open_centers
        }

        assignment = [
            (warehouse_id, region_id)
            for warehouse_id, region_id in link_keys
            if (
                x_capacity[
                    (warehouse_id, region_id)
                ].value()
                or 0
            ) > 0.5
        ]

        return {
            "status": status_name,
            "missing_regions": [],
            "total_cost": pl.value(
                model.objective
            ),
            "transportation_cost": pl.value(
                transportation_expression
            ),
            "fixed_cost": pl.value(
                fixed_cost_expression
            ),
            "processing_cost": 0,
            "n_open": len(open_centers),
            "open_centers": open_centers,
            "selected_type": selected_type,
            "assigned_volume": assigned_volume,
            "assignment": assignment,
            "link_data": link_data,
        }

    return (solve_capacity_network,)


@app.cell
def _(mo, pd, solve_capacity_network):
    capacity_result = solve_capacity_network()

    mo.stop(
        capacity_result["status"] != "Optimal",
        mo.md(
            f"Capacity-model status: "
            f"**{capacity_result['status']}**"
        ),
    )

    capacity_type_counts = (
        pd.Series(
            capacity_result["selected_type"],
            name="tier",
        )
        .value_counts()
        .sort_index()
    )

    capacity_type_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initial result

    The capacity-constrained model is infeasible. Since every customer region has already been verified to have at least one reachable cash center, infeasibility cannot be explained by the reachability constraints alone.

    We therefore investigate whether the newly introduced capacity limits are responsible. Specifically, we check whether every customer region can be fully assigned to at least one reachable cash center without violating its maximum processing capacity.
    """)
    return


@app.cell
def _(
    DEMAND_BY_REGION,
    TIER_BY_WAREHOUSE,
    UPPER_CAPACITY_BY_WAREHOUSE,
    build_scenario_links,
    regions_flat,
):
    # Diagnose: Kann jede Region vollständig von mindestens einem
    # erreichbaren Warehouse innerhalb dessen Kapazität übernommen werden?

    base_link_data, missing_regions = build_scenario_links()

    reachable_capacity_check = (
        base_link_data[
            ["warehouseID", "regionID"]
        ]
        .assign(
            warehouse_tier=lambda df:
                df["warehouseID"].map(TIER_BY_WAREHOUSE),
            warehouse_capacity=lambda df:
                df["warehouseID"].map(
                    UPPER_CAPACITY_BY_WAREHOUSE
                ),
            region_demand=lambda df:
                df["regionID"].map(DEMAND_BY_REGION),
        )
    )

    reachable_capacity_check["can_serve_region"] = (
        reachable_capacity_check["warehouse_capacity"]
        >= reachable_capacity_check["region_demand"]
    )

    region_capacity_diagnostic = (
        reachable_capacity_check
        .groupby("regionID")
        .agg(
            region_demand=("region_demand", "first"),
            n_reachable_centers=("warehouseID", "nunique"),
            max_reachable_capacity=(
                "warehouse_capacity",
                "max",
            ),
            n_capacity_feasible_centers=(
                "can_serve_region",
                "sum",
            ),
        )
        .reset_index()
    )

    infeasible_regions = (
        region_capacity_diagnostic[
            region_capacity_diagnostic[
                "n_capacity_feasible_centers"
            ] == 0
        ]
        .merge(
            regions_flat[
                ["regionID", "city", "yearlyDemand"]
            ],
            on="regionID",
            how="left",
        )
        .sort_values(
            "region_demand",
            ascending=False,
        )
    )

    infeasible_regions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Diagnosing the infeasibility

    The analysis reveals that the infeasibility is indeed caused by the capacity constraints rather than by the optimization algorithm.

    For example, customer region 410 (Zaragoza) has an annual demand of 59,646 deliveries. However, the largest reachable cash center has a maximum capacity of only 45,415 deliveries. Since each region must be assigned entirely to a single cash center, no feasible assignment exists.

    This is not an isolated modeling error but a consequence of combining two assumptions that were not originally intended to be used together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 A Data-Driven Capacity and Cost Model

    The infeasibility of the previous formulation highlights an important limitation. Capacity bounds alone are insufficient when each warehouse is permanently tied to its observed size. In practice, however, CashLog is not restricted to operating every location at its current scale. Existing facilities can be expanded or downsized through investments in equipment, workforce, and infrastructure.

    We therefore extend the model by allowing the optimizer to determine not only which cash centers remain open, but also at which operational size they should be operated. Each location may be assigned to one of five capacity tiers, thereby restoring the flexibility required for the capacity bounds to function as intended.

    Unlike the original course template, however, the associated cost structure is not adopted directly. Instead, the fixed and variable processing costs of each tier are calibrated from the observed CashLog fixed-cost data. This calibration preserves the empirically observed cost differences between facilities while introducing economies of scale through a volume-dependent processing-cost component.

    The resulting model simultaneously determines

    * which locations remain open,
    * which operational size each open location should adopt,
    * how customer regions are assigned, and
    * the corresponding transportation and processing costs.

    Compared with the baseline formulation, this model provides a substantially more realistic representation of CashLog’s operational decision problem, as it combines geographical assignment, capacity planning, and economies of scale within a single optimization framework.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variable Processing Cost

    Having established that fixed warehouse tiers alone lead to an infeasible model, we now extend the formulation by allowing each location to operate at different capacity levels. This requires assigning realistic fixed and variable processing costs to every tier. While the fixed costs are directly calibrated from the observed CashLog data, the variable processing costs must be derived separately, as they are not available in the dataset.

    Rather than assigning arbitrary processing costs to each tier, we derive all values from a single data-driven anchor. The observed fixed-cost ratios between tiers determine their relative scale, while a dampening function models the expected economies of scale.

    Note the unit: $c_t^{var}$ denotes cost *per individual customer visit*, not per shift or per tour — unlike $c_{ij}$, which already aggregates many visits into a single shift cost.
    """)
    return


@app.cell
def _(mo):
    fixed_by_tier = {"v": 1_344_000, "s": 2_580_000, "m": 4_572_000, "l": 15_012_000, "h": 35_904_000}
    scale_factor = {t: f / fixed_by_tier["v"] for t, f in fixed_by_tier.items()}

    _rows = "\n".join(
        f"| {t} | €{fixed_by_tier[t]:,.0f} | {s:.2f}x |"
        for t, s in scale_factor.items()
    )

    mo.md(f"""
    | Tier | Fixed cost | Scaling factor (relative to "v") |
    |---|---:|---:|
    {_rows}
    """)
    return fixed_by_tier, scale_factor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The smallest warehouse tier (v) is used as the reference tier, with a scaling factor of 1.0. The scaling factors for the remaining tiers are obtained by dividing each tier's fixed cost by that of the smallest tier, yielding a relative measure of size across all tiers.


    ### Dampening the Scaling Effect

    Having derived the fixed-cost ratios between tiers, we now determine how strongly variable cost per delivery should decrease as warehouse size increases. A one-to-one translation of the fixed-cost ratio into variable cost savings would be unrealistically strong. While fixed costs primarily reflect investments in capacity (e.g., warehouse space, equipment, and security infrastructure), operational efficiencies per delivery are subject to physical and organizational limits and therefore cannot improve proportionally with warehouse size. To account for this, the scaling effect is dampened rather than applying the fixed-cost ratios directly, representing economies of scale in a more realistic manner. This approach intentionally separates the relative cost differences between tiers from the absolute processing-cost level. The relative differences are determined by the observed fixed-cost ratios, whereas the absolute level is calibrated afterwards using the transport-cost data.

    This is implemented using a power function. Let $s_t$ denote the fixed-cost scaling factor of tier $t$ relative to the smallest warehouse tier. The variable processing cost per delivery is then calculated as

    $$c_t^{var} = c_v^{var} \cdot s_t^{-\alpha}$$

    where $c_v^{var}$ is the variable processing cost per delivery of the reference tier, and $\alpha$ is the dampening exponent. The parameter $\alpha$ controls the strength of the economies of scale: $\alpha = 0$ implies no scaling effect at all, $\alpha = 1$ applies the fixed-cost ratio directly, and intermediate values produce a more moderate, plausible reduction. Since $\alpha$ is the only free parameter of the function and has a clear interpretation, it can be systematically varied in the sensitivity analysis to assess the robustness of the model.

    We set $\alpha = 0.5$ (square-root dampening) as our base case, representing a moderate and commonly used degree of scale sensitivity — strong enough to reflect real economies of scale, but far short of a full 1:1 translation of the fixed-cost ratio. $\alpha$ is treated as an explicit assumption and varied later in the sensitivity analysis.

    ### Deriving $c_v^{var}$

    The scaled variable processing costs depend on the initial reference value $c_v^{var}$, which must therefore be a sensible and defensible figure. $c_v^{var}$ is initialized using our own transport cost as an internal reference point: for each reachable link, the transport cost per individual delivery ($c_{ij}/\text{yearlyDemand}_j$) can be computed.

    In-house processing (counting, sorting) requires no vehicle or fuel, and plausibly less staff time per delivery than a driven tour stop, so it is expected to cost less than transport per delivery. $c_v^{var}$ is therefore set at a sensible fraction of the median transport-per-delivery figure. The percentage itself remains an assumption, since no data separates processing labor from transport labor, but the anchor point it scales from is fully derived from our own data rather than an external or invented number. The impact of this assumption is examined later in the sensitivity analysis.
    """)
    return


@app.cell
def _(mo, scale_factor):
    damp_scale_factor = {t: f ** (-0.5) for t, f in scale_factor.items()}

    _rows = "\n".join(
        f"| {t} | {scale_factor[t]:.2f}x | {s:.3f} |"
        for t, s in damp_scale_factor.items()
    )

    mo.md(f"""
    | Tier | Fixed-cost scaling factor | Dampened factor ($s_t^{{-0.5}}$) |
    |---|---:|---:|
    {_rows}
    """)
    return


@app.cell
def _(cost_base, mo, regions_flat):
    merged = cost_base.merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
    merged["per_delivery_transport_cost"] = merged["transportationCosts"] / merged["yearlyDemand"]

    _stats = merged["per_delivery_transport_cost"].describe()

    mo.md(f"""
    Per-delivery transport cost (`transportationCosts / yearlyDemand`):

    | Statistic | Value |
    |---|---:|
    | Count | {_stats['count']:.0f} |
    | Mean | €{_stats['mean']:.2f} |
    | Std | €{_stats['std']:.2f} |
    | Min | €{_stats['min']:.2f} |
    | 25% | €{_stats['25%']:.2f} |
    | Median (50%) | €{_stats['50%']:.2f} |
    | 75% | €{_stats['75%']:.2f} |
    | Max | €{_stats['max']:.2f} |
    """)
    return (merged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The median transport cost per delivery across all 2,661 reachable links is €47.12. In-house processing cost is estimated at roughly **15%** of this transport-per-delivery figure, reflecting that the bulk of a transport stop's cost consists of driving and logistics overhead rather than the few minutes of actual cash handling. This gives:

    $$c_v^{var} = 0.15 \times 47.12\,\text{€} \approx 7.07\,\text{€ per delivery}$$

    The scaled processing costs can now be calculated accordingly.
    """)
    return


@app.cell
def _(merged, mo, scale_factor):
    alpha = 0.5       # dampening exponent
    median_transport_per_delivery = merged["per_delivery_transport_cost"].median()
    processing_share = 0.15  # Anteil der Transport-pro-Lieferung-Kosten
    c_var_anchor = processing_share * median_transport_per_delivery

    c_var_by_tier = {t: c_var_anchor / (scale_factor[t] ** alpha) for t in scale_factor}

    _rows = "\n".join(f"| {t} | €{c:.2f} |" for t, c in c_var_by_tier.items())

    mo.md(f"""
    Anchor value $c_v^{{var}}$ (15% of median transport-per-delivery, €{median_transport_per_delivery:.2f}): **€{c_var_anchor:.2f} per delivery**

    | Tier | Variable processing cost $c_t^{{var}}$ |
    |---|---:|
    {_rows}
    """)
    return c_var_by_tier, median_transport_per_delivery


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The resulting variable processing costs range from €7.07 per delivery for the smallest tier down to €1.37 for the largest — a 5.17x reduction, matching $\sqrt{s_h} \approx 5.17$ as expected under $\alpha = 0.5$. This is substantially smaller than the 26.7x difference in fixed costs between the same two tiers, confirming that the dampening meaningfully moderates the raw fixed-cost ratio rather than passing it through unchanged. All values remain well below the median transport cost per delivery (€47.12), consistent with in-house processing being cheaper per unit than a driven tour stop.

    The derived variable processing costs are combined with the calibrated fixed-cost structure in the following section to formulate the final upgradeable CashLog optimization model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solving the model
    """)
    return


@app.cell
def _(
    BASE_SHIFT_COST,
    BASE_SHIFT_MINUTES,
    CAPACITY_BY_TIER,
    DEMAND_BY_REGION,
    REGION_IDS,
    WAREHOUSE_IDS,
    build_scenario_links,
    fixed_by_tier,
    median_transport_per_delivery,
    pl,
    scale_factor,
):
    REALISTIC_TYPE_IDS = tuple(
        CAPACITY_BY_TIER.keys()
    )


    def solve_realistic_network(
        demand_factor=1.0,
        shift_cost=BASE_SHIFT_COST,
        shift_minutes=BASE_SHIFT_MINUTES,
        travel_factor=1.0,
        fixed_cost_factor=1.0,
        processing_cost_factor=1.0,
        alpha=0.5,
        processing_share=0.15,
        warm_start_from=None,
        gap_rel=0.005,
        time_limit=300,
    ):
        """
        Solve the upgradeable CashLog network model.

        For every warehouse location, the optimizer decides whether the
        location remains closed or operates as one of five size tiers.

        The objective includes:
        1. annual transportation costs,
        2. tier-specific fixed costs,
        3. tier-specific variable processing costs.

        Each selected tier imposes corresponding lower and upper
        volume bounds.
        """

        link_data, missing_regions = build_scenario_links(
            demand_factor=demand_factor,
            shift_cost=shift_cost,
            shift_minutes=shift_minutes,
            travel_factor=travel_factor,
        )

        if missing_regions:
            return {
                "status": "Infeasible: unreachable regions",
                "missing_regions": missing_regions,
                "total_cost": None,
                "transportation_cost": None,
                "center_fixed_cost": None,
                "processing_cost": None,
                "n_open": None,
                "open_centers": [],
                "selected_type": {},
                "assigned_volume": {},
                "assignment": [],
                "link_data": link_data,
            }

        link_keys = list(
            zip(
                link_data["warehouseID"],
                link_data["regionID"],
            )
        )

        type_keys = [
            (warehouse_id, center_type)
            for warehouse_id in WAREHOUSE_IDS
            for center_type in REALISTIC_TYPE_IDS
        ]

        links_by_region = {
            region_id: []
            for region_id in REGION_IDS
        }

        links_by_center = {
            warehouse_id: []
            for warehouse_id in WAREHOUSE_IDS
        }

        for warehouse_id, region_id in link_keys:
            links_by_region[region_id].append(
                warehouse_id
            )
            links_by_center[warehouse_id].append(
                region_id
            )

        transport_cost_lookup = (
            link_data
            .set_index(
                ["warehouseID", "regionID"]
            )["transportationCosts"]
            .to_dict()
        )

        scenario_demand = {
            region_id: (
                DEMAND_BY_REGION[region_id]
                * demand_factor
            )
            for region_id in REGION_IDS
        }

        # Recalculate tier-specific variable processing costs
        # for the current sensitivity scenario.

        scenario_c_var_anchor = (
            processing_share
            * median_transport_per_delivery
        )

        scenario_c_var_by_tier = {
            center_type: (
                scenario_c_var_anchor
                / (
                    scale_factor[center_type]
                    ** alpha
                )
            )
            for center_type in REALISTIC_TYPE_IDS
        }

        total_scenario_demand = sum(
            scenario_demand.values()
        )

        model = pl.LpProblem(
            "CashLog_Realistic_Upgradeable_Model",
            pl.LpMinimize,
        )

        # 1 if region j is assigned to warehouse i
        x_realistic = pl.LpVariable.dicts(
            "x_realistic",
            link_keys,
            cat=pl.LpBinary,
        )

        # 1 if warehouse i operates as tier t
        y_realistic = pl.LpVariable.dicts(
            "y_realistic",
            type_keys,
            cat=pl.LpBinary,
        )

        # Volume processed by warehouse i under tier t
        z_realistic = pl.LpVariable.dicts(
            "z_realistic",
            type_keys,
            lowBound=0,
            cat=pl.LpContinuous,
        )

        # ---------------------------------------------------------
        # Objective-function components
        # ---------------------------------------------------------

        transportation_expression = pl.lpSum(
            transport_cost_lookup[
                (warehouse_id, region_id)
            ]
            * x_realistic[
                (warehouse_id, region_id)
            ]
            for warehouse_id, region_id in link_keys
        )

        center_fixed_expression = pl.lpSum(
            fixed_by_tier[center_type]
            * fixed_cost_factor
            * y_realistic[
                (warehouse_id, center_type)
            ]
            for warehouse_id in WAREHOUSE_IDS
            for center_type in REALISTIC_TYPE_IDS
        )

        processing_expression = pl.lpSum(
            scenario_c_var_by_tier[center_type]
            * processing_cost_factor
            * z_realistic[
                (warehouse_id, center_type)
            ]
            for warehouse_id in WAREHOUSE_IDS
            for center_type in REALISTIC_TYPE_IDS
        )
        # print("Old:")
        # print(c_var_by_tier)

        # print("\nNew:")
        # print(scenario_c_var_by_tier)

        OBJECTIVE_SCALE = 1_000

        model += (
            transportation_expression
            + center_fixed_expression
            + processing_expression
        ) / OBJECTIVE_SCALE

        # ---------------------------------------------------------
        # Assignment constraints
        # ---------------------------------------------------------

        # A region can only be assigned to an open warehouse
        for warehouse_id, region_id in link_keys:
            model += (
                x_realistic[
                    (warehouse_id, region_id)
                ]
                <= pl.lpSum(
                    y_realistic[
                        (warehouse_id, center_type)
                    ]
                    for center_type in REALISTIC_TYPE_IDS
                )
            )

        # Every region must be assigned to exactly one reachable warehouse
        for region_id in REGION_IDS:
            model += pl.lpSum(
                x_realistic[
                    (warehouse_id, region_id)
                ]
                for warehouse_id
                in links_by_region[region_id]
            ) == 1

        # ---------------------------------------------------------
        # Tier-selection and capacity constraints
        # ---------------------------------------------------------

        # Tighter big-M: a warehouse can never receive more volume than the
        # total demand of the regions actually reachable from it.
        reachable_demand_by_center = {
            warehouse_id: sum(
                scenario_demand[region_id]
                for region_id in links_by_center[warehouse_id]
            )
            for warehouse_id in WAREHOUSE_IDS
        }

        for warehouse_id in WAREHOUSE_IDS:

            assigned_volume_expression = pl.lpSum(
                scenario_demand[region_id]
                * x_realistic[
                    (warehouse_id, region_id)
                ]
                for region_id
                in links_by_center[warehouse_id]
            )

            # Total warehouse volume must equal the volume allocated
            # to its selected tier
            model += pl.lpSum(
                z_realistic[
                    (warehouse_id, center_type)
                ]
                for center_type in REALISTIC_TYPE_IDS
            ) == assigned_volume_expression

            # Each location may operate as at most one tier
            model += pl.lpSum(
                y_realistic[
                    (warehouse_id, center_type)
                ]
                for center_type in REALISTIC_TYPE_IDS
            ) <= 1

            for center_type in REALISTIC_TYPE_IDS:

                lower_bound = (
                    CAPACITY_BY_TIER[
                        center_type
                    ]["lower_bound"]
                )

                upper_bound = min(
                    CAPACITY_BY_TIER[
                        center_type
                    ]["upper_bound"],
                    reachable_demand_by_center[warehouse_id],
                )

                # A warehouse whose reachable demand cannot even fill this
                # tier's lower bound can never operate at this tier.
                if (
                    reachable_demand_by_center[warehouse_id]
                    < lower_bound
                ):
                    model += (
                        y_realistic[
                            (warehouse_id, center_type)
                        ] == 0
                    )

                # Lower volume bound applies only if the tier is selected
                model += (
                    z_realistic[
                        (warehouse_id, center_type)
                    ]
                    >= lower_bound
                    * y_realistic[
                        (warehouse_id, center_type)
                    ]
                )

                # Upper capacity applies only if the tier is selected
                model += (
                    z_realistic[
                        (warehouse_id, center_type)
                    ]
                    <= upper_bound
                    * y_realistic[
                        (warehouse_id, center_type)
                    ]
                )

        # ---------------------------------------------------------
        # Solve
        # ---------------------------------------------------------

        available_solvers = pl.listSolvers(
            onlyAvailable=True
        )

        if "HiGHS" not in available_solvers:
            raise RuntimeError(
                "HiGHS is not available. "
                "Install it with: pip install highspy"
            )

        # ---------------------------------------------------------
        # Warm start: seed the solver with a known good solution.
        # Scenarios stay structurally close to the base case, so the
        # base solution is a strong incumbent and saves the solver the
        # initial search for any feasible solution at all.
        # ---------------------------------------------------------
        if warm_start_from is not None:
            seed_type = warm_start_from["selected_type"]
            seed_assignment = set(warm_start_from["assignment"])

            for warehouse_id, center_type in type_keys:
                y_realistic[
                    (warehouse_id, center_type)
                ].setInitialValue(
                    1 if seed_type.get(warehouse_id) == center_type
                    else 0
                )

            for warehouse_id, region_id in link_keys:
                x_realistic[
                    (warehouse_id, region_id)
                ].setInitialValue(
                    1 if (warehouse_id, region_id) in seed_assignment
                    else 0
                )

        model.solve(
            pl.HiGHS(
                msg=False,
                gapRel=gap_rel,
                timeLimit=time_limit,
                warmStart=warm_start_from is not None,
            )
        )

        # Capture the residual MIP gap so scenario precision is reported,
        # not assumed.
        try:
            _info = model.solverModel.getInfo()
            mip_gap = float(_info.mip_gap)
        except Exception:
            mip_gap = float("nan")

        print("HiGHS status:", model.solverModel.getModelStatus()
          if hasattr(model, "solverModel") else "n/a")

        status_name = pl.LpStatus[
            model.status
        ]

        if status_name != "Optimal":
            return {
                "status": status_name,
                "missing_regions": [],
                "total_cost": None,
                "transportation_cost": None,
                "center_fixed_cost": None,
                "processing_cost": None,
                "n_open": None,
                "open_centers": [],
                "selected_type": {},
                "assigned_volume": {},
                "assignment": [],
                "link_data": link_data,
                "mip_gap": mip_gap,
            }

        # ---------------------------------------------------------
        # Extract solution
        # ---------------------------------------------------------

        selected_type = {}

        for warehouse_id in WAREHOUSE_IDS:
            chosen_types = [
                center_type
                for center_type in REALISTIC_TYPE_IDS
                if (
                    y_realistic[
                        (warehouse_id, center_type)
                    ].value()
                    or 0
                ) > 0.5
            ]

            if chosen_types:
                selected_type[warehouse_id] = (
                    chosen_types[0]
                )

        open_centers = list(
            selected_type.keys()
        )

        assigned_volume = {
            warehouse_id: sum(
                (
                    z_realistic[
                        (warehouse_id, center_type)
                    ].value()
                    or 0
                )
                for center_type in REALISTIC_TYPE_IDS
            )
            for warehouse_id in open_centers
        }

        assignment = [
            (warehouse_id, region_id)
            for warehouse_id, region_id in link_keys
            if (
                x_realistic[
                    (warehouse_id, region_id)
                ].value()
                or 0
            ) > 0.5
        ]

        return {
            "status": status_name,
            "missing_regions": [],
            "total_cost": (
                pl.value(model.objective) * OBJECTIVE_SCALE
            ),
            "transportation_cost": pl.value(
                transportation_expression
            ),
            "center_fixed_cost": pl.value(
                center_fixed_expression
            ),
            "processing_cost": pl.value(
                processing_expression
            ),
            "n_open": len(open_centers),
            "open_centers": open_centers,
            "selected_type": selected_type,
            "assigned_volume": assigned_volume,
            "assignment": assignment,
            "link_data": link_data,
            "mip_gap": mip_gap,
        }

    return REALISTIC_TYPE_IDS, solve_realistic_network


@app.cell
def _(REALISTIC_TYPE_IDS, mo, pd, solve_realistic_network):
    realistic_result = solve_realistic_network()

    mo.stop(
        realistic_result["status"] != "Optimal",
        mo.md(
            f"Realistic-model status: "
            f"**{realistic_result['status']}**"
        ),
    )

    realistic_type_counts = (
        pd.Series(
            realistic_result["selected_type"],
            name="center_type",
        )
        .value_counts()
        .reindex(
            REALISTIC_TYPE_IDS,
            fill_value=0,
        )
        .rename_axis("Center type")
        .reset_index(name="Number of centers")
    )

    realistic_type_counts
    return (realistic_result,)


@app.cell
def _(CITY_BY_WAREHOUSE, REGION_IDS, WAREHOUSE_IDS, mo, realistic_result):
    realistic_open_cities = sorted(
        CITY_BY_WAREHOUSE[warehouse_id]
        for warehouse_id in realistic_result["open_centers"]
    )

    mo.md(f"""
    ### Realistic Model Results

    | Metric | Result |
    |:---|---:|
    | Solver status | **{realistic_result["status"]}** |
    | Total annual cost | **€{realistic_result["total_cost"]:,.0f}** |
    | Transportation cost | **€{realistic_result["transportation_cost"]:,.0f}** |
    | Center fixed cost | **€{realistic_result["center_fixed_cost"]:,.0f}** |
    | Variable processing cost | **€{realistic_result["processing_cost"]:,.0f}** |
    | Open locations | **{realistic_result["n_open"]} of {len(WAREHOUSE_IDS)}** |
    | Assigned regions | **{len(realistic_result["assignment"])} of {len(REGION_IDS)}** |

    #### Open locations

    {", ".join(realistic_open_cities)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation of the realistic model

    The upgradeable model produces a substantially different network from the baseline formulation. Instead of concentrating demand in a small number of facilities, the optimizer selects 25 of the 42 available warehouse locations while simultaneously determining the most economical operating size for each of them.

    ### Cost structure

    Although the total annual cost increases from the baseline solution to €202 million, this should not be interpreted as a deterioration of the network. The baseline model omitted several important operational costs and assumed unlimited processing capacity. The higher objective value therefore represents the cost of operating a network that is both economically and operationally feasible.

    The cost breakdown also provides useful insight into the main cost drivers. Fixed facility costs account for approximately 68 % of the total objective, transportation costs for 28 %, and variable processing costs for only about 4.5 %. This confirms that infrastructure decisions dominate the optimization problem, while the processing-cost component primarily serves to distinguish economically attractive warehouse sizes rather than driving the objective on its own.

    ### Selected center sizes

    The selected tier distribution further supports the plausibility of the solution.

    |Tier | n_centers|
    |---|---|
    |v	|1|
    |s	|8|
    |m	|14|
    |l	|1|
    |h	|1|

    The optimizer strongly favors medium-sized facilities (14 of 25), supplemented by a substantial group of small centers, while opening only one large and one huge center. This is an intuitive outcome. Very small centers cannot efficiently absorb large demand volumes, whereas very large centers incur substantially higher fixed investment despite benefiting from lower processing costs per delivery. Medium-sized facilities therefore provide the best trade-off between capacity, transportation efficiency, and operating cost for much of the network.

    The resulting network is consequently far more balanced than the baseline solution and reflects a realistic mixture of regional and high-capacity facilities rather than relying on only a few oversized hubs.

    Computational note. The calibrated cost structure introduces a 26.7× spread between the smallest and largest tier's fixed cost. This makes every tier decision an economically substantive trade-off — which is the entire point of the extension — but it also weakens the LP relaxation and makes the model considerably harder to solve than a formulation with flatter cost differences. Three modelling refinements were required to obtain a proven optimum: tightening the big-M bound on each location to the total demand of the regions actually reachable from it, pruning tier variables that a location's reachable demand can never fill, and scaling the objective to thousands of euros for numerical stability. With these in place, the model solves to proven optimality in approximately three minutes.

    ### Assignment Summay
    """)
    return


@app.cell
def _(
    CAPACITY_BY_TIER,
    c_var_by_tier,
    fixed_by_tier,
    mo,
    pd,
    realistic_result,
    regions_flat,
    warehouses_flat,
):
    realistic_assignment_df = pd.DataFrame(
        realistic_result["assignment"],
        columns=["warehouseID", "regionID"],
    )

    realistic_center_summary = (
        realistic_assignment_df
        .merge(
            regions_flat[
                ["regionID", "yearlyDemand"]
            ],
            on="regionID",
            how="left",
        )
        .groupby("warehouseID")
        .agg(
            assigned_volume=(
                "yearlyDemand",
                "sum",
            ),
            assigned_regions=(
                "regionID",
                "nunique",
            ),
        )
        .reset_index()
        .merge(
            warehouses_flat[
                [
                    "warehouseID",
                    "city",
                ]
            ],
            on="warehouseID",
            how="left",
        )
    )

    realistic_center_summary["selected_tier"] = (
        realistic_center_summary["warehouseID"]
        .map(realistic_result["selected_type"])
    )

    realistic_center_summary["variable_processing_cost"] = (
        realistic_center_summary["selected_tier"]
        .map(c_var_by_tier)
    )

    realistic_center_summary["upper_capacity"] = (
        realistic_center_summary["selected_tier"]
        .map(
            lambda tier: CAPACITY_BY_TIER[tier]["upper_bound"]
        )
    )

    realistic_center_summary["capacity_utilization"] = (
        realistic_center_summary["assigned_volume"]
        / realistic_center_summary["upper_capacity"]
    )

    realistic_center_summary["tier_fixed_cost"] = (
        realistic_center_summary["selected_tier"]
        .map(fixed_by_tier)
    )

    realistic_center_summary = (
        realistic_center_summary[
            [
                "warehouseID",
                "city",
                "selected_tier",
                "assigned_regions",
                "assigned_volume",
                "upper_capacity",
                "capacity_utilization",
                "tier_fixed_cost",
                "variable_processing_cost",
            ]
        ]
        .sort_values(
            "assigned_volume",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    realistic_center_summary_display = (
        realistic_center_summary.copy()
    )

    realistic_center_summary_display["assigned_volume"] = (
        realistic_center_summary_display[
            "assigned_volume"
        ]
        .map(lambda value: f"{value:,.0f}")
    )

    realistic_center_summary_display["tier_fixed_cost"] = (
        realistic_center_summary_display[
            "tier_fixed_cost"
        ]
        .map(lambda value: f"€{value:,.0f}")
    )

    realistic_center_summary_display[
        "variable_processing_cost"
    ] = (
        realistic_center_summary_display[
            "variable_processing_cost"
        ]
        .map(lambda value: f"€{value:.2f}")
    )

    realistic_center_summary_display["upper_capacity"] = (
        realistic_center_summary_display[
            "upper_capacity"
        ].map(lambda value: f"{value:,.0f}")
    )

    realistic_center_summary_display["capacity_utilization"] = (
        realistic_center_summary_display[
            "capacity_utilization"
        ].map(lambda value: f"{100*value:.1f}%")
    )

    mo.ui.table(
        realistic_center_summary_display,
        selection=None,
        pagination=True,
    )
    return (realistic_assignment_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The volume summary shows that the optimizer uses the available capacity very efficiently. Almost all small, medium, and large facilities operate close to their respective capacity limits, indicating that the selected tier sizes are economically well matched to the assigned demand. Rather than opening unnecessarily large facilities, the optimizer consistently chooses the smallest tier capable of accommodating the allocated volume.

    The only exception is the single huge center, which exhibits a very low utilization rate. This is expected, as the upper capacity of the largest tier is intentionally set to a very large value to avoid imposing an artificial upper limit on the highest-capacity facilities. Consequently, its utilization percentage is not directly comparable to the remaining tiers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Network Visualization
    """)
    return


@app.cell
def _(
    CITY_BY_WAREHOUSE,
    folium,
    mcolors,
    mo,
    plt,
    realistic_assignment_df,
    realistic_result,
    regions_flat,
    warehouses_flat,
):
    # ---------------------------------------------------------
    # Shared data and colors
    # ---------------------------------------------------------

    realistic_open_centers = (
        realistic_result["open_centers"]
    )

    realistic_open_center_set = set(
        realistic_open_centers
    )

    realistic_assignment_map_df = (
        realistic_assignment_df
        .merge(
            regions_flat[
                [
                    "regionID",
                    "city",
                    "lat",
                    "lon",
                    "yearlyDemand",
                ]
            ],
            on="regionID",
            how="left",
        )
    )

    map_center = [
        warehouses_flat["lat"].mean(),
        warehouses_flat["lon"].mean(),
    ]

    # One distinct color for every open center
    realistic_cmap = plt.get_cmap(
        "gist_ncar",
        len(realistic_open_centers),
    )

    color_by_center = {
        warehouse_id: mcolors.to_hex(
            realistic_cmap(position)
        )
        for position, warehouse_id
        in enumerate(realistic_open_centers)
    }


    # ---------------------------------------------------------
    # Map 1: Open and closed centers
    # ---------------------------------------------------------

    open_closed_map = folium.Map(
        location=map_center,
        zoom_start=6,
        tiles="cartodbpositron",
    )

    for row in warehouses_flat.itertuples():

        is_open = (
            row.warehouseID
            in realistic_open_center_set
        )

        selected_tier = (
            realistic_result["selected_type"]
            .get(row.warehouseID)
        )

        assigned_volume = (
            realistic_result["assigned_volume"]
            .get(row.warehouseID, 0)
        )

        marker_color = (
            color_by_center[row.warehouseID]
            if is_open
            else "#9e9e9e"
        )

        folium.CircleMarker(
            location=[
                row.lat,
                row.lon,
            ],
            radius=9 if is_open else 5,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.9 if is_open else 0.35,
            weight=2 if is_open else 1,
            popup=folium.Popup(
                f"""
                <b>{row.city}</b><br>
                Status: {"Open" if is_open else "Closed"}<br>
                Tier: {selected_tier if is_open else "–"}<br>
                Assigned volume: {assigned_volume:,.0f}
                """,
                max_width=250,
            ),
            tooltip=(
                f"{row.city}: "
                f"{'Open' if is_open else 'Closed'}"
            ),
        ).add_to(open_closed_map)


    # ---------------------------------------------------------
    # Map 2: Demand regions colored by assigned center
    # ---------------------------------------------------------

    assignment_map = folium.Map(
        location=map_center,
        zoom_start=6,
        tiles="cartodbpositron",
    )

    for row in (
        realistic_assignment_map_df.itertuples()
    ):

        assigned_center_city = (
            CITY_BY_WAREHOUSE[
                row.warehouseID
            ]
        )

        region_color = (
            color_by_center[
                row.warehouseID
            ]
        )

        folium.CircleMarker(
            location=[
                row.lat,
                row.lon,
            ],
            radius=3,
            color=region_color,
            fill=True,
            fill_color=region_color,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(
                f"""
                <b>Region {row.regionID}</b><br>
                City: {row.city}<br>
                Annual demand: {row.yearlyDemand:,.0f}<br>
                Assigned center: {assigned_center_city}
                """,
                max_width=250,
            ),
            tooltip=(
                f"Region {row.regionID} → "
                f"{assigned_center_city}"
            ),
        ).add_to(assignment_map)

    # Add open cash centers using the same assignment colors
    for row in warehouses_flat.itertuples():

        if (
            row.warehouseID
            not in realistic_open_center_set
        ):
            continue

        center_color = (
            color_by_center[
                row.warehouseID
            ]
        )

        selected_tier = (
            realistic_result["selected_type"][
                row.warehouseID
            ]
        )

        assigned_volume = (
            realistic_result["assigned_volume"][
                row.warehouseID
            ]
        )

        folium.CircleMarker(
            location=[
                row.lat,
                row.lon,
            ],
            radius=9,
            color="#000000",
            fill=True,
            fill_color=center_color,
            fill_opacity=1,
            weight=2,
            popup=folium.Popup(
                f"""
                <b>{row.city}</b><br>
                Tier: {selected_tier}<br>
                Assigned volume: {assigned_volume:,.0f}
                """,
                max_width=250,
            ),
            tooltip=f"Cash Center: {row.city}",
        ).add_to(assignment_map)


    # ---------------------------------------------------------
    # Display both maps in one output cell
    # ---------------------------------------------------------

    mo.ui.tabs(
        {
            "Open / closed centers": open_closed_map,
            "Regional assignments": assignment_map,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Model Comparison
    """)
    return


@app.cell
def _(basic_result, mo, pd, realistic_result):
    model_comparison = pd.DataFrame(
        [
            {
                "Model": "Baseline",
                "Total annual cost": basic_result["total_cost"],
                "Transportation cost": basic_result["transportation_cost"],
                "Center fixed cost": basic_result["fixed_cost"],
                "Variable processing cost": 0,
                "Open locations": basic_result["n_open"],
            },
            {
                "Model": "Realistic upgradeable model",
                "Total annual cost": realistic_result["total_cost"],
                "Transportation cost": realistic_result["transportation_cost"],
                "Center fixed cost": realistic_result["center_fixed_cost"],
                "Variable processing cost": realistic_result["processing_cost"],
                "Open locations": realistic_result["n_open"],
            },
        ]
    )

    model_comparison_display = model_comparison.copy()

    for column in [
        "Total annual cost",
        "Transportation cost",
        "Center fixed cost",
        "Variable processing cost",
    ]:
        model_comparison_display[column] = (
            model_comparison_display[column]
            .map(lambda value: f"€{value:,.0f}")
        )

    mo.ui.table(model_comparison_display)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table summarizes the effect of progressively extending the optimization model. The baseline formulation identifies the lowest-cost network under the simplifying assumption of unlimited processing capacity and fixed facility sizes. The realistic model, in contrast, jointly optimizes facility size, capacity, and variable processing costs.

    As expected, introducing these additional operational considerations increases the total annual cost from €98.6 million to €202.0 million and raises the number of open cash centers from 18 to 25.  This increase should not be interpreted as a deterioration of the solution. Rather, it reflects the cost of operating a network that is both operationally feasible and economically more realistic.

    The comparison also reveals an interesting shift in the cost structure. Transportation cost decreases from €66.3 million to €56.0 million, indicating that the larger number of open facilities allows customer regions to be served from geographically closer locations. These savings are offset by substantially higher facility costs, as the model now explicitly accounts for realistic capacity planning and introduces variable processing costs. Consequently, the optimization no longer minimizes transportation and facility count alone but instead balances transportation efficiency, processing efficiency, and infrastructure investment.

    Overall, the resulting network represents a considerably more balanced solution than the baseline model. Rather than concentrating demand in a small number of inexpensive facilities, the optimizer distributes demand across a larger set of appropriately sized cash centers while exploiting economies of scale where economically beneficial.

    Having established a realistic optimization model, the remaining question is no longer how to formulate the network design problem, but how robust the resulting solution is with respect to the assumptions underlying the cost model. In particular, the variable processing costs, the degree of economies of scale, and several operational parameters were derived from reasonable but ultimately uncertain assumptions. The following sensitivity analysis therefore examines how changes in these assumptions affect both the total cost and the structure of the recommended network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sensitivity Analysis

    The realistic upgradeable model rests on assumptions about demand, costs, and technology that are inherently uncertain. The sensitivity analysis therefore tests how robust the recommended network is when these assumptions are varied.

    The analysis focuses deliberately on external factors that CashLog does not control:

    - the secular decline in cash usage
    -  rising transport costs
    -  technological change

    rather than on the full space of internal calibration parameters. This is also due to the solver taking a long time per solve. Solving more than 15 scenarions with 3 minutes per solve would take up to an hour. A single model-assumption scenario is included as a robustness check on the economies-of-scale exponent, which is the one assumption that directly shapes the tier structure.
    The fixed-cost data is not varied, because it is observed rather than assumed. The processing-cost anchor is not varied separately either: it scales ctvarc_t^{var}
    ctvar​ linearly, exactly as a proportional processing-cost shock would, and its structural effect is already captured by the α\alpha
    α scenario.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Central Scenario Definitions

    The base-case network is solved to proven optimality. The scenarios, by contrast, are solved with a 1 % relative MIP gap and a 120-second time limit to keep the analysis computationally tractable. Their solutions are therefore near-optimal rather than proven optima, and the residual gap is reported for every scenario so that its precision can be judged rather than assumed.

    The analysis reads the results accordingly. Its purpose is to identify directional effects, whether an external shock pushes the network towards consolidation or towards densification, and by roughly how much total cost changes, not to certify the exact optimal facility count under each scenario. Differences of a single facility between two scenarios lie within the solver tolerance and are not interpreted as meaningful.
    """)
    return


@app.cell
def _(BASE_SHIFT_COST, BASE_SHIFT_MINUTES, mo):
    # ---------------------------------------------------------
    # 6.1 Scenario definitions
    # ---------------------------------------------------------
    # Each scenario maps a label to the kwargs passed to
    # solve_realistic_network(). The base case is NOT re-solved; it is
    # reused from realistic_result.

    SENS_GAP_REL = 0.01
    SENS_TIME_LIMIT = 120

    SCENARIO_FAMILIES = {
        "Declining demand": {
            "Strong decline (-30%)": {"demand_factor": 0.7},
            "Severe decline (-50%)": {"demand_factor": 0.5},
        },
        "Rising transport cost": {
            "Strong increase (+50%)": {
                "shift_cost": BASE_SHIFT_COST * 1.5
            },
        },
        "Technology": {
            "Extended operating time (900 min)": {
                "shift_minutes": 900,
                "shift_cost": BASE_SHIFT_COST * (900 / BASE_SHIFT_MINUTES),
            },
        },
        "Model assumptions": {
            "Full scale effect (alpha=1.0)": {"alpha": 1.0},
        },
    }

    n_scenarios = sum(len(f) for f in SCENARIO_FAMILIES.values())

    mo.md(
        "| Scenario family | Scenario | Varied parameter |\n"
        "|---|---|---|\n"
        + "\n".join(
            f"| {family} | {label} | "
            f"{', '.join(f'`{k}`' for k in kwargs)} |"
            for family, scenarios in SCENARIO_FAMILIES.items()
            for label, kwargs in scenarios.items()
        )
        + f"\n\n**{n_scenarios} scenarios** (excl. base case), "
        f"solved at a {SENS_GAP_REL:.0%} MIP gap, "
        f"{SENS_TIME_LIMIT}s limit each."
    )
    return SCENARIO_FAMILIES, SENS_GAP_REL, SENS_TIME_LIMIT, n_scenarios


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Central Solve Run

    All scenarios are solved once, in a single button-triggered cell, and cached. Every subsequent subsection reads from this cache rather than invoking the solver again. Each scenario is warm-started from the base-case solution, which is structurally close to most scenarios and shortens the solver's initial search.
    """)
    return


@app.cell
def _(SENS_TIME_LIMIT, mo, n_scenarios):
    run_sensitivity = mo.ui.run_button(
        label="Run sensitivity analysis"
    )

    mo.md(f"""
    Solving **{n_scenarios}** scenarios
    (worst case ~{n_scenarios * SENS_TIME_LIMIT / 60:.0f} minutes).

    {run_sensitivity}
    """)
    return (run_sensitivity,)


@app.cell
def _(
    SCENARIO_FAMILIES,
    SENS_GAP_REL,
    SENS_TIME_LIMIT,
    mo,
    pd,
    realistic_result,
    run_sensitivity,
    solve_realistic_network,
):
    mo.stop(not run_sensitivity.value)

    import time as _time

    scenario_results = {}
    _rows = []

    for _family, _scenarios in SCENARIO_FAMILIES.items():

        scenario_results[_family] = {}

        for _label, _kwargs in _scenarios.items():

            _start = _time.perf_counter()

            _result = solve_realistic_network(
                warm_start_from=realistic_result,
                gap_rel=SENS_GAP_REL,
                time_limit=SENS_TIME_LIMIT,
                **_kwargs,
            )

            _runtime = _time.perf_counter() - _start
            scenario_results[_family][_label] = _result

            _rows.append({
                "family": _family,
                "scenario": _label,
                "status": _result["status"],
                "mip_gap": _result["mip_gap"],
                "runtime_seconds": round(_runtime, 1),
            })

            print(
                f"[{_family}] {_label}: "
                f"gap={_result['mip_gap']:.2%} "
                f"({_runtime:.0f}s)",
                flush=True,
            )

    sensitivity_runtime = pd.DataFrame(_rows)

    mo.md(f"""
    **Solve complete.** {len(_rows)} scenarios,
    total runtime **{sensitivity_runtime['runtime_seconds'].sum() / 60:.1f} min**.

    Largest residual MIP gap: **{sensitivity_runtime['mip_gap'].max():.2%}**
    """)
    return scenario_results, sensitivity_runtime


@app.cell
def _(CITY_BY_WAREHOUSE, REALISTIC_TYPE_IDS, REGION_IDS, pd):
    def summarize_scenario(family, label, result, base):
        """Build one summary row, including structural change metrics."""

        if result["status"] != "Optimal":
            return {
                "family": family,
                "scenario": label,
                "status": result["status"],
                "mip_gap": result.get("mip_gap", float("nan")),
                "total_cost": None,
                "cost_change": None,
                "transport_cost": None,
                "fixed_cost": None,
                "processing_cost": None,
                "n_open": None,
                "centers_changed": None,
                "regions_reassigned": None,
                "tier_mix": "–",
                "opened_vs_base": "–",
                "closed_vs_base": "–",
            }

        open_now = set(result["open_centers"])
        open_base = set(base["open_centers"])

        opened = sorted(
            CITY_BY_WAREHOUSE[w] for w in open_now - open_base
        )
        closed = sorted(
            CITY_BY_WAREHOUSE[w] for w in open_base - open_now
        )

        # Structural metric 1: how many facilities flipped
        centers_changed = len(open_now ^ open_base)

        # Structural metric 2: share of regions served by a different center
        base_map = dict(
            (r, w) for w, r in base["assignment"]
        )
        now_map = dict(
            (r, w) for w, r in result["assignment"]
        )
        reassigned = sum(
            1
            for region_id in REGION_IDS
            if base_map.get(region_id) != now_map.get(region_id)
        )

        tier_counts = pd.Series(
            result["selected_type"]
        ).value_counts()

        tier_mix = " ".join(
            f"{t}:{tier_counts.get(t, 0)}"
            for t in REALISTIC_TYPE_IDS
        )

        return {
            "family": family,
            "scenario": label,
            "status": result["status"],
            "mip_gap": result["mip_gap"],
            "total_cost": result["total_cost"],
            "cost_change": (
                result["total_cost"] / base["total_cost"] - 1
            ),
            "transport_cost": result["transportation_cost"],
            "fixed_cost": result["center_fixed_cost"],
            "processing_cost": result["processing_cost"],
            "n_open": result["n_open"],
            "centers_changed": centers_changed,
            "regions_reassigned": reassigned / len(REGION_IDS),
            "tier_mix": tier_mix,
            "opened_vs_base": ", ".join(opened) or "–",
            "closed_vs_base": ", ".join(closed) or "–",
        }

    return (summarize_scenario,)


@app.cell
def _(
    mo,
    pd,
    realistic_result,
    scenario_results,
    sensitivity_runtime,
    summarize_scenario,
):
    mo.stop("scenario_results" not in globals())

    _summary_rows = [
        summarize_scenario(
            "Base case",
            "Base case",
            realistic_result,
            realistic_result,
        )
    ]

    for _family, _scenarios in scenario_results.items():
        for _label, _result in _scenarios.items():
            _summary_rows.append(
                summarize_scenario(
                    _family,
                    _label,
                    _result,
                    realistic_result,
                )
            )

    sensitivity_summary = pd.DataFrame(_summary_rows)

    # Merge runtimes
    sensitivity_summary = sensitivity_summary.merge(
        sensitivity_runtime[
            ["family", "scenario", "runtime_seconds"]
        ],
        on=["family", "scenario"],
        how="left",
    )

    mo.stop("scenario_results" not in globals())

    _summary_rows = [
        summarize_scenario(
            "Base case",
            "Base case",
            realistic_result,
            realistic_result,
        )
    ]

    for _family, _scenarios in scenario_results.items():
        for _label, _result in _scenarios.items():
            _summary_rows.append(
                summarize_scenario(
                    _family,
                    _label,
                    _result,
                    realistic_result,
                )
            )

    sensitivity_summary = pd.DataFrame(_summary_rows)

    sensitivity_summary = sensitivity_summary.merge(
        sensitivity_runtime[
            ["family", "scenario", "runtime_seconds"]
        ],
        on=["family", "scenario"],
        how="left",
    )

    mo.ui.table(
        sensitivity_summary,
        selection=None,
        pagination=False,
    )
    return (sensitivity_summary,)


@app.cell
def _(mo, pd, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    _display = sensitivity_summary.copy()

    for _col in [
        "total_cost",
        "transport_cost",
        "fixed_cost",
        "processing_cost",
    ]:
        _display[_col] = _display[_col].map(
            lambda v: f"€{v/1e6:,.1f}M" if pd.notna(v) else "–"
        )

    _display["cost_change"] = _display["cost_change"].map(
        lambda v: f"{v:+.1%}" if pd.notna(v) else "–"
    )
    _display["mip_gap"] = _display["mip_gap"].map(
        lambda v: f"{v:.2%}" if pd.notna(v) else "–"
    )
    _display["regions_reassigned"] = _display[
        "regions_reassigned"
    ].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")

    mo.ui.table(
        _display[
            [
                "family",
                "scenario",
                "total_cost",
                "cost_change",
                "n_open",
                "tier_mix",
                "centers_changed",
                "regions_reassigned",
                "mip_gap",
            ]
        ],
        selection=None,
        pagination=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Declining Cash Demand

    Electronic payment methods continue to displace physical currency. This family tests demand reductions of 30 % and 50 % — long-term stress-test assumptions rather than forecasts for a specific year.
    The relevant question is not whether cost falls (it must), but how: does lower demand lead CashLog to close facilities, or to keep the same locations and operate them at smaller tiers? The two answers imply very different strategies.
    """)
    return


@app.cell
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    sensitivity_summary[
        sensitivity_summary["family"].isin(
            ["Base case", "Declining demand"]
        )
    ][
        [
            "scenario",
            "total_cost",
            "cost_change",
            "transport_cost",
            "fixed_cost",
            "n_open",
            "tier_mix",
            "closed_vs_base",
        ]
    ]
    return


@app.cell(hide_code=True)
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    import re


    def _tier_count(tier_mix_str, tier):
        _match = re.search(rf"{tier}:(\d+)", tier_mix_str)
        return int(_match.group(1)) if _match else 0


    _base = sensitivity_summary[
        sensitivity_summary["scenario"] == "Base case"
    ].iloc[0]

    _strong = sensitivity_summary[
        sensitivity_summary["scenario"] == "Strong decline (-30%)"
    ].iloc[0]

    _severe = sensitivity_summary[
        sensitivity_summary["scenario"] == "Severe decline (-50%)"
    ].iloc[0]

    _base_v = _tier_count(_base["tier_mix"], "v")
    _strong_v = _tier_count(_strong["tier_mix"], "v")
    _base_m = _tier_count(_base["tier_mix"], "m")
    _severe_m = _tier_count(_severe["tier_mix"], "m")
    _severe_h = _tier_count(_severe["tier_mix"], "h")

    mo.md(f"""
    #### Interpretation

    The network downsizes rather than consolidates. Reducing demand by
    30 % lowers total cost by {abs(_strong['cost_change']):.1%}, and a
    50 % reduction lowers it by {abs(_severe['cost_change']):.1%} — less
    than proportionally, because fixed facility costs persist regardless
    of how few customer visits remain.

    The decisive observation, however, is not in the facility count but in
    the tier mix. The number of open centers barely moves
    ({_base['n_open']:.0f} → {_strong['n_open']:.0f} →
    {_severe['n_open']:.0f}), while the size distribution collapses
    downwards: the smallest tier grows from {_base_v} to {_strong_v}
    locations, and medium-sized centers fall from {_base_m} to
    {_severe_m}. Under a 50 % decline, even the single huge center is
    {'closed entirely' if _severe_h == 0 else f'reduced to {_severe_h}'}.

    CashLog's response to declining cash usage is therefore primarily one
    of downsizing existing facilities, not shutting them down. Geographic
    coverage remains valuable even at half the demand, because transport
    cost still scales with distance; what becomes uneconomical is
    operating large processing capacity that is no longer filled. Nearly
    half of all customer regions ({_strong['regions_reassigned']:.1%} and
    {_severe['regions_reassigned']:.1%}) are reassigned in the process,
    indicating that the assignment structure is far more fluid than the
    facility footprint.
    """)
    return (re,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 Rising Wages and Fuel Costs

    The €480 shift cost bundles wages, fuel, depreciation, and maintenance. No component-level breakdown exists, so the analysis varies the complete shift cost rather than estimating separate wage and fuel effects.
    Higher transport cost should make geographic proximity more valuable. If the model responds by opening more centers, that confirms the trade-off at the heart of the problem: fixed cost buys shorter distances.
    """)
    return


@app.cell
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    sensitivity_summary[
        sensitivity_summary["family"].isin(
            ["Base case", "Rising transport cost"]
        )
    ][
        [
            "scenario",
            "total_cost",
            "cost_change",
            "transport_cost",
            "fixed_cost",
            "n_open",
            "tier_mix",
            "opened_vs_base",
        ]
    ]
    return


@app.cell(hide_code=True)
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    _base = sensitivity_summary[
        sensitivity_summary["scenario"] == "Base case"
    ].iloc[0]

    _row = sensitivity_summary[
        sensitivity_summary["scenario"] == "Strong increase (+50%)"
    ].iloc[0]

    _transport_ratio = _row["transport_cost"] / _base["transport_cost"]

    mo.md(f"""
    #### Interpretation

    Higher transport cost densifies the network. A 50 % increase in shift
    cost raises total cost by {_row['cost_change']:.1%} and increases the
    number of open centers from {_base['n_open']:.0f} to
    {_row['n_open']:.0f} — a direct confirmation of the central trade-off:
    fixed cost buys shorter distances, and when distance becomes more
    expensive, that purchase becomes worthwhile.

    The response is visible in the cost structure. Transport cost rises
    from €{_base['transport_cost']/1e6:.1f}M to
    €{_row['transport_cost']/1e6:.1f}M — a factor of
    {_transport_ratio:.2f}, less than the 1.5 shock applied. The
    additional facilities absorb part of the increase by shortening
    average service distances, while fixed cost rises only modestly
    (€{_base['fixed_cost']/1e6:.1f}M → €{_row['fixed_cost']/1e6:.1f}M).

    Notably, this is the scenario with the largest cost change but the
    smallest structural change: only {_row['regions_reassigned']:.1%} of
    regions are reassigned. The network does not reorganize; it thickens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 Technological Development

    Technology is modelled through a longer usable shift: 900 minutes instead of 450. Shift cost is scaled proportionally, so the scenario models a genuinely longer productive shift rather than unpaid labour.
    This scenario differs structurally from all others: it changes the reachability matrix itself. At 900 minutes, links up to 450 minutes of one-way travel become feasible, where previously the limit was 225. Distant regions that no center could reach economically suddenly enter the solution space, and the round-trip time is amortised over a far larger number of stops.
    """)
    return


@app.cell
def _(mo, realistic_result, scenario_results):
    mo.stop("scenario_results" not in globals())

    _tech = scenario_results["Technology"][
        "Extended operating time (900 min)"
    ]

    _n_links_base = len(realistic_result["link_data"])
    _n_links_tech = len(_tech["link_data"])

    mo.md(f"""
    | | Base case | Extended operating time |
    |---|---:|---:|
    | Reachable links | {_n_links_base:,} | **{_n_links_tech:,}** |
    | Feasibility limit (one-way) | 225 min | **450 min** |
    | Open centers | {realistic_result['n_open']} | **{_tech['n_open']}** |
    | Total cost | €{realistic_result['total_cost']/1e6:,.1f}M | **€{_tech['total_cost']/1e6:,.1f}M** |
    """)
    return


@app.cell
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    sensitivity_summary[
        sensitivity_summary["family"].isin(
            ["Base case", "Technology"]
        )
    ][
        [
            "scenario",
            "total_cost",
            "cost_change",
            "transport_cost",
            "n_open",
            "tier_mix",
            "closed_vs_base",
        ]
    ]
    return


@app.cell(hide_code=True)
def _(mo, re, sensitivity_summary):
    mo.stop("scenario_results" not in globals())


    def _tier_count(tier_mix_str, tier):
        _match = re.search(rf"{tier}:(\d+)", tier_mix_str)
        return int(_match.group(1)) if _match else 0


    _base = sensitivity_summary[
        sensitivity_summary["scenario"] == "Base case"
    ].iloc[0]

    _row = sensitivity_summary[
        sensitivity_summary["scenario"] == "Extended operating time (900 min)"
    ].iloc[0]

    _base_s = _tier_count(_base["tier_mix"], "s")
    _row_s = _tier_count(_row["tier_mix"], "s")
    _base_m = _tier_count(_base["tier_mix"], "m")
    _row_m = _tier_count(_row["tier_mix"], "m")

    _other_gaps = sensitivity_summary[
        ~sensitivity_summary["scenario"].isin(
            ["Base case", "Extended operating time (900 min)"]
        )
    ]["mip_gap"]

    mo.md(f"""
    #### Interpretation

    Longer shifts enable consolidation. Doubling the usable shift to 900
    minutes lowers total cost by only {abs(_row['cost_change']):.1%}, but
    reduces the number of open centers from {_base['n_open']:.0f} to
    {_row['n_open']:.0f} and reassigns {_row['regions_reassigned']:.1%} of
    all customer regions, by far the largest structural upheaval of any
    scenario.

    The mechanism is the reachability matrix. At 450 minutes, a region is
    only reachable within 225 minutes of one-way travel; at 900 minutes,
    that limit doubles. Regions that previously had to be served by a
    nearby center can now be reached from a distant one, and the
    round-trip time is amortised over far more stops per shift. The tier
    mix confirms the consequence: small centers disappear entirely (s:
    {_base_s} to {_row_s}) and medium-sized centers absorb their volume
    (m: {_base_m} to {_row_m}).

    Interpretive caveat. This scenario is the computationally hardest of
    the five, because doubling reachability substantially enlarges the
    model. It terminated with a residual MIP gap of
    {_row['mip_gap']:.1%}, compared with {_other_gaps.min():.1%} to
    {_other_gaps.max():.1%} for all other scenarios. The direction of the
    effect, technology favours fewer, larger, more distant hubs, is
    robust, but the specific figures ({_row['n_open']:.0f} locations,
    €{_row['total_cost']/1e6:.1f}M) should be treated as indicative rather
    than precise. A tighter solve would likely find an even lower cost,
    strengthening rather than reversing the conclusion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.6 Robustness of the Economies-of-Scale Assumption

    The base case dampens the fixed-cost ratio with α=0.5\alpha = 0.5
    α=0.5. The extreme case α=1.0\alpha = 1.0
    α=1.0 passes the observed fixed-cost ratio through to variable processing cost unchanged, making large centers maximally attractive per delivery — a 26.7× advantage for the largest tier instead of 5.2×.
    If the network survives this, the tier structure does not depend on the dampening assumption, and the base case is safe.
    """)
    return


@app.cell
def _(mo, run_sensitivity, sensitivity_summary):
    mo.stop(not run_sensitivity.value)

    sensitivity_summary[
        sensitivity_summary["family"].isin(
            ["Base case", "Model assumptions"]
        )
    ][
        [
            "scenario",
            "total_cost",
            "cost_change",
            "processing_cost",
            "n_open",
            "tier_mix",
            "regions_reassigned",
        ]
    ]
    return


@app.cell(hide_code=True)
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    _base = sensitivity_summary[
        sensitivity_summary["scenario"] == "Base case"
    ].iloc[0]

    _row = sensitivity_summary[
        sensitivity_summary["scenario"] == "Full scale effect (alpha=1.0)"
    ].iloc[0]

    mo.md(f"""
    ### Interpretation

    Cost-robust, structurally fluid. Passing the fixed-cost ratio through
    undamped (α = 1.0, a 26.7× per-delivery advantage for the largest
    tier instead of 5.2×) changes total cost by just
    {_row['cost_change']:+.1%}. Variable processing cost falls from
    €{_base['processing_cost']/1e6:.1f}M to
    €{_row['processing_cost']/1e6:.1f}M, but because processing accounts
    for under 5 % of the objective, the total barely moves.

    The network structure, however, does move: {_row['n_open']:.0f}
    locations instead of {_base['n_open']:.0f},
    {_row['centers_changed']:.0f} facilities flipped, and
    {_row['regions_reassigned']:.1%} of regions reassigned. The dampening
    exponent is therefore not a driver of total cost, but it does shift
    which specific centers are selected.

    This is a useful result rather than a problem. It confirms that the
    base case's headline economics do not depend on the α = 0.5
    assumption, while flagging that individual facility decisions near
    the margin should not be attributed to the scale assumption alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.7 Cost Robustness vs. Structural Robustness

    A scenario can shift total cost substantially without changing the network, and a modest cost change can flip facility decisions when two solutions are nearly equivalent. These are different questions, and the recommendation depends on the second.
    """)
    return


@app.cell
def _(mo, plt, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    _plot_df = sensitivity_summary[
        sensitivity_summary["family"] != "Base case"
    ].copy()

    _fig, _ax = plt.subplots(figsize=(8, 5))

    _ax.scatter(
        _plot_df["cost_change"].abs(),
        _plot_df["regions_reassigned"],
        s=80,
        alpha=0.7,
    )

    for _row in _plot_df.itertuples():
        _ax.annotate(
            _row.scenario,
            (
                abs(_row.cost_change),
                _row.regions_reassigned,
            ),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    _ax.set_xlabel("Absolute change in total cost vs. base case")
    _ax.set_ylabel("Share of regions reassigned")
    _ax.set_title(
        "Cost robustness vs. structural robustness\n"
        "(top-left = network flips cheaply; "
        "bottom-right = expensive but structurally stable)"
    )
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    _transport = sensitivity_summary[
        sensitivity_summary["scenario"] == "Strong increase (+50%)"
    ].iloc[0]

    _scale = sensitivity_summary[
        sensitivity_summary["scenario"] == "Full scale effect (alpha=1.0)"
    ].iloc[0]

    _tech = sensitivity_summary[
        sensitivity_summary["scenario"] == "Extended operating time (900 min)"
    ].iloc[0]

    mo.md(f"""
    ### Interpretation

    |Scenario|Cost Change| Reassigned Regions |
    |---|---|---|
    |Rising transport cost|{_transport['cost_change']:+.1%}| {_transport['regions_reassigned']:.1%}
    |Full scale effect|{_scale['cost_change']:+.1%}|{_scale['regions_reassigned']:.1%}
    |Extended operating time| {_tech['cost_change']:+.1%}|{_tech['regions_reassigned']:.1%}

    The scatter reveals that cost sensitivity and structural sensitivity
    are largely decoupled, and in this network even inversely related.
    The scenario with the largest cost impact (rising transport cost,
    {_transport['cost_change']:+.1%}) produces the least structural
    change: the network simply thickens along its existing geography.
    Conversely, the scenarios with almost no cost impact, the scale-effect
    assumption ({_scale['cost_change']:+.1%}) and extended operating time
    ({_tech['cost_change']:+.1%}), trigger the largest reallocation of
    customer regions.

    The practical implication is direct. Forecasting total cost accurately
    requires getting demand and transport cost right. Deciding which
    specific centers to close requires getting the technology and
    capacity assumptions right, and these are precisely the assumptions
    with the weakest empirical foundation. Cost projections and closure
    decisions therefore carry different kinds of risk and should not be
    defended with the same evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.8 Robust Recomendation

    A location is considered robust when it is open in the base case and remains open across the external scenario families. Families are weighted equally, so a family with more tested values does not dominate.
    """)
    return


@app.cell
def _(
    CITY_BY_WAREHOUSE,
    WAREHOUSE_IDS,
    mo,
    np,
    pd,
    realistic_result,
    scenario_results,
):
    mo.stop("scenario_results" not in globals())

    _external = {
        f: s
        for f, s in scenario_results.items()
        if f != "Model assumptions"
    }

    _robust_rows = []

    for _w in WAREHOUSE_IDS:

        _base_open = _w in realistic_result["open_centers"]

        _family_rates = {}
        for _family, _scenarios in _external.items():
            _valid = [
                r
                for r in _scenarios.values()
                if r["status"] == "Optimal"
            ]
            _family_rates[_family] = (
                np.mean([_w in r["open_centers"] for r in _valid])
                if _valid
                else np.nan
            )

        _support = sum(
            rate >= 0.5
            for rate in _family_rates.values()
            if pd.notna(rate)
        )
        _n_families = sum(
            pd.notna(r) for r in _family_rates.values()
        )

        _robust_rows.append({
            "warehouseID": _w,
            "city": CITY_BY_WAREHOUSE[_w],
            "base_open": _base_open,
            **{
                f"{f}_rate": r
                for f, r in _family_rates.items()
            },
            "family_support": _support,
            "n_families": _n_families,
        })

    robustness_df = pd.DataFrame(_robust_rows)


    def classify(row):
        if row["base_open"] and row["family_support"] == row["n_families"]:
            return "Retain – robust"
        if row["base_open"] and row["family_support"] >= 1:
            return "Retain – conditional"
        if not row["base_open"] and row["family_support"] == 0:
            return "Closure candidate – robust"
        return "Scenario-dependent – verify before closure"


    robustness_df["recommendation"] = robustness_df.apply(
        classify, axis=1
    )

    mo.ui.table(
        robustness_df.sort_values(
            ["recommendation", "family_support", "city"],
            ascending=[True, False, True],
        ),
        selection=None,
        pagination=True,
    )
    return (robustness_df,)


@app.cell
def _(mo, robustness_df):
    mo.stop("scenario_results" not in globals())

    _groups = {
        label: sorted(
            robustness_df.loc[
                robustness_df["recommendation"] == label,
                "city",
            ]
        )
        for label in [
            "Retain – robust",
            "Retain – conditional",
            "Scenario-dependent – verify before closure",
            "Closure candidate – robust",
        ]
    }

    mo.md(
        "### Recommended Decision Structure\n\n"
        + "\n\n".join(
            f"**{label}** ({len(cities)})  \n"
            + (", ".join(cities) or "None")
            for label, cities in _groups.items()
        )
        + """

    The robust-retain group forms the strategic core of the future network.
    Conditional locations are supported by the base case but sensitive to
    individual external shocks. Scenario-dependent locations require
    operational validation before an irreversible closure decision.
    Robust closure candidates should still be phased out gradually rather
    than closed simultaneously.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, sensitivity_summary):
    mo.stop("scenario_results" not in globals())

    _base = sensitivity_summary[
        sensitivity_summary["scenario"] == "Base case"
    ].iloc[0]

    _scale = sensitivity_summary[
        sensitivity_summary["scenario"] == "Full scale effect (alpha=1.0)"
    ].iloc[0]

    _severe = sensitivity_summary[
        sensitivity_summary["scenario"] == "Severe decline (-50%)"
    ].iloc[0]

    _all_scenarios = sensitivity_summary[
        sensitivity_summary["scenario"] != "Base case"
    ]
    _min_reassigned = _all_scenarios["regions_reassigned"].min()
    _max_reassigned = _all_scenarios["regions_reassigned"].max()

    mo.md(f"""
    ## 6.9 Conclusion of the Sensitivity Analysis

    The sensitivity analysis supports three conclusions.

    First, the cost projections are robust. Total cost responds
    predictably and proportionally to the two external factors with a
    solid empirical basis: demand and transport cost. The internal
    calibration assumptions (alpha, the processing-cost anchor) move
    total cost by around {abs(_scale['cost_change']):.0%} and are
    therefore not material to the business case.

    Second, the strategic response to declining cash usage is downsizing,
    not closure. Across a 50 % demand decline, the facility count falls
    only from {_base['n_open']:.0f} to {_severe['n_open']:.0f}, while the
    size distribution collapses towards the smallest tiers. Geographic
    coverage retains its value even at halved volume; excess processing
    capacity does not. This is an operationally very different
    recommendation from network consolidation, and it is the most
    decision-relevant finding of the analysis.

    Third, the facility-level recommendation is considerably less robust
    than the cost recommendation. Between {_min_reassigned:.0%} and
    {_max_reassigned:.0%} of customer regions are reassigned depending on
    the scenario, and no single facility set survives all five scenarios
    unchanged. The final recommendation must therefore distinguish
    between the strategic core of the network and locations whose fate
    depends on assumptions the data cannot settle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Limitations and Conclusion

    ### Limitations

    1. Transportation costs are based on an aggregate annual shift
    approximation rather than daily vehicle-routing optimization.
    2. Individual customers are represented by 515 aggregated customer
    regions.
    3. The available warehouse data do not contain physical processing
    capacities. Fixed cost is used as a size proxy, and the volume
    bounds of the five capacity tiers are adopted from the course
    template rather than derived from operational data.
    4. The variable processing costs are calibrated rather than observed.
    Their relative scale follows the empirical fixed-cost ratios, but
    the dampening exponent (α=0.5\alpha = 0.5
    α=0.5) and the anchor level (15 %
    of the median transport cost per delivery) are assumptions. Both
    were examined in the sensitivity analysis; the anchor itself is
    calibrated once against base-case transport costs and deliberately
    held fixed across scenarios.
    5. The upgradeable model assumes that every existing location can
    technically be resized to any of the five center types, and that
    the same type-specific cost function applies at every location.
    Site-specific property, construction, security, labor, and
    regulatory differences are not available in the data.
    6. Demand, cost, and technology parameters are deterministic within
    each scenario. Uncertainty is represented through discrete scenario
    comparison rather than stochastic modeling.
    7. The base case is solved to proven optimality, but the sensitivity
    scenarios are solved with a 1 % relative MIP gap and a 120-second
    time limit. Residual gaps range from 2.1 % to 13.9 %, with the
    extended-operating-time scenario being the least precisely solved.
    Scenario results are therefore read as directional indicators, not
    as certified optima.
    8. The calibrated cost structure introduces a 26.7× fixed-cost spread
    between the smallest and largest tier, which makes the model
    substantially harder to solve than a formulation with flatter cost
    differences. This constrained the number of scenarios that could be
    evaluated within a reasonable computational budget; each scenario
    family is consequently represented by few, deliberately extreme
    parameter values rather than a fine grid.
    9. Closure costs, employee relocation, contractual obligations,
    transition periods, and implementation risks are not included.

    ### Conclusion

    The data-based baseline identifies the cost-minimizing network under
    the currently observable transportation and facility costs, but it
    implicitly assumes unlimited processing capacity. The upgradeable
    extension removes this flaw by making facility size a decision variable
    with data-calibrated, volume-dependent costs, and solves to a proven
    optimum of €202.0 million with 25 open locations.
    The sensitivity analysis shows that this recommendation is cost-robust
    but only partially structure-robust. Total cost responds predictably to
    the external factors with a solid empirical basis — demand and
    transport cost — while the internal calibration assumptions move it by
    around one percent. The facility-level composition, however, shifts
    considerably across scenarios, and no single set of locations survives
    all tested assumptions unchanged.
    The final recommendation is therefore based on agreement across
    scenario families rather than on one optimal solution. The robust core
    of the network should be retained, robust closure candidates should be
    phased out gradually, and the small group of scenario-dependent
    locations — most notably those whose base-case status is reversed in
    every tested scenario — require a facility-level capacity and
    implementation assessment before an irreversible decision is made.
    """)
    return


if __name__ == "__main__":
    app.run()
