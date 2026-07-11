import marimo

__generated_with = "0.23.13"
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

    Two optimization models are used.

    The first model is a data-based baseline. It uses the estimated
    transportation costs and the existing annual fixed costs from the
    warehouse data. Because no physical capacity information is available,
    this model does not impose warehouse-capacity constraints.

    The second model applies the stylized capacity and cost function from
    the course's advanced-analysis notebook. It allows the optimizer to
    select one of five possible size categories for every existing
    location. This model is used as a conditional scenario rather than as
    an estimate of CashLog's actual facility capacities.

    The comparison of both models shows how strongly the network
    recommendation depends on the additional capacity assumptions.
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
                ["warehouseID", "city", "fixedCosts"]
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
    ### 5.2 Baseline Capacity Diagnostic

    The assigned volumes are inspected to identify locations that would
    require operational validation. A high assigned volume is not treated
    as proof of infeasibility because the actual processing capacities are
    unknown.

    Locations with unusually high assigned demand should therefore be
    subject to a physical capacity audit before the baseline recommendation
    is implemented.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 Course-Template Capacity Extension

    The course's advanced-analysis model assumes five possible cash-center
    types. A location is not permanently tied to one type. Instead, the
    optimizer decides whether the location is closed or operated as a very
    small, small, medium, large, or huge center.

    The following parameter values are taken from the course template.
    They are treated as scenario assumptions and are not inferred from
    CashLog's warehouse data.
    """)
    return


@app.cell
def _(pd):
    COURSE_CENTER_TYPES = {
        "v": {
            "lower_bound": 0,
            "upper_bound": 19_348,
            "c_fix": 611_650,
            "c_var": 4.14,
        },
        "s": {
            "lower_bound": 19_349,
            "upper_bound": 45_415,
            "c_fix": 860_710,
            "c_var": 2.85,
        },
        "m": {
            "lower_bound": 45_416,
            "upper_bound": 107_327,
            "c_fix": 1_451_000,
            "c_var": 1.55,
        },
        "l": {
            "lower_bound": 107_328,
            "upper_bound": 199_999,
            "c_fix": 1_451_000,
            "c_var": 1.55,
        },
        "h": {
            "lower_bound": 200_000,
            "upper_bound": 99_999_999,
            "c_fix": 1_451_000,
            "c_var": 1.55,
        },
    }

    COURSE_TYPE_IDS = tuple(
        COURSE_CENTER_TYPES.keys()
    )

    course_type_table = (
        pd.DataFrame(COURSE_CENTER_TYPES)
        .T
    )

    course_type_table
    return COURSE_CENTER_TYPES, COURSE_TYPE_IDS


@app.cell
def _(
    BASE_SHIFT_COST,
    BASE_SHIFT_MINUTES,
    COURSE_CENTER_TYPES,
    COURSE_TYPE_IDS,
    DEMAND_BY_REGION,
    REGION_IDS,
    WAREHOUSE_IDS,
    build_scenario_links,
    pl,
):
    def solve_course_network(
        demand_factor=1.0,
        shift_cost=BASE_SHIFT_COST,
        shift_minutes=BASE_SHIFT_MINUTES,
        travel_factor=1.0,
        processing_cost_factor=1.0,
        center_fixed_cost_factor=1.0,
    ):
        """Solve the scenario-based network model with flexible center types."""

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
            for center_type in COURSE_TYPE_IDS
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
            .set_index(["warehouseID", "regionID"])
            ["transportationCosts"]
            .to_dict()
        )

        scenario_demand = {
            region_id: (
                DEMAND_BY_REGION[region_id]
                * demand_factor
            )
            for region_id in REGION_IDS
        }

        total_scenario_demand = sum(
            scenario_demand.values()
        )

        model = pl.LpProblem(
            "CashLog_Course_Extension",
            pl.LpMinimize,
        )

        x_course = pl.LpVariable.dicts(
            "x_course",
            link_keys,
            cat=pl.LpBinary,
        )

        y_course = pl.LpVariable.dicts(
            "y_course",
            type_keys,
            cat=pl.LpBinary,
        )

        z_course = pl.LpVariable.dicts(
            "z_course",
            type_keys,
            lowBound=0,
            cat=pl.LpContinuous,
        )

        transportation_expression = pl.lpSum(
            transport_cost_lookup[
                (warehouse_id, region_id)
            ]
            * x_course[
                (warehouse_id, region_id)
            ]
            for warehouse_id, region_id in link_keys
        )

        center_fixed_expression = pl.lpSum(
            COURSE_CENTER_TYPES[
                center_type
            ]["c_fix"]
            * center_fixed_cost_factor
            * y_course[
                (warehouse_id, center_type)
            ]
            for warehouse_id in WAREHOUSE_IDS
            for center_type in COURSE_TYPE_IDS
        )

        processing_expression = pl.lpSum(
            COURSE_CENTER_TYPES[
                center_type
            ]["c_var"]
            * processing_cost_factor
            * z_course[
                (warehouse_id, center_type)
            ]
            for warehouse_id in WAREHOUSE_IDS
            for center_type in COURSE_TYPE_IDS
        )

        model += (
            transportation_expression
            + center_fixed_expression
            + processing_expression
        )

        for warehouse_id, region_id in link_keys:
            model += (
                x_course[
                    (warehouse_id, region_id)
                ]
                <= pl.lpSum(
                    y_course[
                        (warehouse_id, center_type)
                    ]
                    for center_type in COURSE_TYPE_IDS
                )
            )

        for region_id in REGION_IDS:
            model += pl.lpSum(
                x_course[
                    (warehouse_id, region_id)
                ]
                for warehouse_id
                in links_by_region[region_id]
            ) == 1

        for warehouse_id in WAREHOUSE_IDS:

            model += pl.lpSum(
                z_course[
                    (warehouse_id, center_type)
                ]
                for center_type in COURSE_TYPE_IDS
            ) == pl.lpSum(
                scenario_demand[region_id]
                * x_course[
                    (warehouse_id, region_id)
                ]
                for region_id
                in links_by_center[warehouse_id]
            )

            model += pl.lpSum(
                y_course[
                    (warehouse_id, center_type)
                ]
                for center_type in COURSE_TYPE_IDS
            ) <= 1

            for center_type in COURSE_TYPE_IDS:
                model += (
                    z_course[
                        (warehouse_id, center_type)
                    ]
                    >= (
                        COURSE_CENTER_TYPES[
                            center_type
                        ]["lower_bound"]
                        * y_course[
                            (warehouse_id, center_type)
                        ]
                    )
                )

                effective_upper_bound = min(
                    COURSE_CENTER_TYPES[
                        center_type
                    ]["upper_bound"],
                    total_scenario_demand,
                )

                model += (
                    z_course[
                        (warehouse_id, center_type)
                    ]
                    <= (
                        effective_upper_bound
                        * y_course[
                            (warehouse_id, center_type)
                        ]
                    )
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
                "center_fixed_cost": None,
                "processing_cost": None,
                "n_open": None,
                "open_centers": [],
                "selected_type": {},
                "assigned_volume": {},
                "assignment": [],
                "link_data": link_data,
            }

        selected_type = {}

        for warehouse_id in WAREHOUSE_IDS:
            chosen_types = [
                center_type
                for center_type in COURSE_TYPE_IDS
                if (
                    y_course[
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
                    z_course[
                        (warehouse_id, center_type)
                    ].value()
                    or 0
                )
                for center_type in COURSE_TYPE_IDS
            )
            for warehouse_id in open_centers
        }

        assignment = [
            (warehouse_id, region_id)
            for warehouse_id, region_id in link_keys
            if (
                x_course[
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
        }

    return (solve_course_network,)


@app.cell
def _(mo, pd, solve_course_network):
    course_result = solve_course_network()

    mo.stop(
        course_result["status"] != "Optimal",
        mo.md(
            f"Course-extension status: "
            f"**{course_result['status']}**"
        ),
    )

    course_type_counts = (
        pd.Series(course_result["selected_type"])
        .value_counts()
        .sort_index()
    )
    return (course_result,)


@app.cell
def _(course_result, mo, pd):
    if course_result["status"] != "Optimal":
        mo.md(
            f"""
    ### Course-Template Extension Results

    The model could not be solved optimally.

    **Solver status:** `{course_result["status"]}`
    """
        )
    else:
        _type_counts_df = (
            pd.Series(
                course_result["selected_type"],
                name="center_type",
            )
            .value_counts()
            .rename_axis("Center type")
            .reset_index(name="Number of centers")
            .sort_values("Center type")
        )

        mo.vstack(
            [
                mo.md(
                    f"""
    ### Course-Template Extension Results

    | Metric | Result |
    |:---|---:|
    | Solver status | **{course_result["status"]}** |
    | Total annual cost | **€{course_result["total_cost"]:,.0f}** |
    | Transportation cost | **€{course_result["transportation_cost"]:,.0f}** |
    | Center fixed cost | **€{course_result["center_fixed_cost"]:,.0f}** |
    | Processing cost | **€{course_result["processing_cost"]:,.0f}** |
    | Open locations | **{course_result["n_open"]} of 42** |

    #### Selected size categories
    """
                ),
                mo.ui.table(_type_counts_df),
            ]
        )
    return


@app.cell
def _(REGION_IDS, course_result, mo):
    assert course_result["status"] == "Optimal"
    assert len(course_result["assignment"]) == len(REGION_IDS)
    assert len(course_result["selected_type"]) == course_result["n_open"]

    mo.md(
        "✅ The course-template model assigns every region and "
        "selects at most one size category per open location."
    )
    return


@app.cell
def _(
    CITY_BY_WAREHOUSE,
    COURSE_CENTER_TYPES,
    DEMAND_BY_REGION,
    course_result,
    np,
    pd,
):
    course_center_table = pd.DataFrame(
        [
            {
                "warehouseID": warehouse_id,
                "city": CITY_BY_WAREHOUSE[warehouse_id],
                "selected_type": center_type,
                "assigned_volume": (
                    course_result["assigned_volume"][warehouse_id]
                ),
                "lower_bound": (
                    COURSE_CENTER_TYPES[
                        center_type
                    ]["lower_bound"]
                ),
                "upper_bound": (
                    COURSE_CENTER_TYPES[
                        center_type
                    ]["upper_bound"]
                ),
            }
            for warehouse_id, center_type
            in course_result["selected_type"].items()
        ]
    )

    course_center_table["within_bounds"] = (
        (
            course_center_table["assigned_volume"]
            >= course_center_table["lower_bound"]
        )
        &
        (
            course_center_table["assigned_volume"]
            <= course_center_table["upper_bound"]
        )
    )

    course_center_table["capacity_utilization"] = np.where(
        course_center_table["selected_type"] == "h",
        np.nan,
        (
            course_center_table["assigned_volume"]
            / course_center_table["upper_bound"]
        ),
    )

    assert course_center_table["within_bounds"].all()

    assert np.isclose(
        course_center_table["assigned_volume"].sum(),
        sum(DEMAND_BY_REGION.values()),
    )

    course_center_table.sort_values(
        "assigned_volume",
        ascending=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Capacity Validation

    Every open location operates within the lower and upper volume bounds of
    its selected size category. The total volume processed by all open
    locations also equals the total demand of the customer regions.

    This confirms that the assumed capacity model is internally consistent.
    It does not validate the assumed capacity bounds against CashLog's
    actual physical facilities.

    No utilization ratio is reported for type `h` because its upper bound is
    a technical modeling value rather than an estimated physical capacity.
    """)
    return


@app.cell
def _(basic_result, course_result, pd):
    model_comparison = pd.DataFrame(
        [
            {
                "model": "Data-based baseline",
                "status": basic_result["status"],
                "total_cost": basic_result["total_cost"],
                "transportation_cost": (
                    basic_result["transportation_cost"]
                ),
                "facility_and_processing_cost": (
                    basic_result["fixed_cost"]
                ),
                "open_locations": basic_result["n_open"],
            },
            {
                "model": "Course-template extension",
                "status": course_result["status"],
                "total_cost": course_result["total_cost"],
                "transportation_cost": (
                    course_result["transportation_cost"]
                ),
                "facility_and_processing_cost": (
                    course_result["center_fixed_cost"]
                    + course_result["processing_cost"]
                ),
                "open_locations": course_result["n_open"],
            },
        ]
    )

    model_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Model Comparison

    The two objective values should not be interpreted as directly competing
    cost estimates because the models use different facility-cost
    structures.

    The relevant comparison is structural: it shows whether the selected
    locations remain similar when flexible size categories and the
    course-template capacity assumptions are introduced. Locations selected
    by both models provide stronger evidence of strategic importance than
    locations selected by only one formulation.
    """)
    return


@app.cell
def _(folium, mcolors, pd, plt, regions_flat, warehouses_flat):
    def create_network_map(result):
        open_centers = result["open_centers"]

        assignment_map_df = pd.DataFrame(
            result["assignment"],
            columns=["warehouseID", "regionID"],
        ).merge(
            regions_flat[
                ["regionID", "lat", "lon", "city"]
            ],
            on="regionID",
        )

        map_center_lat = warehouses_flat["lat"].mean()
        map_center_lon = warehouses_flat["lon"].mean()

        cmap = plt.get_cmap(
            "tab20",
            max(len(open_centers), 1),
        )

        color_by_center = {
            warehouse_id: mcolors.to_hex(cmap(position))
            for position, warehouse_id
            in enumerate(open_centers)
        }

        network_map = folium.Map(
            location=[map_center_lat, map_center_lon],
            zoom_start=6,
            tiles="cartodbpositron",
        )

        for _row in assignment_map_df.itertuples():
            folium.CircleMarker(
                location=[_row.lat, _row.lon],
                radius=2,
                color=color_by_center.get(
                    _row.warehouseID,
                    "#999999",
                ),
                fill=True,
                fill_opacity=0.6,
            ).add_to(network_map)

        selected_type = result.get(
            "selected_type",
            {},
        )

        for _row in warehouses_flat.itertuples():
            is_open = (
                _row.warehouseID in open_centers
            )

            type_label = selected_type.get(
                _row.warehouseID,
                "not specified",
            )

            folium.CircleMarker(
                location=[_row.lat, _row.lon],
                radius=8 if is_open else 4,
                color=(
                    color_by_center.get(
                        _row.warehouseID,
                        "#999999",
                    )
                    if is_open
                    else "#999999"
                ),
                fill=True,
                fill_opacity=1 if is_open else 0.2,
                popup=(
                    f"{_row.city}<br>"
                    f"Status: "
                    f"{'open' if is_open else 'closed'}<br>"
                    f"Type: {type_label}"
                ),
            ).add_to(network_map)

        return network_map

    return (create_network_map,)


@app.cell
def _(basic_result, course_result, create_network_map, mo):
    basic_network_map = create_network_map(
        basic_result
    )

    course_network_map = create_network_map(
        course_result
    )

    mo.ui.tabs(
        {
            "Data-based baseline": basic_network_map,
            "Course-template extension": course_network_map,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sensitivity Analysis

    The sensitivity analysis examines four sources of uncertainty:

    1. declining demand for cash services;
    2. rising shift costs, representing combined wage, fuel, maintenance,
       and vehicle-cost pressure;
    3. rising processing labor costs inside cash centers;
    4. technological changes affecting transport cost or travel time.

    The parameter values are stress-test assumptions rather than forecasts.
    For each scenario, changes in total cost, network size, and selected
    locations are recorded.
    """)
    return


@app.cell
def _(mo):
    run_sensitivity = mo.ui.run_button(
        label="Run sensitivity analysis"
    )

    run_sensitivity
    return (run_sensitivity,)


@app.cell
def _(
    BASE_SHIFT_COST,
    course_result,
    mo,
    pd,
    run_sensitivity,
    solve_course_network,
):
    mo.stop(not run_sensitivity.value)

    import time as _time

    demand_course_results = {
        1.0: course_result,
    }

    shift_course_results = {
        1.0: course_result,
    }

    processing_course_results = {
        1.0: course_result,
    }

    technology_course_results = {
        "base": course_result,
    }

    _runtime_rows = []

    for _factor in [0.9, 0.7, 0.5]:
        _start = _time.perf_counter()

        _result = solve_course_network(
            demand_factor=_factor
        )

        _runtime = (
            _time.perf_counter() - _start
        )

        demand_course_results[_factor] = _result

        _runtime_rows.append(
            {
                "family": "Demand",
                "scenario": f"{_factor:.0%}",
                "status": _result["status"],
                "runtime_seconds": round(
                    _runtime,
                    1,
                ),
            }
        )

        print(
            f"Demand {_factor:.0%}: "
            f"{_result['status']} "
            f"({_runtime:.1f} seconds)",
            flush=True,
        )

    for _factor in [1.2, 1.5]:
        _start = _time.perf_counter()

        _result = solve_course_network(
            shift_cost=BASE_SHIFT_COST * _factor
        )

        _runtime = (
            _time.perf_counter() - _start
        )

        shift_course_results[_factor] = _result

        _runtime_rows.append(
            {
                "family": "Shift cost",
                "scenario": f"x{_factor:.1f}",
                "status": _result["status"],
                "runtime_seconds": round(
                    _runtime,
                    1,
                ),
            }
        )

        print(
            f"Shift cost x{_factor:.1f}: "
            f"{_result['status']} "
            f"({_runtime:.1f} seconds)",
            flush=True,
        )

    for _factor in [1.2, 1.5]:
        _start = _time.perf_counter()

        _result = solve_course_network(
            processing_cost_factor=_factor
        )

        _runtime = (
            _time.perf_counter() - _start
        )

        processing_course_results[_factor] = (
            _result
        )

        _runtime_rows.append(
            {
                "family": "Processing cost",
                "scenario": f"x{_factor:.1f}",
                "status": _result["status"],
                "runtime_seconds": round(
                    _runtime,
                    1,
                ),
            }
        )

        print(
            f"Processing cost x{_factor:.1f}: "
            f"{_result['status']} "
            f"({_runtime:.1f} seconds)",
            flush=True,
        )

    _TECHNOLOGY_SCENARIOS = {
        "lower_transport_cost": {
            "shift_cost": BASE_SHIFT_COST * 0.4,
            "travel_factor": 1.0,
        },
        "faster_travel": {
            "shift_cost": BASE_SHIFT_COST,
            "travel_factor": 0.8,
        },
        "combined_technology": {
            "shift_cost": BASE_SHIFT_COST * 0.4,
            "travel_factor": 0.8,
        },
    }

    for _scenario_name, _parameters in (
        _TECHNOLOGY_SCENARIOS.items()
    ):
        _start = _time.perf_counter()

        _result = solve_course_network(
            **_parameters
        )

        _runtime = (
            _time.perf_counter() - _start
        )

        technology_course_results[
            _scenario_name
        ] = _result

        _runtime_rows.append(
            {
                "family": "Technology",
                "scenario": _scenario_name,
                "status": _result["status"],
                "runtime_seconds": round(
                    _runtime,
                    1,
                ),
            }
        )

        print(
            f"{_scenario_name}: "
            f"{_result['status']} "
            f"({_runtime:.1f} seconds)",
            flush=True,
        )


    sensitivity_runtime = pd.DataFrame(
        _runtime_rows
    )

    sensitivity_runtime
    return (
        demand_course_results,
        processing_course_results,
        shift_course_results,
        technology_course_results,
    )


@app.cell
def _(
    CITY_BY_WAREHOUSE,
    demand_course_results,
    pd,
    processing_course_results,
    shift_course_results,
    technology_course_results,
):
    def summarize_scenario(
        scenario_family,
        scenario_name,
        result,
        reference_result,
    ):
        """Create one summary row for an optimization scenario."""

        if result["status"] != "Optimal":
            return {
                "family": scenario_family,
                "scenario": scenario_name,
                "status": result["status"],
                "total_cost": None,
                "cost_change_vs_base": None,
                "open_locations": None,
                "opened_vs_base": "–",
                "closed_vs_base": "–",
            }

        result_open = set(result["open_centers"])
        reference_open = set(reference_result["open_centers"])

        newly_opened = sorted(
            CITY_BY_WAREHOUSE[warehouse_id]
            for warehouse_id in result_open - reference_open
        )

        newly_closed = sorted(
            CITY_BY_WAREHOUSE[warehouse_id]
            for warehouse_id in reference_open - result_open
        )

        cost_change = (
            result["total_cost"]
            / reference_result["total_cost"]
            - 1
        )

        return {
            "family": scenario_family,
            "scenario": scenario_name,
            "status": result["status"],
            "total_cost": result["total_cost"],
            "cost_change_vs_base": cost_change,
            "open_locations": result["n_open"],
            "opened_vs_base": (
                ", ".join(newly_opened)
                if newly_opened
                else "–"
            ),
            "closed_vs_base": (
                ", ".join(newly_closed)
                if newly_closed
                else "–"
            ),
        }


    scenario_summary_rows = []

    for factor, result in demand_course_results.items():
        scenario_summary_rows.append(
            summarize_scenario(
                scenario_family="Demand",
                scenario_name=f"{factor:.0%}",
                result=result,
                reference_result=demand_course_results[1.0],
            )
        )

    for factor, result in shift_course_results.items():
        scenario_summary_rows.append(
            summarize_scenario(
                scenario_family="Shift cost",
                scenario_name=f"x{factor:.1f}",
                result=result,
                reference_result=shift_course_results[1.0],
            )
        )

    for factor, result in processing_course_results.items():
        scenario_summary_rows.append(
            summarize_scenario(
                scenario_family="Processing cost",
                scenario_name=f"x{factor:.1f}",
                result=result,
                reference_result=processing_course_results[1.0],
            )
        )

    for scenario_name, result in technology_course_results.items():
        scenario_summary_rows.append(
            summarize_scenario(
                scenario_family="Technology",
                scenario_name=scenario_name,
                result=result,
                reference_result=technology_course_results["base"],
            )
        )

    sensitivity_summary = pd.DataFrame(
        scenario_summary_rows
    )

    sensitivity_summary
    return (sensitivity_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Declining Cash Demand

    This analysis identifies the demand level at which the course-template
    network begins to change. The relevant questions are whether lower demand
    reduces only total cost or also leads to facility closures, changes in
    the selected size categories, or geographical consolidation.

    The tested demand factors are long-term stress-test assumptions rather
    than forecasts for a specific year.
    """)
    return


@app.cell
def _(sensitivity_summary):
    sensitivity_summary[
        sensitivity_summary["family"] == "Demand"
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Interpretation

    Reducing demand to 90%, 70%, and 50% of the current level lowers total
    annual cost by approximately 7.9%, 24.0%, and 41.0%, respectively.

    The cost reduction is smaller than the demand reduction because facility
    fixed costs remain even when fewer customer visits are required. Changes
    in the number or identity of open locations indicate the demand levels
    at which consolidation becomes economically preferable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Rising Wages and Fuel Costs

    The €480 shift cost combines wages, fuel, depreciation, and maintenance.
    Because no component-level breakdown is available, the analysis varies
    the complete shift cost rather than attempting to estimate separate wage
    and fuel effects.

    The results reveal whether higher transport costs favor a denser network
    with shorter service distances or whether consolidation remains more
    economical.
    """)
    return


@app.cell
def _(sensitivity_summary):
    sensitivity_summary[
        sensitivity_summary["family"] == "Shift cost"
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Interpretation

    A 20% increase in shift cost raises total annual cost by approximately
    13.0%, while a 50% increase raises it by approximately 32.4%.

    The total cost response is less than proportional because only the
    transport component is changed; center fixed costs and processing costs
    remain unchanged. Any additional locations selected in these scenarios
    would indicate that higher transport costs strengthen the economic value
    of shorter service distances.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Rising Processing Labor Costs

    This scenario applies only to the course-template extension because the
    data-based baseline does not contain a separate processing-cost
    component.

    The analysis shows whether higher in-center labor costs change the
    preferred facility-size mix or strengthen the incentive to concentrate
    volume in larger centers.
    """)
    return


@app.cell
def _(sensitivity_summary):
    sensitivity_summary[
        sensitivity_summary["family"] == "Processing cost"
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Interpretation

    Increasing the assumed processing costs by 20% and 50% raises total
    annual cost by only approximately 1.3% and 3.1%.

    Processing cost therefore represents a comparatively small share of the
    modeled total cost. The network is more sensitive to transportation-cost
    changes than to proportional changes in the assumed variable
    processing-cost rates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 New Technology

    The technology scenarios separate two effects:

    - lower transport cost per shift;
    - shorter travel times.

    The combined scenario applies both effects simultaneously. Shift length
    is kept at 450 minutes so that changes in cost, speed, and operating time
    are not mixed into one assumption.
    """)
    return


@app.cell
def _(sensitivity_summary):
    sensitivity_summary[
        sensitivity_summary["family"] == "Technology"
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Interpretation

    Reducing the transport cost per shift lowers total annual cost by
    approximately 40.4%. Reducing travel time by 20% lowers total cost by
    approximately 6.6%, while the combined technology scenario produces a
    reduction of approximately 47.4%.

    Within the tested assumptions, the cost effect of automation is therefore
    driven mainly by the lower shift cost. Faster travel provides an
    additional benefit, but its isolated impact is considerably smaller.
    These values are scenario results and should not be interpreted as a
    technology forecast.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Robust Recommendation

    A location is considered robust when it is supported by both model
    formulations and remains selected across several independent scenario
    families. Scenarios within the same family are aggregated before the
    final classification so that a family with more tested parameter values
    does not receive disproportionate weight.
    """)
    return


@app.cell
def _(
    CITY_BY_WAREHOUSE,
    WAREHOUSE_IDS,
    basic_result,
    course_result,
    demand_course_results,
    np,
    pd,
    processing_course_results,
    shift_course_results,
    technology_course_results,
):
    _demand_family_results = [
        result
        for factor, result in demand_course_results.items()
        if factor != 1.0
        and result["status"] == "Optimal"
    ]

    _shift_cost_family_results = [
        result
        for factor, result in shift_course_results.items()
        if factor != 1.0
        and result["status"] == "Optimal"
    ]

    _processing_cost_family_results = [
        result
        for factor, result in processing_course_results.items()
        if factor != 1.0
        and result["status"] == "Optimal"
    ]

    _technology_family_results = [
        result
        for scenario_name, result
        in technology_course_results.items()
        if scenario_name != "base"
        and result["status"] == "Optimal"
    ]


    def _calculate_open_rate(
        warehouse_id,
        family_results,
    ):
        """Share of scenarios in which a warehouse remains open."""

        if not family_results:
            return np.nan

        return np.mean(
            [
                warehouse_id in result["open_centers"]
                for result in family_results
            ]
        )


    robustness_rows = []

    for warehouse_id in WAREHOUSE_IDS:
        basic_base_open = (
            basic_result["status"] == "Optimal"
            and warehouse_id
            in basic_result["open_centers"]
        )

        course_base_open = (
            course_result["status"] == "Optimal"
            and warehouse_id
            in course_result["open_centers"]
        )

        demand_open_rate = _calculate_open_rate(
            warehouse_id,
            _demand_family_results,
        )

        shift_cost_open_rate = _calculate_open_rate(
            warehouse_id,
            _shift_cost_family_results,
        )

        processing_cost_open_rate = _calculate_open_rate(
            warehouse_id,
            _processing_cost_family_results,
        )

        technology_open_rate = _calculate_open_rate(
            warehouse_id,
            _technology_family_results,
        )

        family_rates = [
            demand_open_rate,
            shift_cost_open_rate,
            processing_cost_open_rate,
            technology_open_rate,
        ]

        family_support = sum(
            rate >= 0.5
            for rate in family_rates
            if pd.notna(rate)
        )

        robustness_rows.append(
            {
                "warehouseID": warehouse_id,
                "city": CITY_BY_WAREHOUSE[
                    warehouse_id
                ],
                "basic_base_open": basic_base_open,
                "course_base_open": course_base_open,
                "demand_open_rate": demand_open_rate,
                "shift_cost_open_rate": (
                    shift_cost_open_rate
                ),
                "processing_cost_open_rate": (
                    processing_cost_open_rate
                ),
                "technology_open_rate": (
                    technology_open_rate
                ),
                "family_support": family_support,
            }
        )


    robustness_df = pd.DataFrame(
        robustness_rows
    )

    robustness_df
    return (robustness_df,)


@app.cell
def _(pd, robustness_df):
    def classify_location(_row):
        open_in_both = (
            _row["basic_base_open"]
            and _row["course_base_open"]
        )

        open_in_either = (
            _row["basic_base_open"]
            or _row["course_base_open"]
        )

        scenario_rates = [
        _row["demand_open_rate"],
        _row["shift_cost_open_rate"],
        _row["processing_cost_open_rate"],
        _row["technology_open_rate"],
        ]

        never_open_in_scenarios = all(
            pd.isna(rate) or rate == 0
            for rate in scenario_rates
        )

        if (
            open_in_both
            and _row["family_support"] >= 3
        ):
            return "Retain – robust"

        if (
            open_in_either
            and _row["family_support"] >= 1
        ):
            return "Retain – conditional"

        if (
            not open_in_either
            and never_open_in_scenarios
        ):
            return "Closure candidate – robust"

        return "Scenario-dependent – verify before closure"


    robustness_df["recommendation"] = (
        robustness_df.apply(
            classify_location,
            axis=1,
        )
    )

    robustness_df.sort_values(
        [
            "recommendation",
            "family_support",
            "city",
        ],
        ascending=[True, False, True],
    )
    return


@app.cell
def _(mo, robustness_df):
    recommendation_counts = (
        robustness_df["recommendation"]
        .value_counts()
    )

    robust_retain_cities = sorted(
        robustness_df.loc[
            robustness_df["recommendation"]
            == "Retain – robust",
            "city",
        ].tolist()
    )

    conditional_cities = sorted(
        robustness_df.loc[
            robustness_df["recommendation"]
            == "Retain – conditional",
            "city",
        ].tolist()
    )

    scenario_dependent_cities = sorted(
        robustness_df.loc[
            robustness_df["recommendation"]
            == "Scenario-dependent – verify before closure",
            "city",
        ].tolist()
    )

    closure_candidate_cities = sorted(
        robustness_df.loc[
            robustness_df["recommendation"]
            == "Closure candidate – robust",
            "city",
        ].tolist()
    )

    mo.md(f"""
    ### Recommended Decision Structure

    **Robust retain locations:**  
    {", ".join(robust_retain_cities) or "None"}

    **Conditional retain locations:**  
    {", ".join(conditional_cities) or "None"}

    **Scenario-dependent locations:**  
    {", ".join(scenario_dependent_cities) or "None"}

    **Robust closure candidates:**  
    {", ".join(closure_candidate_cities) or "None"}

    The robust-retain group forms the strategic core of the future network.
    Conditional locations are generally supported but remain sensitive to
    individual assumptions. Scenario-dependent locations require additional
    operational validation before an irreversible closure decision is made.
    Robust closure candidates should still be reviewed through a phased
    implementation process rather than closed simultaneously.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Limitations and Conclusion

    ### Limitations

    1. Transportation costs are based on an aggregate annual shift
       approximation rather than daily vehicle-routing optimization.
    2. Individual customers are represented by 515 aggregated customer
       regions.
    3. The available warehouse data do not contain physical processing
       capacities.
    4. The five size categories and their costs originate from the course
       template and are scenario assumptions.
    5. The course-template extension assumes that every existing location
       can technically be resized to any of the five center types.
    6. It also assumes that the same type-specific cost function applies at
       every location. Site-specific property, construction, security, labor,
       and regulatory differences are not available.
    7. Demand, cost, and technology parameters are deterministic within each
       scenario.
    8. Closure costs, employee relocation, contractual obligations,
       transition periods, and implementation risks are not included.

    ### Conclusion

    The data-based model identifies the cost-minimizing network under the
    currently observable transportation and facility costs. The
    course-template extension tests whether the result remains stable when
    facility size and volume-dependent processing costs are introduced.

    The final recommendation is therefore based on agreement across model
    formulations and scenario families rather than on one optimal solution.
    Robust locations should form the core network, while conditional and
    scenario-dependent locations require a facility-level capacity and
    implementation assessment before an irreversible decision is made.
    """)
    return


if __name__ == "__main__":
    app.run()
