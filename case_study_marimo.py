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
    # 1. Problem Definition and Strategic Context

    ## 1.1 Background and Market Dynamics
    CashLog operates as the market leader in cash logistics within the Spanish financial sector. The core business model comprises three primary operational pillars:
    * **Cash Collection:** collecting cash from customers
    * **Processing and Vaulting:** counting, verification, and high-security storage within dedicated Cash Centers.
    * **Distribution:** redistribution of physical currency to client networks.

    Historically, CashLog's network architecture followed a decentralized structure, maintaining one Cash Center in every Spanish provincial capital. This geographic distribution was strictly coupled with the local branch network of the Spanish Central Bank. Due to institutional restructuring, this regulatory dependency has ceased to exist, forcing a strategic reallocation of assets. Currently, the efficiency of the network is constrained by two countervailing macroeconomic pressures:

    1. **Secular Decline in Cash Volume:** The accelerating substitution of physical currency by digital and electronic payment systems has led to a structural contraction in total processed volume per facility.
    2. **High Asset Specificity and Fixed Costs:** Security, surveillance and insurance make every
      cash center highly expensive just to keep open.

    ## 1.2 Current Network Topology
    The current logistics network consists of a set of existing facilities and demand aggregates:
    * **Facilities:** $M = 42$ active Cash Centers.
    * **Demand Points:** $N = 42,000$ client locations, structurally aggregated into $K = 515$ distinct customer regions.

    ## 1.3 Optimization Objective and Operational Constraints
    The strategic objective is to formulate a network optimization problem aimed at identifying the optimal subset of Cash Centers to retain. The mathematical objective is to minimize total annualized expenditures while ensuring demand fulfillment.

    The optimization model is bound by two deterministic business constraints:

    1. **Asset Exclusivity:** The solution space for potential facility locations is strictly bounded by the current network. The opening of new locations is excluded.
    2. **Guaranteed Service Levels:** Demand fulfillment is non-negotiable. A facility closure is only permissible if the associated regional demand can be completely absorbed by the remaining active network without violating capacity or distance thresholds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Formalizing the Problem

    ## 2.1 The Underlying Economic Trade-off

    Centers cannot simply be closed on the basis of cost alone, since an underlying trade-off governs the decision problem. Opening more centers reduces driving time and distance, and thus transport cost, but increases the fixed cost of operating additional Cash Centers. Conversely, opening fewer centers lowers fixed cost but increases transport cost through longer average distances. The cheapest, most stable, and most efficient network therefore lies somewhere between these two extremes, rather than at either boundary.

    ## 2.2 Network Interdependencies and Systemic Optimization

    Scoring each center individually and closing the worst-performing one is not a valid approach, due to network effects. Whether a given center is worth keeping depends on which other centers remain open. If one center is closed, its regions must be reassigned to other centers, which in turn changes their load and cost parameters. Because regions are interdependent, all centers must be decided upon simultaneously, which requires an optimization model rather than a sequence of local decisions.

    ## 2.3 Baseline Formulation: The Warehouse-Location Model

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
    shift_min = 450
    #parameters
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
    # 3. Data

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

    def quick_look(df: pd.DataFrame, name: str, exclude: list[str] | None=None):
        missing = df.isna().sum()
        missing = missing[missing > 0]
        meta_md = f'**`{name}`** — {df.shape[0]:,} rows × {df.shape[1]} columns\n\n'
        meta_md = meta_md + '| column | dtype | missing |\n|---|---|---|\n'
        meta_md = meta_md + '\n'.join((f'| `{c}` | {df[c].dtype} | {missing.get(c, 0)} |' for c in df.columns))
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        id_like = [c for c in numeric_cols if c.lower().endswith(('id', '_id'))]
        drop = set(id_like) | set(exclude or [])
        numeric_cols = [c for c in numeric_cols if c not in drop]
        display(Markdown(meta_md))
        display(df.head(3))
        if numeric_cols:
            display(df[numeric_cols].describe().round(2))
        else:
            display(Markdown('_no non-ID numeric columns to describe_'))

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

    The warehouses dataset contains the city, location, and fixed cost of opening and operating each of the 42 candidate warehouses. Fixed costs range from €1,344,000.00 to €35,904,000.00, with the most expensive warehouses located in Madrid, Barcelona, and Alicante. Likely reflecting the size and property price level of these regions.

    The regions dataset contains information for each of the 515 clustered demand regions, and therefore already addresses the second major limitation of the base model discussed above: customers were pre-aggregated into regions by zip code. For each region, the dataset provides a unique regionID, the city name, its location, yearly demand, and average minutesPerStop. Yearly demand ranges from 16.00 to 247,638.00, and minutesPerStop ranges from 6.20 to 32.12 minutes. The minutesPerStop column already combines the driving time between customers within a region with the time spent at each customer, and therefore represents a single, pre-summarized productivity figure.

    The third dataset, shifts.csv, contains the travel time (in minutes) and transportation cost for every (center, region) pair, totaling 21,360 rows. The transportationCosts column in this dataset is a placeholder equal to the travel time; sensible, logic-based proxies for transportation cost are derived from travel time in the following step.

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

    #### Interpretation

    The farther away a region (the larger the travel time) the smaller the denominator becomes, which increases $c_{ij}$. The more spread out the customers within a region (the larger minutesPerStop) the more expensive it is to serve them. Finally, the larger the demand, the larger the numerator, and thus the higher the cost.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1 Unreachable Links

    Routes for which $2\cdot \text{travelTime}_{ij} \ge 450$ are infeasible. Travel time alone would already exceed the usable time per shift, and the denominator in our cost formula becomes zero or negative.

    Two options exist for handling such links: dropping them entirely (never creating an $x_{ij}$ variable for them), or retaining them with a Big-M cost large enough that the solver would never select them.

    Dropping is the simpler and more robust option, where it is safe to do so. It guarantees that the connection is never chosen, since a variable that does not exist cannot be selected. A Big-M penalty only achieves the same outcome if $M$ happens to be larger than every alternative the model could otherwise choose. A judgment call that would need to be justified separately, and one that cannot yet be made since the model has not been solved and its cost scale is not yet known.

    Our approach is therefore to drop unreachable links, unless doing so would leave some region with no reachable center at all. In that case, dropping would render the model infeasible, and we would fall back to a Big-M penalty instead. This condition is checked in the following step.
    """)
    return


@app.cell
def _(regions, shift_min, shifts, shifts_ref):
    # a link is reachable if 2*travelTime < shift_min (i.e. denominator > 0)
    #transform the shifts and regions dataframes to usable format for easier analysis
    #hier gleich alle dfs auf long machen (nacher)
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

    print(f"Regions with zero reachable centers: {n_unreachable_regions}")
    reach_counts.describe()
    return is_reachable, regions_flat, shifts_long, shifts_ref_long


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decision: Drop Unreachable Links

    The check confirms that every region retains at least one reachable center (minimum 1, median 5, maximum 12 reachable centers per region, across all 515 regions). Dropping unreachable links therefore does not risk infeasibility, and we adopt the simpler, more robust option. Unreachable links are dropped entirely rather than handled via a Big-M penalty.
    """)
    return


@app.cell
def _(is_reachable, shifts_long):
    # Nur die als erreichbar geprüften Links behalten wir (siehe Entscheidung oben).
    shifts_reachable = shifts_long[is_reachable].copy()

    n_total = len(shifts_long)
    n_kept = len(shifts_reachable)
    n_dropped = n_total - n_kept

    print(f"Total possible center-region links: {n_total:,}")
    print(f"Reachable (kept):                   {n_kept:,}  ({n_kept/n_total:.1%})")
    print(f"Unreachable (dropped):               {n_dropped:,}  ({n_dropped/n_total:.1%})")

    shifts_reachable.shape
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
    return cost_base, demand_by_region, stop_time_by_region


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
def _(cost_base, shifts_ref_long):
    benchmark = shifts_ref_long.copy()

    comparison = cost_base.merge(
        benchmark[["warehouseID", "regionID", "transportationCosts"]],
        on=["warehouseID", "regionID"],
        how="left",
        suffixes=("_own", "_benchmark"),
    )

    # Sanity check: hat jede unserer reachable rows eine Entsprechung in der Referenz?
    n_missing = comparison["transportationCosts_benchmark"].isna().sum()
    print(f"Reachable rows without a benchmark match: {n_missing} of {len(comparison):,}")

    comparison["rel_diff"] = (
        (comparison["transportationCosts_own"] - comparison["transportationCosts_benchmark"]).abs()
        / comparison["transportationCosts_benchmark"]
    )
    comparison["rel_diff"].describe()
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
    ## 5. Extending the Model to Fix the Remaining Problem

    We identified three reasons why the baseline warehouse-location model does not fit the CashLog problem. The first two, an undefined $c_{ij}$ and too large a number of individual customers, have already been resolved. The cost formula was derived and validated in the previous section, and the clustering of customers into 515 regions was already provided in the raw dataset. The third problem remains. The baseline model assumes each center has a fixed capacity and a constant fixed cost, whereas at CashLog capacity is a strategic, investable choice, and cost follows economies of scale.

    The model is now extended to address this directly, by introducing discrete center-size tiers, each with its own capacity range and cost structure.

    ### 5.1 The Extended Cost Function

    Our extended objective combines three cost components:

    $$\min\; \underbrace{\sum_i\sum_j x_{ij}\,c_{ij}}_{\text{transport (already derived)}} \;+\; \underbrace{\sum_i f_i\,y_i}_{\text{real, center-specific fixed cost}} \;+\; \underbrace{\sum_i\sum_t c_t^{var}\,z_{it}}_{\text{size-dependent processing cost (new)}}$$

    The first term is already fully derived and validated. The second term uses each center's actual fixed cost directly from the warehouses dataset; since this information is already available for every individual center, replacing it with a coarser tier average would discard real data unnecessarily. The third term is new. It captures the volume-dependent variable processing cost incurred inside the center, which is not covered by the transport formula and, due to economies of scale, is not constant per delivery.

    To make the third term concrete, two components are still required:

    1. a classification of each center into a size tier, and
    2. a per-tier processing cost $c_t^{var}$ that decreases with tier size.

    Both are derived from the data below, before the full set of decision variables and constraints is specified.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Center Size Tiers

    We begin by classifying the cash centers into size tiers. The dataset does not contain a direct measure of each center's physical capacity, so the fixed-cost entry for each warehouse is used as a proxy for physical size. The fixed cost of operating a warehouse is typically driven by factors such as square footage, electricity costs, salary costs, and regional price levels. Largely indicators of physical size.

    The tier boundaries are derived from natural gaps in the fixed-cost distribution. This remains an assumption. Fixed cost correlates with size but is not equivalent to it, so the resulting tiers should be understood as a reasonable approximation rather than a precise capacity measurement.
    """)
    return


@app.cell
def _(plt, warehouses_flat):
    sorted_costs = warehouses_flat['fixedCosts'].sort_values().reset_index(drop=True)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(range(len(sorted_costs)), sorted_costs, marker='o')
    _ax.set_xlabel('Warehouses, sorted by fixed cost')
    _ax.set_ylabel('Fixed cost (€)')
    _ax.set_title('Looking for natural breaks in the fixed-cost distribution')
    plt.show()
    return (sorted_costs,)


@app.cell
def _(pd, sorted_costs):
    #biggest gaps in the fixed-cost distribution
    gaps = sorted_costs.diff()
    gap_summary = pd.DataFrame({
        "index": range(len(sorted_costs)),
        "fixedCosts": sorted_costs,
        "gap_to_previous": gaps,
    }).sort_values("gap_to_previous", ascending=False)

    gap_summary.head(6)
    return


@app.cell
def _(sorted_costs):
    sorted_costs.iloc[0]        # der Wert an Position 0 = der gemeinsame Tier-1-Wert
    sorted_costs.iloc[:19].nunique()   # sollte 1 sein, bestätigt "alle 19 identisch"

    print(sorted_costs.iloc[0])
    print(sorted_costs.iloc[:19].nunique())
    return


@app.cell
def _(warehouses_flat):
    #check if we considered all warehouses
    n_unique = warehouses_flat["fixedCosts"].nunique()
    counts = warehouses_flat["fixedCosts"].value_counts().sort_index()
    print(f"Distinct values: {n_unique}")
    print(counts)
    print(f"Sum of counts: {counts.sum()} (should be 42)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gap analysis reveals that all 42 warehouses fall into exactly five distinct fixed-cost levels.

    | Tier | Fixed Cost (€) | Number of Warehouses |
    | ---- | -------------: | --------------------: |
    | 1    |      1,344,000 |                    19 |
    | 2    |      2,580,000 |                    15 |
    | 3    |      4,572,000 |                     6 |
    | 4    |     15,012,000 |                     1 |
    | 5    |     35,904,000 |                     1 |

    These results provide strong, data-driven support for using five size tiers. While the two largest tiers each contain only a single warehouse, they are retained as separate categories rather than being merged, since their fixed costs (€15.0M and €35.9M) represent clearly distinct capacity levels rather than minor variation around a common value.

    Each warehouse is mapped to its size tier directly via its fixed-cost value. Since there are exactly five distinct values, this mapping is exact, rather than an approximation or a clustering result with ambiguous boundary cases.
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 Variable Processing Costs

    Having classified each cash center into a size tier, we now derive the variable processing cost per delivery ($c_t^{var}$) for each tier. Rather than estimating five independent values, a single anchor value is scaled using the fixed-cost ratios between tiers established above. Since fixed cost serves as our size proxy, a larger tier's fixed-cost ratio also indicates how much larger it likely is, and economies of scale imply that variable cost per delivery should fall accordingly.

    Note the unit: $c_t^{var}$ denotes cost *per individual customer visit*, not per shift or per tour — unlike $c_{ij}$, which already aggregates many visits into a single shift cost.
    """)
    return


@app.cell
def _():
    fixed_by_tier = {"v": 1_344_000, "s": 2_580_000, "m": 4_572_000, "l": 15_012_000, "h": 35_904_000}
    scale_factor = {t: f / fixed_by_tier["v"] for t, f in fixed_by_tier.items()}
    scale_factor
    return (scale_factor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The smallest warehouse tier (v) is used as the reference tier, with a scaling factor of 1.0. The scaling factors for the remaining tiers are obtained by dividing each tier's fixed cost by that of the smallest tier, yielding a relative measure of size across all tiers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dampening the Scaling Effect

    Having derived the fixed-cost ratios between tiers, we now determine how strongly variable cost per delivery should decrease as warehouse size increases. A one-to-one translation of the fixed-cost ratio into variable cost savings would be unrealistically strong. While fixed costs primarily reflect investments in capacity (e.g., warehouse space, equipment, and security infrastructure), operational efficiencies per delivery are subject to physical and organizational limits and therefore cannot improve proportionally with warehouse size. To account for this, the scaling effect is dampened rather than applying the fixed-cost ratios directly, representing economies of scale in a more realistic manner.

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
def _(scale_factor):
    #dampened scaling factors berechnen
    damp_scale_factor = {t: f ** (-0.5) for t, f in scale_factor.items()}
    damp_scale_factor
    return


@app.cell
def _(cost_base, regions_flat):
    #per delivery transport cost = yearly total transport cost per link / yearly demand of the region
    #cost_base = df mit den transportkosten pro link
    merged = cost_base.merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
    merged["per_delivery_transport_cost"] = merged["transportationCosts"] / merged["yearlyDemand"]

    merged["per_delivery_transport_cost"].describe()
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
def _(merged, scale_factor):
    alpha = 0.5       # dampening exponent

    median_transport_per_delivery = merged["per_delivery_transport_cost"].median()
    processing_share = 0.15  # Anteil der Transport-pro-Lieferung-Kosten

    c_var_anchor = processing_share * median_transport_per_delivery
    print(f"c_v^var anchor: {c_var_anchor:.2f} € per delivery")

    c_var_by_tier = {t: c_var_anchor / (scale_factor[t] ** alpha) for t in scale_factor}
    c_var_by_tier
    return (
        alpha,
        c_var_by_tier,
        median_transport_per_delivery,
        processing_share,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The resulting variable processing costs range from €7.07 per delivery for the smallest tier down to €1.37 for the largest — a 5.17x reduction, matching $\sqrt{s_h} \approx 5.17$ as expected under $\alpha = 0.5$. This is substantially smaller than the 26.7x difference in fixed costs between the same two tiers, confirming that the dampening meaningfully moderates the raw fixed-cost ratio rather than passing it through unchanged. All values remain well below the median transport cost per delivery (€47.12), consistent with in-house processing being cheaper per unit than a driven tour stop.

    All parameters required to build the extended model are now in place.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The complete extended model, as we implement it

    **Sets and indices**
    - $i$ — cash center (warehouse), $i\in\{1,\dots,42\}$
    - $j$ — customer region, $j\in\{1,\dots,515\}$
    - Only pairs $(i,j)$ that are reachable within a single shift are considered
      (2,661 of the 21,630 possible pairs), all others are simply
      not included as variables.

    | Symbol | Meaning | Source |
    |--------|---------|--------|
    | $c_{ij}$ | annual transport cost, center $i$ to region $j$ | derived & validated in previous part |
    | $d_j$ | annual demand (deliveries) of region $j$ | `regions["yearlyDemand"]` |
    | $f_i$ | annual fixed cost of center $i$ | `warehouses["fixedCosts"]` (real, individual) |
    | $\text{tier}(i)$ | size tier of center $i$ (v/s/m/l/h) | derived from $f_i$ (exactly 5 distinct values) |
    | $c_{\text{tier}(i)}^{\mathrm{var}}$ | processing cost per delivery for center $i$'s tier | derived from fixed-cost scaling + damping ($\alpha=0.5$) + transport-cost anchor |

    **Decision variables**
    - $x_{ij}\in\{0,1\}$ — region $j$ is served by center $i$
    - $y_i\in\{0,1\}$ — center $i$ is open

    **Objective**

    $$\min \sum_{(i,j)\text{ reachable}} x_{ij}\Big(c_{ij} + c_{\text{tier}(i)}^{var}\cdot d_j\Big) \;+\; \sum_i f_i\,y_i$$

    **Constraints**

    $$x_{ij} \le y_i \qquad \forall (i,j)\text{ reachable}$$
    $$\sum_{i:(i,j)\text{ reachable}} x_{ij} = 1 \qquad \forall j$$
    $$x_{ij}, y_i \in \{0,1\}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the annual demand assigned to warehouse $(i)$ is given by

    $$z_i=\sum_{j:(i,j)\text{ reachable}} d_j\, x_{ij}$$

    it is already a linear expression in the assignment variables. Substituting this expression directly into the objective yields an equivalent formulation in which the variable processing cost is incorporated into the effective assignment cost

    $$c_{ij}+c_{\text{tier}(i)}^{var}d_j$$

    This avoids introducing an additional decision variable while preserving the linearity of the optimization model.
    """)
    return


@app.cell
def _(c_var_by_tier, cost_base, regions_flat, warehouses_flat):
    # step 1: calculating effective costs

    model_data = (
        cost_base
        .merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
        .merge(warehouses_flat[["warehouseID", "tier", "fixedCosts"]], on="warehouseID")
    )
    model_data["c_var_i"] = model_data["tier"].map(c_var_by_tier)
    model_data["effective_cost"] = (
        model_data["transportationCosts"] + model_data["c_var_i"] * model_data["yearlyDemand"]
    )

    model_data[["warehouseID", "regionID", "transportationCosts", "c_var_i", "effective_cost"]].head()
    return (model_data,)


@app.cell
def _(model_data, pl, regions_flat, warehouses_flat):
    # step 2: pulp problem setup

    prob = pl.LpProblem("CashLog_Extended", pl.LpMinimize)

    # x_ij nur für die 2.661 tatsächlich erreichbaren Paare, kein Big-M nötig 
    link_keys = list(zip(model_data["warehouseID"], model_data["regionID"]))
    x = pl.LpVariable.dicts("x", link_keys, cat="Binary")

    # y_i für alle 42 Center
    all_warehouse_ids = warehouses_flat["warehouseID"].tolist()
    y = pl.LpVariable.dicts("y", all_warehouse_ids, cat="Binary")

    #objective function

    transport_and_processing_cost = pl.lpSum(
        row.effective_cost * x[(row.warehouseID, row.regionID)]
        for row in model_data.itertuples()
    )
    fixed_cost = pl.lpSum(
        f * y[i] for i, f in zip(warehouses_flat["warehouseID"], warehouses_flat["fixedCosts"])
    )

    prob += transport_and_processing_cost + fixed_cost

    #constraints

    # Region kann nur einem offenen Center zugeordnet werden
    for i, j in link_keys:
        prob += x[(i, j)] <= y[i]

    # jede Region wird genau einem (erreichbaren) Center zugeordnet
    for j in regions_flat["regionID"]:
        reachable_i = [i for (i, jj) in link_keys if jj == j]
        prob += pl.lpSum(x[(i, j)] for i in reachable_i) == 1

    #solve the problem

    status = prob.solve(pl.PULP_CBC_CMD(msg=1))
    print("Status:", pl.LpStatus[status])
    print("Total cost: €", round(pl.value(prob.objective), 2))
    return all_warehouse_ids, link_keys, x, y


@app.cell
def _(all_warehouse_ids, display, link_keys, warehouses_flat, x, y):
    open_centers = [_i for _i in all_warehouse_ids if y[_i].value() == 1]
    n_open = len(open_centers)
    print(f'Open centers: {n_open} of 42')
    print(f'Closed centers: {42 - n_open}')
    assigned_tiers = warehouses_flat.set_index('warehouseID').loc[open_centers, ['city', 'fixedCosts', 'tier']]
    print(f"Overview of the {assigned_tiers['city'].count()} open centers:")
    assignment = [(_i, _j) for _i, _j in link_keys if x[_i, _j].value() == 1]
    print(f'Total assignments: {len(assignment)}')
    num_tiers = assigned_tiers['tier'].value_counts().sort_index()
    display(assigned_tiers.sort_values('fixedCosts'))
    print(num_tiers)
    return assignment, open_centers


@app.cell
def _(
    assignment,
    display,
    folium,
    mcolors,
    open_centers,
    pd,
    plt,
    regions_flat,
    warehouses_flat,
):
    # Karte 1: Offene/geschlossene Lager
    _center_lat = warehouses_flat['lat'].mean()
    _center_lon = warehouses_flat['lon'].mean()
    m1 = folium.Map(location=[_center_lat, _center_lon], zoom_start=6, tiles='cartodbpositron')
    for _row in warehouses_flat.itertuples():
        is_open = _row.warehouseID in open_centers
        folium.CircleMarker(location=[_row.lat, _row.lon], radius=9 if is_open else 5, color='#2ca02c' if is_open else '#d62728', fill=True, fill_opacity=0.85, popup=f'{_row.city} (ID {_row.warehouseID})').add_to(m1)
    assignment_df = pd.DataFrame(assignment, columns=['warehouseID', 'regionID'])
    assignment_df = assignment_df.merge(regions_flat[['regionID', 'lat', 'lon', 'city']], on='regionID')
    m2 = folium.Map(location=[_center_lat, _center_lon], zoom_start=6, tiles='cartodbpositron')
    cmap = plt.cm.get_cmap('tab20', len(open_centers))
    color_by_center = {wid: mcolors.to_hex(cmap(_i)) for _i, wid in enumerate(open_centers)}
    color_by_center = {wid: mcolors.to_hex(cmap(_i)) for _i, wid in enumerate(open_centers)}
    for _row in assignment_df.itertuples():
        folium.CircleMarker(location=[_row.lat, _row.lon], radius=2, color=color_by_center.get(_row.warehouseID, '#999999'), fill=True, fill_opacity=0.6).add_to(m2)
    for _row in warehouses_flat.itertuples():
        is_open = _row.warehouseID in open_centers
        folium.CircleMarker(location=[_row.lat, _row.lon], radius=8 if is_open else 5, color=color_by_center.get(_row.warehouseID, '#999999'), fill=True, fill_color=color_by_center.get(_row.warehouseID, '#999999'), fill_opacity=1).add_to(m2)
    display(m1)
    # Karte 2: Regionen-Zuordnung
    # Zuordnung als DataFrame: welche Region gehört zu welchem offenen Center
    # Lagerstandorte
    # Regionen
    display(m2)
    return (assignment_df,)


@app.cell
def _(assignment_df, display, regions_flat):
    madrid_region_id = regions_flat.loc[regions_flat["city"].str.contains("MADRID", case=False), "regionID"]
    assignment_df[assignment_df["regionID"].isin(madrid_region_id)]

    barcelona_region_id = regions_flat.loc[regions_flat["city"].str.contains("BARCELONA", case=False), "regionID"]
    assignment_df[assignment_df["regionID"].isin(barcelona_region_id)]

    display(assignment_df[assignment_df["regionID"].isin(madrid_region_id)])
    display(assignment_df[assignment_df["regionID"].isin(barcelona_region_id)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimal network keeps 18 of 42 centers open, serving all 515 regions. The open set is dominated by small ("v") tier centers (13 of 18), together with a handful of "s" tier and one "m" tier center, but no "l" or "h" tier centers. This suggests that, under our cost assumptions, the fixed-cost savings of many small centers outweigh the processing-cost efficiencies that a few very large centers would offer. No single region's demand concentration is large enough to justify the €15M+ or €35M+ fixed cost, even at the correspondingly lower per-delivery processing cost.

    The map confirms that this is not merely a numerical artifact, but a geographically sensible outcome. The two most expensive centers, Madrid and Barcelona, are both closed, yet the surrounding demand remains fully covered: nearby, much cheaper "s" or "v" tier centers absorb the regions that would otherwise have been served from the metropolis itself.

    This is a direct illustration of the network effects discussed at the outset of this report. A center's value cannot be judged by whether it constitutes a good location in isolation, but by whether a cheaper combination of other open centers can serve the same demand equally well. Here, the model identifies exactly such a combination: Madrid's demand (regions 90 and 244) is served by warehouse 19 (Guadalajara, tier "v", €1.34M fixed cost), and Barcelona's demand (region 438) is served by warehouse 43 (Tarragona, tier "s", €2.58M fixed cost).

    Madrid and Barcelona are not "bad locations" in themselves. On the contrary, they sit at the center of Spain's highest-demand areas. Rather, the network as a whole is cheaper without them, because their demand can be absorbed by nearby centers at a fraction of the cost a facility directly in either metropolis would incur.

    This conclusion depends on our economies-of-scale assumption ($\alpha = 0.5$); a stronger scaling effect could make large centers more attractive. This is tested explicitly in the sensitivity analysis below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Checking the No-Volume-Cap Decision

    Before proceeding with the sensitivity analysis, one further potential problem must be addressed. It was earlier decided not to impose explicit volume caps per tier, since no real capacity data was available to set them credibly. This assumption is now verified to ensure it does not compromise the validity of the results.
    """)
    return


@app.cell
def _(
    assignment_df,
    display,
    open_centers,
    regions_flat,
    shifts_long,
    warehouses_flat,
):
    volume_per_center = (
        assignment_df.merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
        .groupby("warehouseID")["yearlyDemand"]
        .sum()
        .rename("assigned_volume")
    )

    volume_check = (
        warehouses_flat.set_index("warehouseID")
        .loc[open_centers, ["city", "tier", "fixedCosts"]]
        .join(volume_per_center)
        .sort_values("tier")
    )
    display(volume_check)

    tier_volume_stats = volume_check.groupby("tier")["assigned_volume"].describe()
    display(tier_volume_stats)

    guadalajara_regions = assignment_df[assignment_df["warehouseID"] == 19]
    display(guadalajara_regions.merge(shifts_long, on=["warehouseID", "regionID"])[["regionID", "travelTime"]].describe())
    return (volume_check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Warehouse 19 (Guadalajara, tier "v", €1.34M fixed cost) is assigned 716,917 units of demand across 19 regions. Nearly 8x the median volume among "v" tier centers (91,506), and more than every "s" tier center and the single "m" tier center. This is the same location that absorbs Madrid's demand.

    Its travel times to these regions range from 6.6 to 183.2 minutes (median 83), comfortably below the 225-minute feasibility limit, so this is not an artifact of barely-reachable links being forced together. Guadalajara is simply very centrally located and can therefore reach many regions efficiently. The geographic logic is sound; the open question is a different one: whether a facility built at the fixed-cost level of the smallest tier could physically handle this volume in reality.

    This is a direct consequence of not capping volume, and it is accepted here as a disclosed limitation rather than an error. Without real capacity data, any cap that could be imposed would itself be an unfounded number, and trusting such an invented cap would not be more defensible than trusting the model's cost logic in this instance. Guadalajara's assignment is therefore flagged as a point CashLog should verify against the facility's actual physical capacity, and the sensitivity of our results to this assumption is tested directly below by re-solving the model with volume caps derived from the observed distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Comparison: Capacity Check

    As a comparison scenario, each tier's volume is capped at 1.5x its own 75th percentile from the uncapped solution, not because this represents a known real capacity, but to test how much the recommendation changes if no center is permitted to carry a disproportionate share of demand relative to its peers within the same tier.

    This percentile is computed excluding Guadalajara itself. Including the very outlier the cap is meant to constrain would let it inflate the percentile designed to limit it, weakening the cap exactly where it matters most.

    Using the raw 75th percentile as a hard cap would still penalize any center simply on the higher end of normal variation within its tier, rather than only genuine outliers. The 1.5x multiplier is itself a judgment call, chosen to provide reasonable headroom for normal variation while still meaningfully constraining extreme cases such as Guadalajara (716,917 — still nearly 3.7x even this corrected "v" tier cap).

    From this point on, models must be re-solved repeatedly, both for the capacity checks here and for the sensitivity analysis that follows. Rather than duplicating the PuLP setup each time, it is wrapped into a function using the same objective, constraints, and logic as the base model above, with adjustable parameters used throughout the remainder of this notebook.
    """)
    return


@app.cell
def _(
    all_warehouse_ids,
    alpha,
    demand_by_region,
    median_transport_per_delivery,
    pl,
    processing_share,
    regions_flat,
    scale_factor,
    shift_cost_base,
    shift_min,
    shifts_long,
    stop_time_by_region,
    warehouses_flat,
):
    OFFICIAL_V_BOUNDS = {   #volume bounds per tier, from the official lecutre notebook
        "v": (0, 19_348),
        "s": (19_349, 45_415),
        "m": (45_416, 107_327),
        "l": (107_328, 199_999),
        "h": (200_000, 99_999_999),
    }

    def solve_network(alpha=alpha, demand_factor=1.0, shift_cost=shift_cost_base,
                       shift_min=shift_min, travel_factor=1.0, processing_share=processing_share, force_open=None, force_closed=None, use_official_bounds=False):
        # 1. variable costs per tier unter den neuen Annahmen
        c_var_anchor_local = processing_share * median_transport_per_delivery   
        c_var = {t: c_var_anchor_local / (scale_factor[t] ** alpha) for t in scale_factor}

        # 2. effektive Kosten neu berechnen (Nachfrage-/Schicht-/Fahrzeit-Faktoren einbezogen)
        usable = shift_min - 2 * shifts_long["travelTime"] * travel_factor
        feasible = usable > 0
        dem = shifts_long["regionID"].map(demand_by_region) * demand_factor
        mps = shifts_long["regionID"].map(stop_time_by_region)
        cost = shift_cost * dem * mps / usable

        md = shifts_long[feasible].copy()
        md["transportationCosts"] = cost[feasible]
        md = md.merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
        md = md.merge(warehouses_flat[["warehouseID", "tier", "fixedCosts"]], on="warehouseID")
        md["yearlyDemand"] = md["yearlyDemand"] * demand_factor
        md["c_var_i"] = md["tier"].map(c_var)
        md["effective_cost"] = md["transportationCosts"] + md["c_var_i"] * md["yearlyDemand"]

        # 3. PuLP-Modell
        keys = list(zip(md["warehouseID"], md["regionID"]))
        xs = pl.LpVariable.dicts("x", keys, cat="Binary")
        ys = pl.LpVariable.dicts("y", all_warehouse_ids, cat="Binary")

        prob = pl.LpProblem("scenario", pl.LpMinimize)
        prob += pl.lpSum(row.effective_cost * xs[(row.warehouseID, row.regionID)] for row in md.itertuples()) \
              + pl.lpSum(f * ys[i] for i, f in zip(warehouses_flat["warehouseID"], warehouses_flat["fixedCosts"]))
        for i, j in keys:
            prob += xs[(i, j)] <= ys[i]
        for j in regions_flat["regionID"]:
            reach = [i for (i, jj) in keys if jj == j]
            prob += pl.lpSum(xs[(i, j)] for i in reach) == 1

        #falls volumen caps verwendet werden
        if use_official_bounds:
            tier_by_wh = warehouses_flat.set_index("warehouseID")["tier"]
            demand_lookup = md.set_index(["warehouseID", "regionID"])["yearlyDemand"]
            for i in all_warehouse_ids:
                lb, ub = OFFICIAL_V_BOUNDS[tier_by_wh[i]]
                links_i = [(ii, j) for (ii, j) in keys if ii == i]
                demand_expr = pl.lpSum(demand_lookup[(i, j)] * xs[(i, j)] for (ii, j) in links_i)
                prob += demand_expr <= ub * ys[i]
                prob += demand_expr >= lb * ys[i]

        #force_open/force_closed constraints
        for i in (force_open or []):
            prob += ys[i] == 1
        for i in (force_closed or []):
            prob += ys[i] == 0

        prob.solve(pl.PULP_CBC_CMD(msg=0))
        open_ids = [i for i in all_warehouse_ids if ys[i].value() == 1]
        assignment = [(i, j) for (i, j) in keys if xs[(i, j)].value() == 1]
        return {
            "status": pl.LpStatus[prob.status],
            "total_cost": pl.value(prob.objective),
            "open_centers": open_ids,
            "n_open": len(open_ids),
            "assignment": assignment,
        }

    return OFFICIAL_V_BOUNDS, solve_network


@app.cell
def _(volume_check):
    GUADALAJARA_ID = 19

    volume_excl_outlier = volume_check.drop(index=GUADALAJARA_ID)  # falls warehouseID der Index ist
    tier_volume_stats_clean = volume_excl_outlier.groupby("tier")["assigned_volume"].describe()

    V_ub_by_tier = (tier_volume_stats_clean["75%"] * 1.5).to_dict()
    # Für Tiers ohne beobachtetes Volumen in der Basislösung (l, h) setzen wir keine cap, da keine Referenzverteilung existiert, aus der sich ein plausibler Cap ableiten ließe.
    for missing_tier in ["l", "h"]:
        if missing_tier not in V_ub_by_tier:
            V_ub_by_tier[missing_tier] = float("inf")

    V_ub_by_tier
    return (V_ub_by_tier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: the "m" tier's cap is based on a single observed center (Alicante), so its 75th percentile is simply that center's own volume . A less robust basis than the "v" and "s" tier caps, which are each derived from a larger number of centers. This is nonetheless treated as a reasonable approximation, since Alicante's assigned volume is, by construction, well within this self-referential cap.
    """)
    return


@app.cell
def _(
    V_ub_by_tier,
    all_warehouse_ids,
    link_keys,
    model_data,
    pl,
    regions_flat,
    warehouses_flat,
):
    warehouse_tier = warehouses_flat.set_index("warehouseID")["tier"]


    prob_capped = pl.LpProblem("CashLog_Extended_Capped", pl.LpMinimize)
    x_c = pl.LpVariable.dicts("x", link_keys, cat="Binary")
    y_c = pl.LpVariable.dicts("y", all_warehouse_ids, cat="Binary")


    prob_capped += pl.lpSum(
        row.effective_cost * x_c[(row.warehouseID, row.regionID)] for row in model_data.itertuples()
    ) + pl.lpSum(f * y_c[wh] for wh, f in zip(warehouses_flat["warehouseID"], warehouses_flat["fixedCosts"]))


    for wh, reg in link_keys:
        prob_capped += x_c[(wh, reg)] <= y_c[wh]

    for reg in regions_flat["regionID"]:
        reachable_wh = [wh for (wh, jj) in link_keys if jj == reg]
        prob_capped += pl.lpSum(x_c[(wh, reg)] for wh in reachable_wh) == 1


    demand_by_link = model_data.set_index(["warehouseID", "regionID"])["yearlyDemand"]

    for wh in all_warehouse_ids:
        cap = V_ub_by_tier[warehouse_tier[wh]]
        if cap == float("inf"):
            continue  # l/h bleiben ungekappt, siehe Begründung oben

        links_for_wh = [(ii, reg) for (ii, reg) in link_keys if ii == wh]

        prob_capped += pl.lpSum(
            demand_by_link[(wh, reg)] * x_c[(wh, reg)] for (ii, reg) in links_for_wh
        ) <= cap * y_c[wh]


    status_c = prob_capped.solve(pl.PULP_CBC_CMD(msg=0))
    print("Status:", pl.LpStatus[status_c])
    print("Total cost: €", round(pl.value(prob_capped.objective), 2))
    return x_c, y_c


@app.cell
def _(all_warehouse_ids, display, warehouses_flat, y_c):
    open_centers_capped = [_i for _i in all_warehouse_ids if y_c[_i].value() == 1]
    print(f'Open centers: {len(open_centers_capped)} of 42')
    display(warehouses_flat.set_index('warehouseID').loc[open_centers_capped, ['city', 'tier', 'fixedCosts']].sort_values('tier'))
    display(warehouses_flat.set_index('warehouseID').loc[open_centers_capped, 'tier'].value_counts())
    if 19 in open_centers_capped:
        print('Guadalajara still open')
    else:
        print('Guadalajara closed in the capped scenario')
    return (open_centers_capped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The capped comparison confirms that Guadalajara's role in the base solution was not incidental. When no center may exceed the 1.5x-buffered, outlier-excluded volume typical for its tier, the network changes in a very specific way: Guadalajara, which previously absorbed 716,917 units including Madrid's demand, closes, and Madrid itself reopens. This time as an "h" tier facility. The model effectively substitutes one large, expensive hub for another, rather than spreading Madrid's demand thinly across several small centers. This indicates that the demand surrounding Madrid is large enough to require either (a) a genuinely large facility, or (b) an artificially overloaded small one; a "normal-sized" small center cannot absorb it.

    The network also shifts from a predominantly "v" tier composition (13 of 18) to a more balanced mix (11 "v", 6 "s", one "m", one "h"), with one additional center open overall (19 vs. 18). Total cost rises by 25.4% (+€29.7M) — a substantial increase, confirming that a meaningful share of the base solution's efficiency stemmed specifically from allowing Guadalajara to operate far outside its tier's typical volume range.

    | Metric | Uncapped (base) | Capped |
    |---|---:|---:|
    | Total annual cost | €117,082,899 | €146,803,705 |
    | Cost difference | — | +€29,720,806 (+25.4%) |
    | Open centers | 18 | 19 |
    | Tier mix (v / s / m / l / h) | 13 / 4 / 1 / 0 / 0 | 11 / 6 / 1 / 0 / 1 |
    | Guadalajara (WH 19) | open | closed |
    | Madrid (WH 28) | closed | open (h-tier) |

    The unconstrained solution represents the cost-minimizing network under the assumptions of the optimization model. However, it assigns a volume to Guadalajara that may not be physically realistic for that facility, raising the question of whether this allocation would be operationally feasible in practice.

    To further assess the robustness of this result, a second sanity check is performed using the volume capacity limits provided in the course notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.5 A Second Capacity Check: The Course's Official Volume Bounds

    As an additional, independently sourced comparison, the course's own extended-model template (`advancedAnalysis.py`) specifies concrete volume bounds per tier. The model is re-solved using these official bounds, alongside the percentile-based cap derived above.
    """)
    return


@app.cell
def _(solve_network):
    official_capped = solve_network(use_official_bounds=True)
    print(f"Status: {official_capped['status']}")
    print(f"Cost: €{official_capped['total_cost']:,.0f}")
    print(f"Open centers: {official_capped['n_open']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Under these capacity constraints, the model becomes infeasible, indicating that no network configuration can simultaneously satisfy all customer demand while respecting the prescribed facility capacity limits. To better understand this result, the potential causes of the infeasibility are examined.

    The first question is whether the specified volume limits are overly restrictive in general. To assess this, the maximum throughput that could be achieved if all 42 candidate distribution centers were opened and operated at their respective upper capacity bounds is compared against total annual customer demand.
    """)
    return


@app.cell
def _(OFFICIAL_V_BOUNDS, all_warehouse_ids, regions_flat, warehouses_flat):
    #generell mit allen 42 strukturell unmöglich gesamte nachfrage zu bedienen?
    total_capacity_all_open = sum((OFFICIAL_V_BOUNDS[warehouses_flat.set_index('warehouseID')['tier'][_i]][1] for _i in all_warehouse_ids))
    print(f'Max capacity if ALL 42 centers were open at their upper bound: {total_capacity_all_open:,.0f}')
    print(f"Total demand: {regions_flat['yearlyDemand'].sum():,.0f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The combined maximum capacity of all potential facilities amounts to approximately 101.9 million units, whereas total annual demand is only about 2.9 million units. This confirms that the capacity limits are not restrictive at the network level; the infeasibility must therefore arise from the interaction between the capacity constraints and other model restrictions, rather than from insufficient total capacity.

    A more likely explanation is the interaction between the upper capacity limits and the assignment constraints. In particular, a single region whose annual demand exceeds the capacity of every reachable cash center would render the model infeasible, even if substantial unused capacity remained elsewhere in the network.

    To investigate whether the lower volume bounds contribute to this behavior, the model is solved again after temporarily removing the minimum volume constraints for the selected cash centers. This isolates the effect of the upper capacity limits and clarifies whether the lower bounds are responsible for the infeasibility.
    """)
    return


@app.cell
def _(
    OFFICIAL_V_BOUNDS,
    all_warehouse_ids,
    alpha,
    demand_by_region,
    median_transport_per_delivery,
    pl,
    processing_share,
    regions_flat,
    scale_factor,
    shift_cost_base,
    shift_min,
    shifts_long,
    stop_time_by_region,
    warehouses_flat,
):
    #debugging function to check if the model is feasible with official volume bounds
    def solve_network_debug(use_upper_only=True, alpha=alpha, demand_factor=1.0, shift_cost=shift_cost_base,
                      force_open=None, force_closed=None, travel_factor=1.0, use_official_bounds=False):
        # 1. variable costs per tier unter den neuen Annahmen
        c_var_anchor_local = processing_share * median_transport_per_delivery   
        c_var = {t: c_var_anchor_local / (scale_factor[t] ** alpha) for t in scale_factor}

        # 2. effektive Kosten neu berechnen (Nachfrage-/Schicht-/Fahrzeit-Faktoren einbezogen)
        usable = shift_min - 2 * shifts_long["travelTime"] * travel_factor
        feasible = usable > 0
        dem = shifts_long["regionID"].map(demand_by_region) * demand_factor
        mps = shifts_long["regionID"].map(stop_time_by_region)
        cost = shift_cost * dem * mps / usable

        md = shifts_long[feasible].copy()
        md["transportationCosts"] = cost[feasible]
        md = md.merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
        md = md.merge(warehouses_flat[["warehouseID", "tier", "fixedCosts"]], on="warehouseID")
        md["yearlyDemand"] = md["yearlyDemand"] * demand_factor
        md["c_var_i"] = md["tier"].map(c_var)
        md["effective_cost"] = md["transportationCosts"] + md["c_var_i"] * md["yearlyDemand"]

        # 3. PuLP-Modell
        keys = list(zip(md["warehouseID"], md["regionID"]))
        xs = pl.LpVariable.dicts("x", keys, cat="Binary")
        ys = pl.LpVariable.dicts("y", all_warehouse_ids, cat="Binary")

        prob = pl.LpProblem("scenario", pl.LpMinimize)
        prob += pl.lpSum(row.effective_cost * xs[(row.warehouseID, row.regionID)] for row in md.itertuples()) \
              + pl.lpSum(f * ys[i] for i, f in zip(warehouses_flat["warehouseID"], warehouses_flat["fixedCosts"]))
        for i, j in keys:
            prob += xs[(i, j)] <= ys[i]
        for j in regions_flat["regionID"]:
            reach = [i for (i, jj) in keys if jj == j]
            prob += pl.lpSum(xs[(i, j)] for i in reach) == 1
        print(md.set_index(["warehouseID", "regionID"]).index.duplicated().sum())
        #falls volumen caps verwendet werden
        if use_official_bounds:
            tier_by_wh = warehouses_flat.set_index("warehouseID")["tier"]
            demand_lookup = md.set_index(["warehouseID", "regionID"])["yearlyDemand"]
            for i in all_warehouse_ids:
                lb, ub = OFFICIAL_V_BOUNDS[tier_by_wh[i]]
                links_i = [(ii, j) for (ii, j) in keys if ii == i]
                demand_expr = pl.lpSum(demand_lookup[(i, j)] * xs[(i, j)] for (ii, j) in links_i)
                prob += demand_expr <= ub * ys[i]
                #prob += demand_expr >= lb * ys[i]

        #force_open/force_closed constraints
        for i in (force_open or []):
            prob += ys[i] == 1
        for i in (force_closed or []):
            prob += ys[i] == 0

        prob.solve(pl.PULP_CBC_CMD(msg=0))
        open_ids = [i for i in all_warehouse_ids if ys[i].value() == 1]
        assignment = [(i, j) for (i, j) in keys if xs[(i, j)].value() == 1]
        return {
            "status": pl.LpStatus[prob.status],
            "total_cost": pl.value(prob.objective),
            "open_centers": open_ids,
            "n_open": len(open_ids),
            "assignment": assignment,
            "md": md,
        }


    official_capped_debug = solve_network_debug(use_official_bounds=True)
    print(f"Status: {official_capped_debug['status']}")
    print(f"Cost: €{official_capped_debug['total_cost']:,.0f}")
    print(f"Open centers: {official_capped_debug['n_open']}")
    return (solve_network_debug,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After removing the minimum volume constraints, the optimization model remains infeasible. This indicates that the lower capacity bounds are not responsible for the infeasibility, and the conflicting constraints must therefore originate elsewhere in the model.

    The next step is to examine whether any individual region cannot be assigned to a feasible cash center under the official capacity limits. For each region, the maximum capacity of all reachable cash centers is compared against the region's annual demand.
    """)
    return


@app.cell
def _(OFFICIAL_V_BOUNDS, regions_flat, solve_network_debug):
    result_for_diag = solve_network_debug(use_official_bounds=True)
    md_diag = result_for_diag["md"]

    region_max_reachable_cap = (
        md_diag.assign(tier_ub=md_diag["tier"].map(lambda t: OFFICIAL_V_BOUNDS[t][1]))
        .groupby("regionID")["tier_ub"]
        .max()
    )

    demand_by_region_series = regions_flat.set_index("regionID")["yearlyDemand"]

    impossible_regions = demand_by_region_series[
        demand_by_region_series > region_max_reachable_cap.reindex(demand_by_region_series.index)
    ]
    print(f"Regions that cannot be assigned to ANY reachable center under official bounds: {len(impossible_regions)}")
    impossible_regions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This analysis identifies exactly one problematic region. Region 410 has an annual demand of 59,646 units, which exceeds the capacity of every cash center able to serve it. Consequently, no feasible assignment exists for this region, regardless of the available capacity elsewhere in the network. This explains the infeasibility observed in the optimization model and confirms that it is caused by a local capacity bottleneck rather than by insufficient overall network capacity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Official Bounds Are Infeasible for This Dataset

    Re-solving the optimization model using the official volume bounds from the course notebook results in an infeasible solution. This outcome is not caused by an implementation error; rather, the analysis reveals that one region (regionID 410, with an annual demand of 59,646 deliveries) exceeds the maximum capacity of every cash center within its service range. As a result, this region cannot be assigned to any feasible facility without violating the corresponding upper capacity limit, rendering the overall optimization problem infeasible.

    This finding demonstrates that the official tier-based capacity bounds cannot be applied directly to the demand distribution of the present dataset, and consequently cannot serve as a meaningful benchmark for evaluating alternative network designs in this case study. The percentile-based capacity limits introduced earlier are therefore used for the robustness analysis instead, while the infeasibility of the official bounds is reported as an important observation regarding the applicability of the reference model to this specific problem instance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transitioning to the Sensitivity Analysis

    Having validated the base model and identified its main structural limitation — Guadalajara's uncapped volume requirement — along with the cost impact of the more conservative capped alternative, the analysis now shifts from internal feasibility checks to external sensitivity.

    While the preceding sections focused on whether the model is internally consistent and operationally feasible under different capacity assumptions, the following analysis examines how robust the recommended network is to changes in external conditions. In particular, it investigates how variations in cash demand, wage and fuel costs, technological improvements, and the assumed economies of scale affect the optimal network configuration and total cost. This constitutes the sensitivity analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sensitivity Analysis

    The robustness of the recommended network is evaluated along four dimensions:

    1. **Cost parameters: economies-of-scale strength ($\alpha$) and the processing-cost anchor** — these represent internal modeling assumptions. The analysis examines whether the observed preference for smaller centers remains stable under stronger or weaker scale effects.
    2. **Declining cash demand** — reflecting industry-wide reductions in cash usage, scenarios of -10%, -30%, and -50% demand are considered.
    3. **Increasing wages and fuel costs** — operational cost sensitivity is assessed by systematically increasing shift-related costs.
    4. **Technological development** — potential innovations such as autonomous vehicles are represented through reduced crew costs, extended usable shifts, and improved travel efficiency.

    For each scenario, the optimization model is re-solved and the resulting network is compared to the baseline solution in terms of total cost, number of open centers, and changes in facility activation status. The results are then consolidated into a robustness assessment distinguishing between consistently selected facilities ("keep"), consistently closed facilities ("close"), and scenario-dependent facilities ("uncertain") requiring further consideration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 a Economies-of-scale strength ($\alpha$)

    The economies-of-scale parameter $\alpha$, scales how strongly variable processing costs decrease with increasing facility size. We test the sensitivity of our results by re-solving the model across the full range of $\alpha \in [0, 1]$ in steps of 0.05, while keeping all other parameters fixed at their baseline levels. This lets us plot the cost response over the entire range and then look at specific values more concretely. In particular, we examine $\alpha = 0.3$, $\alpha = 0.5$ (our baseline), $\alpha = 0.7$ and $\alpha = 1.0$ in more detail.

    Stronger economies of scale are expected to concentrate volume in fewer, larger centers, whereas weaker economies of scale should distribute volume more evenly across facilities. However, since $\alpha$ primarily affects the relative cost advantage between tiers rather than shifting absolute cost levels, its impact on the overall network structure is expected to be less impactful than Parameters changing the entire cost structure.
    """)
    return


@app.cell(hide_code=True)
def _(np, pd, plt, solve_network):
    alphas = np.linspace(0, 1, 21)   

    results_alphas = []

    for ap in alphas:
        result_alphas = solve_network(alpha=ap)
        results_alphas.append({
            "alpha": ap,
            "total_cost": result_alphas["total_cost"]
        })

    df_alphas = pd.DataFrame(results_alphas)

    plt.figure(figsize=(7, 4))
    plt.plot(df_alphas["alpha"], df_alphas["total_cost"], marker="o")
    plt.xlabel("Alpha")
    plt.ylabel("Total Cost (€)")
    plt.title("Total Cost over Alpha")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(pd, solve_network):
    alpha_results = {a: solve_network(alpha=a) for a in [0.3, 0.5, 0.7, 1.0]}
    pd.DataFrame({a: {"cost": r["total_cost"], "n_open": r["n_open"]} for a, r in alpha_results.items()}).T
    return (alpha_results,)


@app.cell
def _(alpha_results, display, warehouses_flat):
    # Welche Center gehen bei alpha=0.5 vs alpha=1.0 verloren?
    closed_at_high_alpha = set(alpha_results[0.5]["open_centers"]) - set(alpha_results[1.0]["open_centers"])
    print("Centers open at α=0.5 but closed at α=1.0:", closed_at_high_alpha)
    display(warehouses_flat.loc[
        warehouses_flat["warehouseID"].isin(closed_at_high_alpha), 
        ["warehouseID", "city", "tier", "fixedCosts"]])

    opened_at_high_alpha = set(alpha_results[1.0]["open_centers"]) - set(alpha_results[0.5]["open_centers"])
    print("Centers open at α=1.0 but not at α=0.5:", opened_at_high_alpha)

    warehouses_flat.loc[
        warehouses_flat["warehouseID"].isin(opened_at_high_alpha),
        ["warehouseID", "city", "tier", "fixedCosts"]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall, total cost decreases smoothly and monotonically as $\alpha$ increases, falling by only about 3% across the full range from $\alpha = 0$ to $\alpha = 1$. This confirms the expectation that $\alpha$ has a relatively minor effect on overall cost levels, with no abrupt jumps that would indicate major structural changes in the network.

    At $\alpha = 1.0$, five small ("v"/"s") tier cash centers close: Avila, Cordoba, Guadalajara, Huelva, and Orense. At the same time, four larger centers open: Pontevedra, Sevilla, Toledo, and Zamora. From a geographical perspective, this pattern appears consistent with an intuitive redistribution of service areas: based on their locations, Sevilla likely absorbs demand previously served by Huelva and Cordoba, Toledo likely takes over the service areas of Avila and Guadalajara, and Pontevedra appears to replace Orense in the northwest region. These interpretations are based solely on geographic proximity.

    The exact reassignment of regions is analyzed in the following section to confirm and quantify these changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verifying the reassignment
    """)
    return


@app.cell
def _(alpha_results, display, pd, warehouses_flat):
    base_assignment = pd.DataFrame(alpha_results[0.5]['assignment'], columns=['warehouseID', 'regionID'])
    a1_assignment = pd.DataFrame(alpha_results[1.0]['assignment'], columns=['warehouseID', 'regionID'])
    merged_1 = base_assignment.merge(a1_assignment, on='regionID', suffixes=('_base', '_a1'))
    # Zusammenführen: für jede Region, wer bediente sie bei α=0.5, wer bei α=1.0
    changed = merged_1[merged_1['warehouseID_base'] != merged_1['warehouseID_a1']]
    reassignment_summary = changed.groupby(['warehouseID_base', 'warehouseID_a1']).size().rename('n_regions').reset_index()
    # Nur die Regionen, deren Center sich geändert hat
    city_lookup = warehouses_flat.set_index('warehouseID')['city']
    reassignment_summary['from_city'] = reassignment_summary['warehouseID_base'].map(city_lookup)
    # Übersicht: von welchem Center zu welchem Center wie viele Regionen gewandert sind
    reassignment_summary['to_city'] = reassignment_summary['warehouseID_a1'].map(city_lookup)
    display(reassignment_summary[['from_city', 'to_city', 'n_regions']].sort_values('n_regions', ascending=False))
    print(f'Total regions with changed assignment: {len(changed)}')
    # Stadtnamen statt IDs, für Lesbarkeit
    print(f"Sum of reassignment_summary: {reassignment_summary['n_regions'].sum()}")
    return base_assignment, changed


@app.cell
def _(base_assignment, changed):
    # Wie viele Regionen gehörten den 5 geschlossenen Centern in der Basislösung?
    closed_ids = {5, 14, 19, 21, 32}
    regions_from_closed = base_assignment[base_assignment["warehouseID"].isin(closed_ids)]
    print(f"Regions that belonged to the 5 closed centers: {len(regions_from_closed)}")

    # Wie viele der "changed"-Regionen kamen nicht von einem der 5 geschlossenen Center?
    changed_not_from_closed = changed[~changed["warehouseID_base"].isin(closed_ids)]
    print(f"Changed regions NOT from a closed center: {len(changed_not_from_closed)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tracing the actual region-level reassignment confirms the geographic intuition, although the pattern is more nuanced. In total, 189 of 515 regions are reassigned to a different center when $\alpha = 1.0$.

    A substantial share of these changes is directly driven by the closure of the five small centers, which together account for 158 of the 189 reassigned regions. Sevilla alone absorbs all 35 regions previously served by Huelva (19) and Cordoba (16), fully consistent with the initial hypothesis. The remaining 31 reassigned regions, however, are redistributed among centers that remain active in both scenarios, indicating that changes in $\alpha$ affect relative cost structures across the entire network rather than only locally around the facilities that open and close.

    Despite this additional reshuffling, the broader structural pattern remains stable: stronger economies of scale favor a moderate consolidation toward higher-tier centers, rather than extreme centralization. This contrasts with the capped-volume scenario analyzed earlier, where capacity constraints forced the activation of a full "h" tier center (Madrid).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1b Processing-Cost Anchor Sensitivity

    Besides $\alpha$, the second free parameter in the derivation of $c_t^{var}$ is the 15% share of median transport cost per delivery used to anchor $c_v^{var}$. The sensitivity of this assumption is assessed by re-solving the model across the full range from 10% to 25% in steps of 1%, while keeping $\alpha = 0.5$ fixed. This allows the cost response to be plotted over the entire range before examining the anchor shares of 10%, 15% (the baseline), and 25% more concretely.

    Increasing the anchor share raises $c_v^{var}$ proportionally across all tiers, thereby increasing the relative cost of in-house processing compared to transportation. In principle, this should shift the network toward a stronger emphasis on transport cost efficiency. However, since this adjustment affects all tiers uniformly, its impact is expected to be much smaller than that of $\alpha$, which alters the relative cost differences between tiers rather than only shifting the overall cost level.
    """)
    return


@app.cell(hide_code=True)
def _(np, pd, plt, solve_network):
    shares = np.round(np.arange(0.10, 0.251, 0.01), 3)
    res_anchor = [{"share": s, "total_cost": solve_network(processing_share=s)["total_cost"]}
                  for s in shares]
    df_anchor = pd.DataFrame(res_anchor)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_anchor["share"], df_anchor["total_cost"], marker="o")
    ax.set_xlabel("Processing-cost anchor share")
    ax.set_ylabel("Total Cost (€)")
    ax.set_title("Total Cost over anchor share")
    ax.grid(True)
    fig
    return (ax,)


@app.cell
def _(display, pd, solve_network, warehouses_flat):
    anchor_results = {share: solve_network(processing_share=share) for share in [0.1, 0.15, 0.25]}
    anchor_df = pd.DataFrame({share: {'cost': r['total_cost'], 'n_open': r['n_open']} for share, r in anchor_results.items()}).T
    display(anchor_df)
    for share in [0.1, 0.15, 0.25]:
        _tiers = warehouses_flat.set_index('warehouseID').loc[anchor_results[share]['open_centers'], 'tier'].value_counts()
        print(f"processing_share={share}: n_open={anchor_results[share]['n_open']}, tiers={_tiers.to_dict()}")
    return (anchor_results,)


@app.cell
def _(anchor_results, warehouses_flat):
    closed_anchor = set(anchor_results[0.15]["open_centers"]) - set(anchor_results[0.25]["open_centers"])
    print(f"Closed at anchor_factor={0.25}:", warehouses_flat.set_index("warehouseID").loc[list(closed_anchor), "city"].tolist())

    opened_anchor = set(anchor_results[0.25]["open_centers"]) - set(anchor_results[0.15]["open_centers"])
    print(f"Opened at anchor_factor={0.25}:", warehouses_flat.set_index("warehouseID").loc[list(opened_anchor), "city"].tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The processing-cost anchor shows a similarly modest sensitivity to $\alpha$: across the tested range (10% to 25%), which more than covers plausible uncertainty around the baseline value of 15%, total cost increases almost linearly with the anchor share, varying by approximately -5% to +10% relative to baseline. The tier structure changes only marginally at the upper end of the range.

    Taken together with the results for $\alpha$, this confirms that the derivation of $c_t^{var}$, although based on two explicit modeling assumptions (the anchor share and the economies-of-scale parameter), does not lead to a fragile network recommendation. When varied independently within a reasonable range, neither parameter alters the qualitative conclusion that a network dominated by smaller centers is cost-optimal.

    Interestingly, the same structural reconfiguration observed under $\alpha = 1.0$ recurs here: in both cases, the identical set of five centers closes (Avila, Cordoba, Guadalajara, Huelva, and Orense), while the identical set of four centers opens (Pontevedra, Sevilla, Toledo, and Zamora). This is not coincidental: both $\alpha$ and the anchor share shift incentives in the same direction by increasing the relative attractiveness of larger tiers, albeit through different mechanisms: relative cost curvature in the case of $\alpha$, and a uniform upward shift in processing costs in the case of the anchor share.

    The consistency of this five-center consolidation across two independent parameter variations provides a strong robustness signal. In particular, the repeated emergence of Sevilla absorbing demand from Huelva and Cordoba suggests that this restructuring is not an artifact of a specific parameter choice, but a structurally stable feature of the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Falling Cash Demand

    Cash usage has been declining industry-wide as electronic payment methods continue to grow. To capture this trend, the model is re-solved across the full range of demand reductions from 0% to 50% in steps of 5%, while keeping all other parameters at their baseline values. This allows the network response to be plotted over the entire range before examining reductions of 10%, 30%, and 50% more specifically.

    Mathematically, lower demand reduces both transport costs ($c_{ij}$) and variable processing costs ($c_t^{var} \cdot d_j$), whereas fixed facility costs ($f_i$) remain unchanged. As a result, fixed costs become increasingly dominant in the overall cost structure as demand decreases. The model is therefore expected to close more centers, concentrating the reduced demand onto fewer facilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pd, solve_network):
    demand_factors = np.round(np.arange(0.5, 1.001, 0.05), 2)  # 0.5 ... 1.0
    demand_sweep = {float(_d): solve_network(demand_factor=float(_d)) for _d in demand_factors}

    # kompakte Tabelle für Plot/Nachschlagen
    base_cost = demand_sweep[1.0]["total_cost"]
    sweep_df = pd.DataFrame({
        _d: {
            "n_open": r["n_open"],
            "total_cost": r["total_cost"],
            "pct_reduction": (1 - r["total_cost"] / base_cost) * 100,
        }
        for _d, r in demand_sweep.items()
    }).T.sort_index()

    demand_slider = mo.ui.slider(
        steps=[float(x) for x in demand_factors],
        value=1.0,
        label="Demand factor",
        show_value=True,
    )
    demand_slider
    return demand_slider, demand_sweep, sweep_df


@app.cell(hide_code=True)
def _(demand_slider, demand_sweep, mo, warehouses_flat):
    _d = demand_slider.value
    _r = demand_sweep[_d]

    _tiers = (warehouses_flat.set_index("warehouseID")
              .loc[_r["open_centers"], "tier"].value_counts().to_dict())

    _closed = set(demand_sweep[1.0]["open_centers"]) - set(_r["open_centers"])
    _closed_cities = warehouses_flat.set_index("warehouseID").loc[list(_closed), "city"].tolist()

    mo.md(f"""
    **Demand factor = {_d:.2f}**  (reduction: {(1-_d)*100:.0f}%)

    - Open centers: **{_r['n_open']}**
    - Total cost: **{_r['total_cost']:,.0f} €**
    - Cost reduction vs. baseline: **{(1 - _r['total_cost']/demand_sweep[1.0]['total_cost'])*100:.1f}%**
    - Tier breakdown: {_tiers}
    - Closed vs. baseline ({len(_closed_cities)}): {', '.join(_closed_cities) if _closed_cities else '–'}
    """)
    return


@app.cell(hide_code=True)
def _(ax, demand_slider, plt, sweep_df):
    _x = sweep_df.index.to_numpy()            # demand factors
    _y = sweep_df["n_open"].to_numpy()
    _sel = demand_slider.value

    fig_, ax_ = plt.subplots(figsize=(7, 4))
    ax_.step(_x, _y, where="mid", marker="o")
    ax_.scatter([_sel], [sweep_df.loc[_sel, "n_open"]],
               s=180, zorder=5, color="C3", label=f"selected ({_sel:.2f})")
    ax_.set_xlabel("Demand factor")
    ax_.set_ylabel("# open centers")
    ax_.set_title("Network size over demand factor")
    ax_.set_yticks(range(int(_y.min()), int(_y.max()) + 1))  # ganzzahlige Ticks
    ax_.invert_xaxis()  
    ax_.grid(True, alpha=0.3)
    ax.legend()
    fig_
    return


@app.cell(hide_code=True)
def _(display, pd, solve_network, warehouses_flat):
    demand_results = {_d: solve_network(demand_factor=_d) for _d in [1.0, 0.9, 0.7, 0.5]}
    demand_red_df = pd.DataFrame({_d: {'cost': r['total_cost'], 'n_open': r['n_open']} for _d, r in demand_results.items()}).T
    demand_red_df['pct_reduction'] = (1 - demand_red_df['cost'] / demand_red_df.loc[1.0, 'cost']) * 100
    display(demand_red_df)
    for _d in [1.0, 0.9, 0.7, 0.5]:
        _tiers = warehouses_flat.set_index('warehouseID').loc[demand_results[_d]['open_centers'], 'tier'].value_counts()
        print(f"demand_factor={_d}: n_open={demand_results[_d]['n_open']}, tiers={_tiers.to_dict()}")
    return (demand_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This expectation holds, with one notable nuance. Up to a -20% demand reduction, the network remains unchanged, indicating that moderate demand decreases are not sufficient to render any facility unprofitable. The adjustment takes effect only at higher reduction levels: one "v" tier center closes at -30% demand, followed by two additional "v" tier closures at -50%. In all cases, only "v" tier facilities are affected, while the "s" and "m" tiers remain stable across all scenarios. This confirms that the smallest centers are the first to become uneconomical as volume decreases, consistent with the dominance of fixed costs.

    A second notable finding is that total cost decreases more slowly than demand. A 50% reduction in demand leads to a cost decrease of only approximately 36.7% (from €117.1M to €74.1M), rather than a proportional reduction. This deviation is driven by fixed costs, which remain unchanged for all open facilities and therefore dampen the overall cost decline. From a managerial perspective, this implies that the network exhibits a degree of cost rigidity: even under a severe demand contraction, total costs do not fall proportionally, since a reduced number of facilities can still efficiently absorb the remaining volume.

    As a final step, the set of closed facilities is examined across scenarios, checking whether the same centers that close under declining demand also appear in the previously analyzed scaling and economies-of-scale scenarios. Such overlap would indicate structurally unstable or consistently marginal locations within the network.
    """)
    return


@app.cell(hide_code=True)
def _(demand_results, warehouses_flat):
    for _d in [0.7, 0.5]:
        closed = set(demand_results[1.0]['open_centers']) - set(demand_results[_d]['open_centers'])
        print(f'Closed at demand_factor={_d}:', warehouses_flat.set_index('warehouseID').loc[list(closed), 'city'].tolist())
        open = set(demand_results[_d]['open_centers']) - set(demand_results[1.0]['open_centers'])
        print(f'Opened at demand_factor={_d}:', warehouses_flat.set_index('warehouseID').loc[list(open), 'city'].tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interestingly, the centers affected in this scenario (Burgos and Lugo) differ from those observed in the $\alpha$ sensitivity analysis (Avila, Cordoba, Guadalajara, Huelva, and Orense). This suggests that demand shocks and changes in economies-of-scale assumptions affect different parts of the network structure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Rising Wages and Fuel Costs

    Labor and fuel costs may increase due to inflation or regulatory changes. To capture this effect, the model is re-solved across the full range from +0% to +50% shift-cost increase in steps of 5% (from €480 up to €720), while keeping all other model assumptions at their baseline values. This allows the network response to be plotted over the entire range before examining the increases of +20% (€576) and +50% (€720) more concretely.

    Shift costs directly affect the transport cost component $c_{ij}$. Higher shift costs increase the cost of every transport link, with a stronger impact on long-distance and thus less efficient assignments. As a result, the model is expected to favor a more decentralized network structure with a higher number of smaller centers, each serving geographically closer regions.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pd, shift_cost_base, solve_network):
    shift_factors = np.round(np.arange(1.0, 1.501, 0.05), 2)   # 1.0 ... 1.5
    shift_sweep = {float(f): solve_network(shift_cost=shift_cost_base * float(f))
                   for f in shift_factors}

    _base = shift_sweep[1.0]["total_cost"]
    shift_df = pd.DataFrame({
        f: {
            "n_open": r["n_open"],
            "total_cost": r["total_cost"],
            "pct_increase": (r["total_cost"] / _base - 1) * 100,
        }
        for f, r in shift_sweep.items()
    }).T.sort_index()

    shift_slider = mo.ui.slider(
        steps=[float(x) for x in shift_factors],
        value=1.0,
        label="Shift-cost factor",
        show_value=True,
    )
    return shift_df, shift_slider, shift_sweep


@app.cell(hide_code=True)
def _(
    mo,
    plt,
    shift_cost_base,
    shift_df,
    shift_slider,
    shift_sweep,
    warehouses_flat,
):
    _f = shift_slider.value
    _r = shift_sweep[_f]

    _x = shift_df.index.to_numpy()
    _y = shift_df["n_open"].to_numpy()

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.step(_x, _y, where="mid", marker="o")
    _ax.scatter([_f], [shift_df.loc[_f, "n_open"]],
                s=180, zorder=5, color="C3", label=f"selected (x{_f:.2f})")
    _ax.set_xlabel("Shift-cost factor")
    _ax.set_ylabel("# open centers")
    _ax.set_title("Network size over shift-cost factor")
    _ax.set_yticks(range(int(_y.min()), int(_y.max()) + 1))
    _ax.grid(True, alpha=0.3)
    _ax.legend()

    _tiers = (warehouses_flat.set_index("warehouseID")
              .loc[_r["open_centers"], "tier"].value_counts().to_dict())
    _opened = set(_r["open_centers"]) - set(shift_sweep[1.0]["open_centers"])
    _opened_cities = warehouses_flat.set_index("warehouseID").loc[list(_opened), "city"].tolist()

    mo.vstack([
        shift_slider,
        _fig,
        mo.md(f"""
    **Shift-cost factor = x{_f:.2f}**  (+{(_f-1)*100:.0f}%, €{shift_cost_base*_f:.0f})

    - Open centers: **{_r['n_open']}**
    - Total cost: **{_r['total_cost']:,.0f} €** (+{(_r['total_cost']/shift_sweep[1.0]['total_cost']-1)*100:.1f}%)
    - Tier breakdown: {_tiers}
    - Newly opened vs. baseline ({len(_opened_cities)}): {', '.join(_opened_cities) if _opened_cities else '–'}
    """)
    ])
    return


@app.cell(hide_code=True)
def _(display, pd, shift_cost_base, solve_network, warehouses_flat):
    shiftcost_results = {factor: solve_network(shift_cost=shift_cost_base * factor) for factor in [1.0, 1.2, 1.5]}
    shiftcost_df = pd.DataFrame({factor: {'cost': r['total_cost'], 'n_open': r['n_open']} for factor, r in shiftcost_results.items()}).T
    display(shiftcost_df)
    for factor in [1.0, 1.2, 1.5]:
        _tiers = warehouses_flat.set_index('warehouseID').loc[shiftcost_results[factor]['open_centers'], 'tier'].value_counts()
        print(f"shift_cost x{factor}: n_open={shiftcost_results[factor]['n_open']}, tiers={_tiers.to_dict()}")
    return (shiftcost_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results are more nuanced than the initial hypothesis suggested. While the expectation was that higher transport costs would favor a more decentralized network with a larger number of smaller centers, the opposite pattern emerges at the highest cost increase (+50%): the network shifts toward fewer very small centers and a greater number of medium-sized facilities (9 "v", 6 "s", 3 "m", compared to 13/4/1 in the baseline).

    This outcome is consistent with the structure of the cost model. Transport costs ($c_{ij}$) and processing costs ($c_t^{var}$) are driven by different mechanisms: increasing shift costs affects only $c_{ij}$, leaving the relative processing-cost advantages across tiers unchanged. As transport becomes uniformly more expensive, proximity to demand becomes more important; however, the optimization increasingly favors reallocating demand to already well-positioned larger or medium-sized centers rather than opening additional small facilities. As a result, consolidation into medium tiers dominates over further fragmentation into small centers.

    At intermediate levels (+10% to +30%), the network temporarily opens one additional facility before returning to 18 open centers at +50%, albeit with a different tier composition. This indicates that while the direction of the effect is consistent, the adjustment in the number of active facilities is not monotonic.

    To verify that this behavior is not an artifact of the solver, the newly opened "m" tier centers under the +50% shift-cost scenario are inspected next.
    """)
    return


@app.cell
def _(shiftcost_results, warehouses_flat):
    tier_1_5 = warehouses_flat.set_index("warehouseID").loc[shiftcost_results[1.5]["open_centers"], ["city", "tier"]]
    tier_1_5[tier_1_5["tier"].isin(["m", "l", "h"])]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alicante was already present in the base solution, whereas Sevilla and Valencia emerge as new medium-tier facilities under the +50% shift-cost scenario. The appearance of Sevilla is consistent with the $\alpha$ and processing-cost-anchor sensitivity analyses, where it also acted as a consolidation hub for southwestern Spain. The fact that three independent stress tests identify the same structural role for Sevilla strengthens the evidence that this location is a robust candidate for a higher-capacity facility, rather than an artifact of a specific parameter choice.

    Valencia, by contrast, is specific to the rising-transport-cost scenario. Its emergence suggests that higher shift costs make consolidation in the eastern coastal region economically attractive. A pattern that was not triggered under either the demand or cost-parameter variations. This highlights that different cost drivers activate different regional consolidation mechanisms within the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 New Technology

    Emerging technologies such as autonomous trucks or drone-based delivery systems could fundamentally alter the cost structure of the network. To capture these potential developments, two scenarios are considered:

    * **Autonomous (no crew):** the removal of the three-person crew reduces the dominant cost component of a shift. This is modeled as a reduction in shift costs to 40% of the current level (from €480 to €192).
    * **Autonomous + 24/7 operation + increased speed:** in addition to automation, vehicles are assumed to operate continuously (increasing usable shift time from 450 to 900 minutes) and to travel faster, reducing travel times by 20%.

    In the first scenario, lower shift costs directly reduce the transport cost component ($c_{ij}$). This lowers the penalty of serving distant regions and may therefore favor a more centralized structure, as longer routes become relatively less costly compared to maintaining additional local facilities.

    The second scenario further increases effective network reachability, as more assignments fall within feasible time constraints due to extended operating hours and reduced travel times. As a result, a more substantial restructuring of the network can be expected, potentially affecting both facility locations and assignment patterns.
    """)
    return


@app.cell
def _(
    display,
    pd,
    shift_cost_base,
    shifts_long,
    solve_network,
    warehouses_flat,
):
    tech_results = {'base': solve_network(), 'autonomous_no_crew': solve_network(shift_cost=shift_cost_base * 0.4), 'autonomous_full': solve_network(shift_cost=shift_cost_base * 0.4, shift_min=900, travel_factor=0.8)}
    tech_df = pd.DataFrame({name: {'cost': r['total_cost'], 'n_open': r['n_open']} for name, r in tech_results.items()}).T
    display(tech_df)
    usable_full_tech = 900 - 2 * shifts_long['travelTime'] * 0.8
    n_feasible_tech = (usable_full_tech > 0).sum()
    print(f'Feasible links in autonomous_full scenario: {n_feasible_tech} (vs. 2,661 in the base case)\n')
    for name, r in tech_results.items():
        _tiers = warehouses_flat.set_index('warehouseID').loc[r['open_centers'], 'tier'].value_counts()
        print(f"{name}: n_open={r['n_open']}, tiers={_tiers.to_dict()}")
    return (tech_results,)


@app.cell
def _(pd, regions_flat, tech_results, warehouses_flat):
    volume_per_center_tech = (
        pd.DataFrame(tech_results["autonomous_full"]["assignment"], columns=["warehouseID", "regionID"])
        .merge(regions_flat[["regionID", "yearlyDemand"]], on="regionID")
        .groupby("warehouseID")["yearlyDemand"].sum()
    )
    volume_per_center_tech.reset_index().merge(warehouses_flat[["warehouseID", "city", "tier"]], on="warehouseID").sort_values("tier")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The "no crew" scenario (15 centers, approximately -35% cost) represents a moderate and plausible shift, consistent with the direction observed in other cost-based scenarios. The "full automation" scenario, however, leads to a fundamentally different result. The number of feasible links increases sharply (11,548 vs. 2,661), as 900 minutes of usable shift time combined with faster travel speeds makes nearly every region accessible from almost every center.

    At this point, neither transport costs nor processing cost differences meaningfully constrain the model. Instead, the solution is primarily driven by fixed costs, resulting in a collapse toward a minimal set of four centers with the lowest combined fixed cost. Given that no volume caps are imposed in the model, this allows a small number of facilities to absorb the entire national demand.

    Concretely, in this scenario two "s"-tier centers alone, Valladolid (1,366,181 deliveries per year) and Tarragona (892,083), would together handle almost the entire annual demand of Spain at volumes far below what would be operationally realistic for such facilities. This outcome is therefore not an operational recommendation, but a direct consequence of combining near-universal reachability with an unconstrained capacity formulation. A realistic evaluation of such a "hub reduction" strategy would require explicit capacity limits, which are not available in the current dataset.

    Accordingly, the four-center solution is not interpreted as a feasible recommendation. However, the direction of the effect remains informative. The underlying trade-off between fixed facility costs and transport costs becomes substantially weaker once distance ceases to be a binding constraint. While autonomous vehicles and drone technologies remain speculative, the results suggest that significant reductions in effective distance costs could materially shift the economics of maintaining a dense network of small "v"-tier centers, making selective consolidation into larger hubs more attractive than in the current setting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 Overlap Comparison

    Before finalizing the recommendation, one additional robustness question is addressed. The decision not to impose volume caps results in a network that is structurally dominated by small centers. This pattern remains consistent across the sensitivity analyses, as none of the economic scenarios introduce explicit capacity constraints. To further validate the robustness of the resulting network structure, a final cross-check is conducted.

    The set of open centers in the capacity-constrained (capped) model is compared with the centers favored under independent economic stress scenarios, including stronger economies of scale, higher processing-cost anchors, and rising transport costs. This comparison is conducted deliberately against the scenario-based results rather than the uncapped baseline, since the latter differs by construction due to the absence of capacity restrictions.

    If the capped solution shows substantial overlap with the centers repeatedly selected under these independent parameter variations, this provides evidence that the resulting network structure is not merely an artifact of a specific capacity threshold, and should therefore be weighted more heavily in the final recommendation.
    """)
    return


@app.cell
def _(
    alpha_results,
    anchor_results,
    open_centers_capped,
    pd,
    shiftcost_results,
):
    capped_open = set(open_centers_capped)
    alpha_1_open = set(alpha_results[1.0]["open_centers"])
    anchor_25_open = set(anchor_results[0.25]["open_centers"])
    shiftcost_15_open = set(shiftcost_results[1.5]["open_centers"])

    overlap_df = pd.DataFrame([
        {"scenario": "alpha=1.0",     "overlap": len(capped_open & alpha_1_open),     "of_capped": len(capped_open)},
        {"scenario": "anchor=0.25",   "overlap": len(capped_open & anchor_25_open),   "of_capped": len(capped_open)},
        {"scenario": "shiftcost=1.5", "overlap": len(capped_open & shiftcost_15_open),"of_capped": len(capped_open)},
    ])
    overlap_df["pct"] = (overlap_df["overlap"] / overlap_df["of_capped"] * 100).round(1)
    overlap_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results confirm strong and consistent overlap between the capped network and the centers favored under independent economic stress scenarios. Of the 19 centers open in the capped solution, 13 (68.4%) are also open under both $\alpha = 1.0$ and anchor $= 0.25$, while 12 (63.2%) remain open under shiftcost $= 1.5$.

    This consistency across three independently derived scenarios indicates that the capacity-constrained network is not merely an artifact of the specific 1.5x-percentile threshold chosen, but reflects a structurally stable core that also emerges under entirely different economic pressures. The capped solution can therefore be weighted more heavily in the final recommendation, rather than treated as an isolated sensitivity check.

    ### 6.6 Volume Redistribution Under Capacity Constraints

    As a final check, the redistribution of volume within the capped model is examined more closely. This step was not part of the earlier capacity comparison, where the capped and uncapped solutions were contrasted only in terms of total cost, network size, and tier composition. Here, the focus shifts to how volume is distributed across centers once caps are introduced, providing a clearer picture of whether the capped network relies on a small number of facilities operating near their limits or spreads volume more evenly across the network.
    """)
    return


@app.cell
def _(link_keys, open_centers_capped, pd, regions_flat, warehouses_flat, x_c):
    volume_check_capped = pd.DataFrame([(_i, _j) for _i, _j in link_keys if x_c[_i, _j].value() == 1], columns=['warehouseID', 'regionID']).merge(regions_flat[['regionID', 'yearlyDemand']], on='regionID').groupby('warehouseID')['yearlyDemand'].sum()
    warehouses_flat.set_index('warehouseID').loc[open_centers_capped, ['city', 'tier']].join(volume_check_capped).sort_values('yearlyDemand', ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The disagreements are informative. Madrid appears only in the capped network, consistent with two earlier findings: (a) economic stress scenarios consolidate into "s"/"m" tiers but never reach for a full "h" tier facility on their own, and (b) the optimizer will assign a very high volume to a small warehouse if capacity bounds are not accounted for — only a hard capacity constraint forces the network that far. Conversely, Pontevedra, Toledo, and Sevilla are favored under economic stress but not needed once a hard cap is imposed; the capped model instead spreads the corresponding demand across more, smaller centers (including Almeria, Cordoba, Santander, and Lerida) rather than consolidating it into a medium-sized hub.

    After checking the capped solution itself for new outliers, Madrid (now "h" tier) carries 826,543 units — even more than Guadalajara's original 716,917. Unlike Guadalajara, however, this figure is plausible: Madrid's fixed cost (€35.9M) reflects the size class with the highest fixed-cost level in the tier structure, and 826,543 lies well within a plausible volume range for that tier. The second-largest facility, Alicante ("m", 245,965), and the remaining "s"/"v" centers all fall within plausible ranges for their respective tiers — the capped model does not introduce a new hidden-outlier problem.

    This provides confidence that the capped network is not an artifact of one arbitrary threshold choice: a large majority of its structure is independently corroborated by scenarios that have no knowledge of capacity limits. Notably, this also resolves the concern raised earlier: whereas Guadalajara's oversized volume sat implausibly within the smallest tier, Madrid's comparably large volume here sits within the tier explicitly built for the largest facilities. The capped model is therefore weighted heavily in the final recommendation, while the handful of centers that appear in one view but not the other are treated as points warranting closer, case-by-case review rather than settled conclusions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Robust Recommendation

    All scenarios are synthesized into a single recommendation. For each center, it is recorded whether it was open across every scenario solved. While the sensitivity analysis in Section 6 was framed around a three-way distinction between consistently selected, consistently closed, and scenario-dependent facilities, the results reveal that the "scenario-dependent" category itself contains meaningfully different situations: some centers are contested because they are favored only under capacity constraints, others only under economic stress, and still others under both simultaneously. The classification is therefore refined into a more granular set of categories:

    - **Keep** — open in the base case and in all or all but one scenario, including the capped model. Safe regardless of how the future unfolds.
    - **Review** — open in the base case but unpredictable across multiple scenarios. Worth monitoring, not an immediate action.
    - **Reconsider** — open in the base case but closes under multiple independent scenarios, or is the specific center flagged by the capacity check (Guadalajara). The strongest candidates for closer attention.
    - **Confirmed closed** — closed in the base case and in virtually every scenario tested. No evidence supports opening these centers under any considered condition.
    - **Expansion candidate (capacity + economic — strongest signal)** — closed in the base case but opens under both the capacity-constrained model and independent economic stress scenarios. The most strongly supported candidates for future expansion.
    - **Expansion candidate (capacity-driven)** — closed in the base case but opens only once a hard capacity constraint is imposed, never under any purely economic scenario.
    - **Expansion candidate (economic stress only)** — closed in the base case but opens only under economic stress scenarios, not under the capacity-constrained model.
    - **Negligible** — appears in only a single scenario, with no clear or consistent signal in either direction.

    ### A methodological note on Madrid and similar cases

    Madrid and similar centers appear open only in the capacity-constrained model, never under any of the 11 purely economic scenarios. This is not a weak signal — if anything, it deserves more weight than the economic-stress signals discussed above, not less. None of the 11 economic scenarios impose any capacity limit, so they are structurally incapable of ever favoring a large facility: an uncapped small center is always cheaper on paper, regardless of how extreme the economic parameters become. Only the capacity-constrained model actually tests the question these centers answer.

    As argued earlier, the capped model is the best available proxy for real-world capacity limits — the one dimension of the analysis with a genuine, data-grounded reason to be trusted over the uncapped alternative. A center favored specifically by that model should therefore be read as a stronger, not weaker, signal than one that merely appears across several economically stressed but still uncapped scenarios.
    """)
    return


@app.cell
def _(
    all_warehouse_ids,
    alpha_results,
    anchor_results,
    demand_results,
    open_centers_capped,
    pd,
    shiftcost_results,
    solve_network,
    warehouses_flat,
):
    # Alle gelösten Szenarien in einem Dict sammeln (nutzt eure bereits gespeicherten Ergebnisse)
    scenarios = {
        "base":            solve_network(),                                  
        "alpha_0.3":       alpha_results[0.3],
        "alpha_0.7":       alpha_results[0.7],
        "alpha_1.0":       alpha_results[1.0],
        "anchor_0.10":     anchor_results[0.10],
        "anchor_0.25":     anchor_results[0.25],
        "demand_0.9":      demand_results[0.9],
        "demand_0.7":      demand_results[0.7],
        "demand_0.5":      demand_results[0.5],
        "shiftcost_1.2":   shiftcost_results[1.2],
        "shiftcost_1.5":   shiftcost_results[1.5],
        "capped":          {"open_centers": open_centers_capped},
    }

    # Für jedes Center: in wie vielen Szenarien war es offen - wichtig econ und cap trennen
    econ_scenario_names = [n for n in scenarios if n != "capped"]

    rows = []
    for wid in all_warehouse_ids:
        open_in = {name: (wid in s["open_centers"]) for name, s in scenarios.items()}
        n_open_econ = sum(open_in[n] for n in econ_scenario_names)
        rows.append({
            "warehouseID": wid,
            "n_open_econ": n_open_econ,
            "n_total_econ": len(econ_scenario_names),
            "in_base": open_in["base"],
            "in_capped": open_in["capped"],
        })

    status_df = (
        pd.DataFrame(rows)
        .merge(warehouses_flat[["warehouseID", "city", "tier"]], on="warehouseID")
    )

    status_df[["city", "tier", "n_open_econ", "n_total_econ", "in_base", "in_capped"]] #df mit übersicht über alle scenarien
    return (status_df,)


@app.cell
def _(status_df):
    def classify(row):
        if row["in_base"]:
            if row["in_capped"]:
                if row["n_open_econ"] >= row["n_total_econ"] - 1:
                    return "Keep"
                else:
                    return "Review"  # still in capped network, but less robust economically
            else:
                return "Reconsider"  # capacity check drops it, regardless of economic count
        else:
            if row["in_capped"] and row["n_open_econ"] >= 2:
                return "Expansion candidate (capacity + economic — strongest signal)"
            elif row["in_capped"]:
                return "Expansion candidate (capacity-driven)"
            elif row["n_open_econ"] >= 2:
                return "Expansion candidate (economic stress only)"
            elif row["n_open_econ"] == 0:
                return "Confirmed closed"
            else:
                return "Negligible (single scenario, no clear signal)"

    status_df["category"] = status_df.apply(classify, axis=1)
    status_df.sort_values(["category", "n_open_econ"], ascending=[True, False])[["city", "tier", "category", "n_open_econ", "in_base", "in_capped"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the Classification

    **Keep — the stable core (10 centers).** Vitoria, Alicante, Almeria, Caceres, Gerona, Huesca, Jaen, Leon, Tarragona, and Cuenca remain open in 10 or 11 of the 11 economic scenarios and are also selected by the capacity-constrained model. These centers form the robust backbone of the recommended network and remain stable under virtually every future tested.

    **Review — demand-sensitive (5 centers).** Lugo, Teruel, Cordoba, Huelva, and Orense remain open in both the base case and the capacity-constrained model, but drop out under several independent economic stress scenarios (7 to 9 of 11). They are part of the recommended network but are not as unconditionally robust as the "Keep" group, and should be monitored if economic conditions shift.

    **Reconsider — potential consolidation candidates (3 centers).** Burgos, Avila, and Guadalajara are part of today's network but are not selected by the capacity-constrained model, despite remaining open under most purely economic scenarios. Burgos is a particularly clear illustration of why the capacity check takes precedence: it shows one of the highest economic-scenario counts overall (9 of 11) — higher, in fact, than several centers classified as "Review" — yet it is still not selected once realistic capacity constraints are imposed. Guadalajara remains especially noteworthy, since the earlier volume analysis identified it as carrying an implausibly high throughput in the unconstrained solution. These locations are the strongest candidates for network consolidation or operational review.

    **Confirmed closed (15 centers).** Barcelona, Ciudad Real, La Coruña, Granada, San Sebastian, La Rioja, Navarra, Oviedo, Palencia, Salamanca, Segovia, Soria, Valladolid, Bilbao, and Zaragoza never appear in any recommended network, neither across the 11 economic scenarios nor under the capacity-constrained model. This provides the strongest possible evidence that these facilities are not required in the future network. Barcelona is especially notable: despite being a large ("l") tier center, it never becomes attractive, whereas Madrid (tier "h") does once realistic capacity limits are introduced.

    **Expansion candidate — capacity + economic (1 center).** Zamora is the strongest expansion signal in the entire analysis. It is selected by the capacity-constrained model and also appears in five of the eleven economic scenarios, making it a location independently supported by both modeling approaches.

    **Expansion candidate — capacity-driven (3 centers).** Madrid, Lerida, and Santander appear almost exclusively once realistic capacity constraints are introduced (Santander appears in one economic scenario as well). Although they receive little support from the purely economic scenarios, this reflects the fact that those models assume unlimited facility capacity. Since capacity realism is considered the preferred planning perspective in this analysis, these locations deserve serious consideration.

    **Expansion candidate — economic stress only (4 centers).** Pontevedra and Sevilla receive the strongest support within this group, each appearing in four economic scenarios, followed by Toledo (three) and Valencia (two). Because none are confirmed by the capacity-constrained model, these represent secondary expansion opportunities rather than immediate recommendations.

    **Negligible (1 center).** Albacete appears in only a single economic scenario and is not supported by the capacity-constrained model. No practical conclusion is drawn from this isolated result.

    ### Overall Recommendation

    The main conclusion of this study is not simply which facilities should remain open, but that the unconstrained cost-minimizing network systematically relies on an unrealistic assumption of unlimited facility capacity. Removing the overloaded Guadalajara facility in isolation changes total cost only marginally, since the model reallocates the displaced volume to another small center rather than restructuring the network. Introducing realistic capacity limits, by contrast, yields a substantially different configuration, demonstrating that capacity assumptions fundamentally shape the optimal solution.

    We therefore recommend adopting the capacity-constrained network as the primary implementation plan. This recommendation is reinforced by the robustness analysis: ten facilities remain stable across virtually all scenarios, a further five are retained but show greater sensitivity to economic conditions, and fifteen are consistently rejected. The remaining twelve locations require managerial judgment: three are candidates for consolidation, eight are candidates for expansion under varying degrees of support, and one shows no clear signal in either direction. Among the potential expansions, Zamora emerges as the strongest candidate, as it is independently supported by both the economic scenario analysis and the capacity-constrained optimization.
    """)
    return


@app.cell
def _(folium, status_df, warehouses_flat):
    # Mittelpunkt der Karte
    center_lat = warehouses_flat["lat"].mean()
    center_lon = warehouses_flat["lon"].mean()

    m_final = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="cartodbpositron"
    )

    # Farben der finalen Klassifikation
    category_colors = {
        "Keep": "#2ca02c",                                         # green
        "Review": "#f1c40f",                                       # yellow
        "Reconsider": "#ff7f0e",                                   # orange
        "Confirmed closed": "#d62728",                             # red
        "Expansion candidate (capacity + economic — strongest signal)": "#1f77b4",
        "Expansion candidate (capacity-driven)": "#17becf",
        "Expansion candidate (economic stress only)": "#9467bd",
        "Negligible (single scenario, no clear signal)": "#7f7f7f"
    }

    plot_df = warehouses_flat.merge(
        status_df[["city","category","n_open_econ","in_base","in_capped"]],
        on="city"
    )

    for row in plot_df.itertuples():

        color = category_colors[row.category]

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1.5,
            popup=f"""
            <b>{row.city}</b><br>
            Tier: {row.tier}<br>
            Category: {row.category}<br>
            Economic scenarios: {row.n_open_econ}/11<br>
            Base network: {row.in_base}<br>
            Capacity network: {row.in_capped}
            """
        ).add_to(m_final)

    # Legende # ausklappbar
    legend_html = """
    <div style="
    position: absolute;
    top: 12px;
    right: 12px;
    z-index:9999;
    background:white;
    border:2px solid grey;
    border-radius:6px;
    font-size:13px;
    max-width:300px;
    ">
    <details>
    <summary style="padding:8px 12px; cursor:pointer; font-weight:bold;">
    Final network classification
    </summary>
    <div style="padding:0 12px 12px 12px;">
    <span style="color:#2ca02c;">&#9679;</span> Keep<br>
    <span style="color:#f1c40f;">&#9679;</span> Review<br>
    <span style="color:#ff7f0e;">&#9679;</span> Reconsider<br>
    <span style="color:#d62728;">&#9679;</span> Confirmed closed<br>
    <span style="color:#1f77b4;">&#9679;</span> Expansion (capacity + economic)<br>
    <span style="color:#17becf;">&#9679;</span> Expansion (capacity-driven)<br>
    <span style="color:#9467bd;">&#9679;</span> Expansion (economic only)<br>
    <span style="color:#7f7f7f;">&#9679;</span> Negligible
    </div>
    </details>
    </div>
    """

    m_final.get_root().html.add_child(folium.Element(legend_html))

    m_final
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Review & Limitations

    ### Review of Model Development and Key Findings

    This study developed a location–allocation optimization model for CashLog's Spanish cash center network, starting from a transportation-cost structure that was not predefined and therefore had to be derived endogenously from the available data. After constructing and validating a distance-based cost function — which closely matched the course benchmark — transport costs were found to increase non-linearly as assignments approach their feasibility limits, confirming the importance of spatial realism in the network structure. During validation, several zero-travel-time pairs with heterogeneous costs were identified; these were traced back to consistent data patterns rather than data errors, supporting the reliability of the underlying dataset.

    The model was then extended by introducing size-dependent processing costs based on five empirically derived facility tiers. Economies of scale were modeled using a dampened power function, calibrated through a combination of fixed-cost ratios and an externally anchored baseline processing cost. This introduced several necessary but assumption-driven design choices — including the scaling exponent, the functional form, and the cost anchor — all of which were subsequently tested in the sensitivity analysis.

    A central design decision was to use fixed costs as a proxy for facility size and to exclude explicit capacity bounds in the primary model. This allowed a clean keep-or-close formulation but introduced the risk of unrealistically large allocations to small facilities, most notably Guadalajara. To address this, a separate capacity-constrained comparison model was constructed, which materially altered the network and ultimately served as the basis for the final recommendation.

    Robustness checks confirmed that the main structural results are stable across a wide range of parameter settings (economies of scale, processing-cost anchors, demand reductions, wage and fuel shocks, and technological changes). A stable core of ten facilities remained open across virtually all scenarios, forming the backbone of the final recommendation. Notably, the analysis also revealed that different types of stress affect different parts of the network: cost-structure variations (economies of scale, processing-cost anchor) consistently pressured the same five southwestern centers, whereas demand reductions and rising transport costs affected entirely different facilities. This indicates that no single group of "marginal" centers exists; rather, vulnerability depends on the specific type of structural change considered.

    ### Key Structural Limitations

    Despite the extensive robustness analysis, several limitations remain that should be considered when interpreting the results.

    First, the transportation cost function had to be derived entirely from the available data, since the case study did not specify a cost model. Although the formulation closely matched the course benchmark and produced plausible cost patterns, alternative cost functions could lead to different network designs.

    Second, the processing-cost model relies on several assumptions. Facility size was inferred from fixed costs, economies of scale were represented by a dampened power function, and the initial processing-cost anchor was derived as 15% of the median transportation cost. While the two most influential parameters (the scaling exponent and the processing-cost anchor) were explicitly tested in the sensitivity analysis, the underlying functional form remains an assumption rather than an empirically validated relationship.

    Third, the primary optimization model deliberately assumes unlimited processing capacity at each cash center and treats center size as fixed rather than as a decision variable. This is a conscious methodological choice rather than a convenience: both explicit capacity caps and endogenous facility resizing would each require a cascade of parameters that the available data simply does not support, and inventing them would introduce more arbitrariness than insight.

    Concretely, imposing hard volume caps would require a defensible capacity limit for every tier — yet the dataset contains no observed throughput ceilings, only fixed costs. Any cap would therefore be a number chosen by us rather than grounded in the data, and as the capacity analysis showed, the model's feasibility and structure are highly sensitive to exactly where that cap is set (the course's own official bounds even rendered the model infeasible for a single region). A fabricated cap would not be more trustworthy than the cost logic it overrides.

    Allowing centers to be upgraded to a larger tier would compound the problem further. Such a formulation would require, at minimum: an upgrade-cost factor for every tier-to-tier transition (how much more expensive does a "v" center become when upgraded to "s", "m", and so on), whether these costs are linear or scale non-linearly across jumps, one-time transition or construction costs, and possibly time-phased investment budgets. None of these quantities exist in the dataset. Each would have to be assumed, and the final network would then be driven as much by our invented upgrade economics as by the actual cost data.

    For these reasons, the model is restricted to keep-or-close decisions based on historically observed cost tiers, which are fully grounded in the data. However, this simplification results in solutions that assign unrealistically high volumes to some small facilities, most notably Guadalajara. A separate capacity-constrained comparison model was therefore developed as a bounded sensitivity check — deliberately kept as a comparison rather than the primary model, precisely because its own capacity threshold is itself an assumption (see the fourth limitation below). It substantially changed the recommended network and ultimately became the basis for the final recommendation. Consequently, the uncapped model should be interpreted primarily as an economic benchmark rather than as an operational implementation plan.

    Fourth, the capacity-constrained comparison model itself introduces uncertainty, since its volume limits were derived from the uncapped solution rather than from observed operational capacities. The chosen threshold (1.5 × the 75th percentile, after excluding the identified outlier) serves as a pragmatic calibration rather than a validated engineering limit. While the comparison successfully demonstrates the structural importance of capacity constraints, the exact capacities of individual facilities should be verified before implementation.

    Fifth, the scenario analysis intentionally focuses on a limited number of business drivers: economies of scale, processing-cost assumptions, declining cash demand, rising operating costs, and technological change. Other potentially relevant factors — including regional demand growth, investment costs for facility expansion, relocation costs, workforce availability, regulatory changes, and stochastic disruptions — were outside the scope of this study. Incorporating these aspects would provide an even more comprehensive assessment of long-term network robustness.

    Sixth, a constant time per stop is assumed in the cost formulation. Specifically, stops per shift are modeled as scaling strictly proportionally with usable time, treating minutesPerStop as a constant divisor. In reality, tour efficiency is unlikely to be perfectly linear, as routing density typically improves with a higher number of stops per tour. The simpler proportional specification is nevertheless adopted because minutesPerStop in the data already reflects an empirically averaged, region-specific productivity measure, making a constant-slope approximation a reasonable first-order assumption. A more refined approach would estimate region-specific, potentially non-linear productivity functions from observed tour data, where available.
    """)
    return


if __name__ == "__main__":
    app.run()
