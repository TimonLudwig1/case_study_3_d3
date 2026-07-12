# Migration Plan: Making Center Size a Decision Variable

This document describes, step by step, how to convert `case_study_marimo.py` from the current
**frozen-size model** (each center's tier is fixed by its historical fixed cost) to the
**size-as-decision model** the lecture intended (the solver chooses which size to build at each site).

---

## 1. What changes conceptually

### Current model
- Tier of center *i* is read off its fixed cost and **frozen**: `tier(i)` is a parameter.
- Decision variables: `y_i` (open/closed), `x_ij` (assignment).
- Capacity is either ignored (base model) or bolted on afterwards (percentile caps, official bounds).
- Consequence: the official volume bounds are **infeasible** (region 410, demand 59,646, only
  reaches v/s-tier centers, max cap 45,415).

### New model
- Every site can be built at **any of the five tiers** (or closed). The five observed fixed-cost
  values are reinterpreted as a **price list** `f_t` — justified by the data: fixed costs are
  *exactly* tier-constant (all 19 v-tier centers cost exactly €1,344,000; zero within-tier
  variation), so fixed cost is a property of the size, not the location.
- The official volume bounds `[lb_t, ub_t]` become **core model parameters**: they are contiguous,
  non-overlapping bands that partition the volume axis, i.e. the volume a center processes
  determines which size it must be (and therefore which `f_t` and `c_t^var` it pays).
- The region-410 infeasibility disappears by construction: the solver can now build Zaragoza (or
  Lerida) at tier m (cap 107,327 ≥ 59,646).

### The one assumption the new model makes (state it explicitly)
Re-sizing is free: a site pays only the annualized fixed cost of its *chosen* tier. There is no
one-time conversion/construction capex and no constraint that ties a site to its historical size.
The model therefore answers "what network would we design" rather than "what should we convert,
given conversion is not free." This replaces the current Section 8 argument that size-as-decision
is impossible — it is not impossible, it just assumes transition costs away. (See §8 below for an
optional transition-cost extension.)

---

## 2. The new mathematical model

**Sets:** sites *i* (42), regions *j* (515, reachable pairs only), tiers *t ∈ {v, s, m, l, h}*.

**Parameters:**

| Symbol | Meaning | Source |
|---|---|---|
| `c_ij` | annual transport cost | unchanged (Section 4, validated) |
| `d_j` | annual demand | unchanged |
| `f_t` | annual fixed cost of a tier-*t* facility | the 5 observed fixed-cost values (price list) |
| `lb_t, ub_t` | volume band of tier *t* | `OFFICIAL_V_BOUNDS` (already in the notebook) |
| `c_t^var` | processing cost per delivery at tier *t* | unchanged (Section 5.3 derivation) |

**Decision variables:**

- `x_ij ∈ {0,1}` — region *j* served by site *i* (reachable pairs only, as before)
- `y_it ∈ {0,1}` — site *i* is built at tier *t* (**new**, replaces `y_i`)
- `v_it ≥ 0` — volume processed at site *i* if built at tier *t* (**new**, continuous helper)

**Objective:**

```
min   Σ_ij c_ij · x_ij            (transport — no longer folded into an "effective cost")
    + Σ_i Σ_t f_t · y_it          (fixed cost of the chosen size)
    + Σ_i Σ_t c_t^var · v_it      (size-dependent processing cost)
```

**Constraints:**

```
Σ_i x_ij = 1                       ∀ j            (every region served, reachable links only)
x_ij ≤ Σ_t y_it                    ∀ (i,j)        (only assign to a site that is built)
Σ_t y_it ≤ 1                       ∀ i            (at most one size per site; 0 = closed)
Σ_t v_it = Σ_j d_j · x_ij          ∀ i            (volume bookkeeping)
lb_t · y_it ≤ v_it ≤ ub_t · y_it   ∀ i, t         (volume must lie in the chosen tier's band)
```

**Why this is correct and still linear:** `Σ_t y_it ≤ 1` forces `v_it = 0` for every non-chosen
tier, so exactly one band is active per open site. The solver cannot "buy" a big tier's cheap
`c_t^var` for a small volume because the band's lower bound `lb_t` forbids it, and cannot overload
a small tier because of `ub_t`. Since the bands partition the volume axis, the chosen tier is
effectively *determined* by the volume — which is exactly the lecture's "volume and size are the
decision" framing. The old `effective_cost = c_ij + c_tier(i)^var · d_j` linearization trick no
longer works (the tier is no longer known in advance), which is why processing cost moves into
the separate `v_it` term.

**Model size:** 2,661 binary `x` + 210 binary `y` + 210 continuous `v` — trivially solvable by CBC.

---

## 3. What stays completely unchanged

| Notebook part | Lines (current file) | Why untouched |
|---|---|---|
| Sections 1, 3, 4 (problem, data, transport-cost derivation + benchmark validation) | 1–527 | `c_ij` is independent of tier |
| Reachability analysis and link dropping | 271–357 | unchanged |
| Tier *identification* (gap analysis, 5 exact values, tier mapping) | 557–652 | the *finding* is reused; only its interpretation changes (see Step 2) |
| Derivation of `c_t^var` (scaling, dampening α, anchor) | 659–804 | the per-tier values are reused as-is |

---

## 4. Step-by-step migration

### Step 1 — Rewrite the framing of "limitation 3" (Section 2.3, lines ~96–98)

Currently limitation 3 says the fixed cost `f_i` is a single constant and the fix is a
volume-dependent *processing* cost. Strengthen it: the deeper flaw is that the baseline model
treats **size as an attribute instead of a decision**. State that the extension will let the model
choose each site's size, with fixed cost, capacity band, and processing cost all following from
that choice.

### Step 2 — Reinterpret Section 5.2 (tiers) as a price list; move the official bounds here

- Keep the gap analysis and the five-value finding (lines 557–652) verbatim — it is now even more
  load-bearing: zero within-tier variation is the *evidence* that fixed cost is a size price list
  `f_t`, not a location attribute. Add one paragraph making this argument explicit (it currently
  says the opposite — that Madrid/Barcelona prices reflect "regional price levels"; soften the
  Section 3 data commentary, line ~225, accordingly).
- **Move `OFFICIAL_V_BOUNDS`** (currently defined at line 1175 inside the 5.4 sensitivity block)
  up into Section 5.2 as a core parameter table, introduced as "the course's capacity bands per
  tier." Point out that the bands are contiguous and partition the volume axis.
- Keep `warehouses_flat["tier"]` (line 651) but **rename it conceptually to "current tier"** —
  it is no longer a model input, but it is needed later to report upgrades vs. downgrades.

### Step 3 — Keep Section 5.3 (`c_t^var`) unchanged

No edits needed. The α-dampened, anchor-based derivation produces the per-tier `c_t^var` menu the
new model consumes directly.

### Step 4 — Replace the "complete extended model" and the base solve

- Rewrite the model-statement markdown cell (lines 807–857) with the formulation from §2 above.
  The follow-up cell explaining the `effective_cost` substitution (lines 844–857) is **deleted** —
  that linearization is specific to a known tier.
- The `model_data` prep cell (lines 861–875): drop `c_var_i` / `effective_cost` columns; you now
  only need `transportationCosts`, `yearlyDemand` per link.
- Replace the base PuLP cell (lines 879–923) with the new formulation. Skeleton:

```python
TIERS = ["v", "s", "m", "l", "h"]
F_T = {"v": 1_344_000, "s": 2_580_000, "m": 4_572_000, "l": 15_012_000, "h": 35_904_000}

prob = pl.LpProblem("CashLog_SizeDecision", pl.LpMinimize)
x = pl.LpVariable.dicts("x", link_keys, cat="Binary")
y = pl.LpVariable.dicts("y", [(i, t) for i in all_warehouse_ids for t in TIERS], cat="Binary")
v = pl.LpVariable.dicts("v", [(i, t) for i in all_warehouse_ids for t in TIERS], lowBound=0)

prob += (
    pl.lpSum(row.transportationCosts * x[(row.warehouseID, row.regionID)]
             for row in model_data.itertuples())
  + pl.lpSum(F_T[t] * y[(i, t)] for i in all_warehouse_ids for t in TIERS)
  + pl.lpSum(c_var_by_tier[t] * v[(i, t)] for i in all_warehouse_ids for t in TIERS)
)

for i, j in link_keys:
    prob += x[(i, j)] <= pl.lpSum(y[(i, t)] for t in TIERS)
for j in regions_flat["regionID"]:
    reach = [i for (i, jj) in link_keys if jj == j]
    prob += pl.lpSum(x[(i, j)] for i in reach) == 1
for i in all_warehouse_ids:
    prob += pl.lpSum(y[(i, t)] for t in TIERS) <= 1
    links_i = [(ii, j) for (ii, j) in link_keys if ii == i]
    vol_i = pl.lpSum(demand_by_link[(i, j)] * x[(i, j)] for (ii, j) in links_i)
    prob += pl.lpSum(v[(i, t)] for t in TIERS) == vol_i
    for t in TIERS:
        lb, ub = OFFICIAL_V_BOUNDS[t]
        prob += v[(i, t)] >= lb * y[(i, t)]
        prob += v[(i, t)] <= ub * y[(i, t)]
```

- Result extraction changes: a site is open iff `Σ_t y[(i,t)].value() == 1`; its **chosen tier**
  is the `t` with `y[(i,t)] == 1`. Build `open_centers` and a new `chosen_tier_by_wh` dict from
  this. All downstream tables/maps that show "tier" must now distinguish **chosen tier** vs.
  **current tier**.

### Step 5 — Repurpose the Guadalajara / no-cap analysis as *motivation*, don't delete it

The old frozen-size uncapped solve and the Guadalajara volume check (lines 1074–1139) tell the
story of *why* size must be a decision. Two options:

- **Recommended:** keep a compact version *before* the new model as a motivating subsection
  ("what goes wrong when sizes are frozen and capacity is ignored: Guadalajara at 716,917 units in
  a v-tier shell"), then present the new model as the fix.
- Alternative: delete it entirely and let the region-410 infeasibility carry the motivation alone.
  This loses a good narrative beat; not recommended.

### Step 6 — Delete Section 5.4 (percentile caps) entirely

Lines 1143–1375 (the 1.5×-75th-percentile cap derivation, `prob_capped`, its result cells, and the
uncapped-vs-capped comparison table). This whole construction existed only because the frozen-size
model had no defensible capacity numbers. The official bands now provide real capacities inside
the main model, so the percentile cap is obsolete — including:

- the `V_ub_by_tier` cell (lines 1249–1274) and its "m-tier single observation" caveat (1277–1282),
- `prob_capped` / `x_c` / `y_c` (lines 1285–1337) and the capped-open-centers cell (1340–1353),
- the capped-vs-uncapped comparison markdown (1356–1376).

**Caution:** `x_c`, `y_c`, `open_centers_capped` are consumed later by Sections 6.5, 6.6, and 7 —
those consumers are deleted/reworked in Steps 9–10, so delete in that order (marimo will flag the
dangling references otherwise).

### Step 7 — Transform Section 5.5 (official bounds) from "dead end" into "the punchline"

Keep the infeasibility run and the region-410 diagnosis (lines 1379–1583) — it is your strongest
analytical finding — but reframe the conclusion (lines 1574–1583):

- Old claim: "the official bounds cannot be applied to this dataset."
- New claim: "the official bounds are infeasible **only when sizes are frozen at their historical
  tiers** (region 410 reaches only v/s sites, max cap 45,415 < 59,646). Under the size-decision
  model the solver upgrades a reachable site (e.g. Zaragoza → m) and the bounds are satisfied —
  evidence that the bounds were *designed* for the size-decision model."
- Then show the new model's solution status (Optimal) directly beneath, closing the arc.
- The `solve_network_debug` cell (lines 1439–1531) can be deleted; its job (isolating lb vs. ub as
  the cause) is done and the finding is preserved in prose plus the compact per-region certificate
  cell (lines 1544–1563), which should stay.

### Step 8 — Rewrite `solve_network` (currently lines 1159–1246)

Relocate it right after the new base model (it currently lives inside the deleted 5.4 block) and
rebuild it around the new formulation. Signature stays almost identical so all sweep cells keep
working:

```python
def solve_network(alpha=alpha, demand_factor=1.0, shift_cost=shift_cost_base,
                  shift_min=shift_min, travel_factor=1.0,
                  processing_share=processing_share,
                  force_open=None, force_closed=None):
```

- Internals: same recomputation of `c_var` and link costs, then the §2 formulation instead of the
  old one. `use_official_bounds` disappears as a flag — the bounds are always part of the model.
- `force_open` now means `Σ_t y_it == 1`; `force_closed` means `Σ_t y_it == 0`.
- Return value: add `"chosen_tier": {i: t}` alongside `open_centers`, `n_open`, `assignment`,
  `total_cost`, `status`.
- Optional but valuable: a `freeze_sizes=False` parameter that, when `True`, adds
  `y_it = 0 ∀ t ≠ current_tier(i)` — this reproduces the frozen-size world inside the same
  function and lets you show the feasible/infeasible contrast with one flag.

### Step 9 — Re-run Section 6 (sensitivity) and rewrite its prose

The sweep *code* (α sweep, anchor sweep, demand slider, shift-cost slider, tech scenarios — lines
1598–2183) keeps working unchanged once `solve_network` is swapped, **but every number and every
named center in the interpretation prose is stale** and must be rewritten from the new outputs.
Expect qualitatively different results:

- The tech "full automation" collapse to 4 tiny centers absorbing all of Spain (lines 2172–2183)
  **cannot happen anymore** — capacity bands now bind. That whole passage must be rewritten; the
  new result will itself be an interesting finding (automation + capacity limits).
- Add one new dimension to every scenario readout: the **tier profile** (how many v/s/m/l/h are
  built) and **upgrades/downgrades vs. current tiers**.
- Delete Section 6.5 (overlap with the capped model, lines 2186–2233) and Section 6.6 (capped
  volume redistribution, lines 2235–2258) — both compare against the deleted percentile-cap model.
  Replace with a single new subsection: *volume and tier utilization in the base solution*
  (volume per site vs. its band, which sites sit near their `ub`, which sites got upgraded).

### Step 10 — Rework Section 7 (robust recommendation, lines 2261–2495)

- The scenario dict (lines 2299–2312) loses the `"capped"` entry; the `classify()` function
  (lines 2342–2361) loses all `in_capped` branches. Classification becomes: open-frequency across
  the (now capacity-aware) scenarios.
- **Delete** the "methodological note on Madrid" (lines 2277–2282) — it argued that only the
  capped model could ever favor large facilities. That asymmetry no longer exists: every scenario
  now prices sizes properly, so Madrid competes on equal footing everywhere.
- **Add** the genuinely new deliverable this model enables: a per-site **size recommendation** —
  for each site, its current tier, its modal chosen tier across scenarios, and a label like
  *keep-as-is / upgrade candidate / downsize candidate / close*. This is the answer to the
  lecture's actual question and should headline the recommendation.
- The final folium map (lines 2403–2495): keep the open/close categories, but encode the size
  recommendation too (e.g. marker size = chosen tier, color = keep/upgrade/downsize/close).
- All center names, counts, and cost figures in the prose (lines 2373–2399) must be rewritten
  from the new results.

### Step 11 — Rewrite Section 8 (limitations, lines 2498–2534)

- **Delete** the "impossible without a cascade of assumptions" argument (lines 2521–2527) and the
  percentile-cap limitation (lines 2529). Both are superseded.
- **Add** the honest new limitation: *free re-sizing*. The model prices every (site, size)
  combination off the annual fixed-cost menu; it ignores one-time conversion capex, construction
  time, and whether a physical upgrade is possible at a given site. Recommendations of the form
  "upgrade Zaragoza to m" are therefore contingent on conversion costs being small relative to
  the annual savings — the one number family the data genuinely does not contain.
- Keep limitations 1, 2, 5, 6 (cost-function derivation, processing-cost assumptions, scenario
  scope, constant minutes-per-stop) — they still apply verbatim.

---

## 5. Resulting notebook structure (after migration)

```
1. Problem definition                                     (unchanged)
2. Formalizing the problem                                (limitation 3 reworded → size is a decision)
3. Data                                                   (unchanged; soften "Madrid property prices" remark)
4. Transport cost c_ij: derivation + validation           (unchanged)
5. The size-decision model
   5.1 Extended cost function (new objective with y_it, v_it)
   5.2 Tiers as a price list f_t + official capacity bands [lb_t, ub_t]
   5.3 Variable processing cost c_t^var                   (unchanged)
   5.4 Motivation: what frozen sizes get wrong
        - uncapped frozen-size solve → Guadalajara outlier   (kept, compressed)
        - official bounds + frozen sizes → INFEASIBLE, region-410 certificate (kept, reframed)
   5.5 The full model: formulation, solve, feasible solution, chosen sizes
6. Sensitivity analysis                                   (same sweeps, new solve_network, new prose,
                                                           + tier-profile per scenario,
                                                           − 6.5/6.6 capped-model comparisons,
                                                           + volume/band utilization subsection)
7. Robust recommendation                                  (classification without capped-branch;
                                                           NEW: per-site size recommendation
                                                           keep / upgrade / downsize / close; map)
8. Review & limitations                                   (− "impossible" argument, − percentile-cap caveat,
                                                           + free-resizing limitation)
```

Deleted outright: percentile-cap model (old 5.4), `solve_network_debug`, Sections 6.5/6.6,
the Madrid methodological note. Everything else is kept, reframed, or re-run.

---

## 6. Verification checklist after implementation

1. **Feasibility:** the new base model must report `Optimal`. Region 410 must be assigned to a
   reachable site built at tier ≥ m (expect Zaragoza, travel time 10.8 min).
2. **Band consistency (assert in code):** for every open site, assigned volume lies within its
   chosen tier's `[lb_t, ub_t]`. This single check replaces the whole Guadalajara-outlier hunt.
3. **One size per site:** `Σ_t y_it ≤ 1` holds in the solution (0 or 1 exactly).
4. **Frozen-size cross-check:** `solve_network(freeze_sizes=True)` must reproduce the old
   infeasibility — same certificate, region 410.
5. **Cost plausibility:** report the new optimum next to the two old anchors (uncapped frozen:
   €117.1M; percentile-capped frozen: €146.8M). Neither old model is a relaxation of the new one
   (different fixed costs and capacity sets), so don't assert an ordering — just explain the
   differences.
6. **Prose audit:** grep the markdown cells for stale hard-coded facts — `117,082,899`,
   `146,803,705`, `18 of 42`, `716,917`, `Guadalajara`, `13 of 18`, tier-mix strings like
   `13 / 4 / 1 / 0 / 0` — every hit must be re-derived or rewritten.
7. **Maps and tables:** every place that displays "tier" must say whether it is the *current*
   (historical) or *chosen* tier; the assignment map's marker-size bug (`is_open` vs
   `is_open_m2`, line 1028) and the demand-plot legend bug (`ax.legend()` vs `ax_.legend()`,
   line 1909) should be fixed while touching those cells.
8. **Runtime:** the model grows (~210 extra binaries, per sweep point). If CBC gets slow across
   the ~60 sweep solves, reduce sweep granularity before reaching for a different solver.

---

## 7. Expected headline changes in the results (sanity expectations, not guarantees)

- The official bounds become feasible; the infeasibility finding survives as a statement about the
  frozen-size world.
- Guadalajara can no longer carry 716,917 units in a v-shell; either it is upgraded (paying m/l
  fixed cost) or the volume redistributes. Watch which one the solver picks — it is the single
  most interesting result of the migration.
- Madrid/large tiers now compete fairly in *every* scenario, so the "large centers only appear
  under caps" asymmetry disappears; the final classification will likely move some sites between
  categories.
- The dominance of many small centers may weaken: cheap `c_t^var` at higher tiers is now
  purchasable anywhere volume justifies it.

---

## 8. Optional extension: transition costs (one honest knob)

If you want the model to answer "what should we *convert*" rather than "what would we *build*":
add a one-time conversion cost, annualized, only for sites whose chosen tier differs from the
current one:

```
+ Σ_i Σ_{t ≠ current(i)} δ · f_t · y_it
```

with a single assumed surcharge factor δ (e.g. 10–30% of the target tier's annual fixed cost),
swept in the sensitivity analysis like α and the anchor share. This keeps the assumption count at
exactly one, makes it explicit, and shows how robust the upgrade recommendations are to it. Not
required for the migration — but it directly addresses the free-resizing limitation from Step 11.
