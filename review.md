# Overview of every decision & finding
## General modeling and analysis

- Problem: transportation cost $c_{ij}$ was not defined, we had to derive it using our own logical formula
- Problem: unreachable links - Decision: links that exeed the possible travelTime were dropped. For several reasons: more robust (solver doesn't even see them with Big-M there might be a chance), and we checked if every region still has on reachable center (they did). This reduced the model from 21,630 possible links to 2,661
- After cost calculation we see that the closer we get to the feasibility limit, annual transport cost cost increases 
- Finding: when we displayed the head of the transport cost table, we found 6 (center, region) pairs with 0 travel time but very different transportationCosts. We therefore checked if both cities of center and region matched: they did! Based on those 6 centers we said, that variation like that is not a data issue (we said this is true for all cases this variation is the case, even for ones we didn't see and check yet)
- Transport cost were validated against the courses benchmark: it was basically a perfect match
- We then extended the model to fix the remaining problems. New: size-dependent processing cost
- We classified centers into size tiers based on natural fixed cost gaps to the previous centers
- We found 5 center tiers, with the second largest ("l") and the largest ("h") only containing a single center. We decided to not combine the two larges centers into a single tier because the fixed cost gap was too big (20.892.000)
- We then derived the variable processing cost per center. This was based heavily on assumptions and "data-driven ideas". Idea: the bigger the center, the stronger the Economies of Scale apply. We made this operational by using the fixed cost ratios between the smallest center "v" and the other center tiers. Those ratios were too aggressive so we dampened it with a function: $$c_t^{var} = c_v^{var} \cdot s_t^{-\alpha}$$. We decided to use alpha = 0.5. To get the final processing cost we needed to derive an "initial variable processing cost" we could scale with the new dampened scaling factors. We decided to use 15% of the median transport cost per delivery across all 2,661 links. A lot of decisions were made here:
1. fixed cost ratio as basis 
2. scaling the ratios
3. using a power function for dampening
4. setting alpha to 0.5
5. using a fraction of the median as the inital processing cost
6. setting the fraction to 15% 

- complete model: for the complete model we deliberately did not include volume bounds for each center tier. We know that this might introduce problems but it was a logical decision for several reasons: 

1. We used fixed cost as a proxy for center size, since we did not have any entries indicating center size. 
2. Based on this decision we could only use the decision "should CashLog upgrade the size of center or not" as a modeling choice. 
3. This would introduce: a) a new binary variable (Upgrade: yes/no) and b) we would need to derive a "shift factor" for every combination of possible upgrades. The factor the fixed cost would increase if for example a center "v" was upgraded to "l". This would be another additional decision we had to defend. I also expect some centers to be at the exact volume limit if we introduced it, so the "artificial limit" we would introduce would have a significant impact on the Network design. So the design would strongly depend on an assumption we made that wasn't really based on solid evidence. 
4. Here in this particular dataset, we could easily calculate the factors, because we have 5 distinct price entries for the tiers. But this is not realistic at all in real life.
5. We therefore see this as a limitation and just closely monitor if it impacts our outputs

- we solve the problem with our extended model: The two biggest centers were closed, because fixed costs were just to high, so spreading out demand over several centers was cheaper. But here, our volume cap decision caused a problem. Warehouse 19 (Guadalajara) as a "v" center was assigned 716,917 units. Way too much for a small center to handle. We therefore did build a simple comparison model where we introduced volume caps, just to see if it would change the network,

- comparison model: we decided to cap volume at 1.5x 75% percentile from the uncapped solution, excluding the outlier. This is not really based on anything, we just use it as a simple way of building a comparison model to see if there were significant differences. Findings: 
1. total cost: +29,720,806, +1 open center, tier mix(new): v:11; s:6; m:1; l:0; h:1, tier mix(old): v:13; s:4; m:1; l:0; h:0. Madrid was the newly opened big warehouse and Guadalajara was closed. 
2. We see this as a recomendation. If Guadalajara can be ugraded and still be cheaper than madrid after that upgrade, they should do so. Otherwise a different tier mix would be better. 

## Sensitivity Analysis

- We test 4 main axes with one split into two parts:
1. Economoes of scale strength (alpha) and pct of medain (15%):
- alpha: we varied alpha from: 0.3; 0.5; 0.7 and 1.0: Findings: at 0.3 and 0.5 we have 18 open centers. At 0.7 and 1.0 we have 17 open. So we looked closer at 0.5 (base) vs. 1.0: at 1.0, five small centers closed (Avila, Cordoba, Guadalajara, Huelva, and Orense) and 4 larger centers opened (Pontevedra, Sevilla, Toledo, and Zamora). We looked at the reassignement next: 189 regions were reassigned, most of those came directly from closing centers (Regions that belonged to the 5 closed centers: 158). Sevilla absorbed all 35 regions previously split between Huelva and Cordoba. Summary: stronger economies of scale reward moderate consolidation into the next tier up.

- processing-cost anchor: we resolved the model with 10%, 15% and 25%: similar but more modest sensitvity to alpha: cost shift to a max of around +10% (for 15% of median). Tier mix only shifts slightely at the upper end (processing_share=0.15: n_open=18, tiers={'v': 13, 's': 4, 'm': 1} processing_share=0.25: n_open=17, tiers={'v': 10, 's': 5, 'm': 2}). I added new code here: 

```python
closed_anchor = set(anchor_results[0.15]["open_centers"]) - set(anchor_results[0.25]["open_centers"])
print(f"Closed at anchor_factor={0.25}:", warehouses_flat.set_index("warehouseID").loc[list(closed_anchor), "city"].tolist())
opened_anchor = set(anchor_results[0.25]["open_centers"]) - set(anchor_results[0.15]["open_centers"])

print(
    "Opened at anchor_factor={0.25}:",
    warehouses_flat.set_index("warehouseID")
                   .loc[list(opened_anchor), "city"]
                   .tolist()
)
```
we see: Closed at anchor_factor=0.25: ['Orense', 'Avila', 'Cordoba', 'Guadalajara', 'Huelva']; Opened at anchor_factor=0.25: ['Sevilla', 'Pontevedra', 'Toledo', 'Zamora']; (I did not check reassignment here yet - tell me if I should). Those centers mimik the 4 centers opened at the alpha check. This is a very strong point for our final recomendation.

2. Falling Cash demand: we test what happens when annual demand drops by: 10%, 30% and 50%. Findings: Network unchanged at -10%. At -30% one "v" center closes and at -50% two more close. "s" and "m" center remain unchanged. The centers that were closed [Burgos, Lugo] were not yet notable. No overlap to other closed centers

3. Rising wages and fuel costs: we test a 