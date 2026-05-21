# Mileage Bucketing Data Science Sprint

| Field | Value |
| --- | --- |
| Status | **READY v9** — all v8 decisions + library fallback applied: `pyqreg` install failed on Python 3.11 + numpy 2.x + Cython 3 (unmaintained package, last release 2022). Switched to `statsmodels.regression.quantile_regression.QuantReg` with iid-SE caveat documented. Algorithm unchanged; standard errors are iid rather than cluster-robust. Ready to commence work. |
| Created | 2026-05-20 |
| Owner | Daan |
| Triggers | Patrol R2.08M finding (see `docs/DRIVE_VALUE_COMPARISON_FINDINGS.md` Finding 4). Two distinct problems with current mileage handling: arbitrary boundaries (dead buckets) and model-agnostic thresholds (50,000 km means different things on a Hilux vs a 911). |
| Scope | Two independent projects, three retail-values outputs. Project 1 = bucket-boundary swap (~1 day) → second retail-values dataset. Project 2 = model-relative mileage classification via quantile regression with brand-tier × body-class segment fallback + retail-values rebuild using the smart labels (~5.5 days) → third retail-values dataset. Status-quo retail values continues unchanged. |
| Out of scope | Cascade-fallback fix from Finding 4 (separate work). Matcher changes. Cloud rollout. API design. |

---

## Executive summary

**Project 1** swaps the 12-bucket `mileage_cat` scheme for a 7-bucket scheme (`0-30k`, `30-60k`, ..., `180k+`). Two source-of-truth files change. Output is a parallel retail-values dataset for A/B comparison. Uniform 30k intervals are stakeholder-specified and empirically suboptimal; we run it for the controlled comparison the stakeholder asked for. Carries a deprecation note — consumers should migrate to the continuous percentile rank from Project 2 within Q3.

**Project 2** builds per-(Make, Model) **mileage-age curves** via **linear+quadratic quantile regression of `log(mileage)` on age** (expressed in months, computed via a mathematically consistent midpoint method), fitted at τ ∈ {0.33, 0.50, 0.66}. Library: `statsmodels.regression.quantile_regression.QuantReg` (iid SEs; documented caveat — see §Library choice). Sparse and unknown (Make, Model) cells fall through a **5-level hierarchy** ending at brand-tier × body-class segments (so a Ferrari F40 falls back to "ultra-luxury passenger" rather than "all coupes"). Output is fitted coefficients + classification function returning **`Low` / `Medium` / `High` / `out_of_scope`** label, continuous percentile rank 0–100, plus provenance and data-quality fields. Project 2 ALSO produces a retail-values rebuild that slices on the new mileage label instead of the integer bucket.

The data philosophy: **filter aggressively for curve fitting; label everything in the output.** Suspicious-mileage listings (placeholders, fat-finger patterns) are excluded from the QR fit but still receive a label at output time with a `mileage_data_quality` flag.

### Architecture — single matched_results, three retail-values builds

The cleanest architecture (decided 2026-05-21): **enrich `matched_results` with all three mileage classifications as columns once**, then run the existing retail-values calculation three times — each switching its slicing column via config — to produce the three parallel datasets.

**Single enriched matched_results** carries:

| Column | Source | Notes |
| --- | --- | --- |
| `mileage_category` | status-quo (12-bucket integer from current `mileage_cat` macro) | Unchanged |
| `mileage_category_7bucket` | Project 1 (7-bucket integer from new macro) | New column added to matched_results |
| `mileage_label` | Project 2 (Low/Med/High/out_of_scope) | New column |
| `mileage_percentile_rank` | Project 2 (0–100 continuous) | New column |
| `mileage_label_reason` | Project 2 (fit-grain provenance) | New column |
| `mileage_data_quality` | Project 2 (ok / placeholder_value / suspicious_pattern / anomaly_high) | New column |

Once these columns are landed in matched_results, the three retail-values runs are simple config switches:

| Version | retail-values config | Mileage slicing column | Output blob path |
| --- | --- | --- | --- |
| **v1 — Status quo** | unchanged | `mileage_category` | `retail_values/` |
| **v2 — Simple resize** (Project 1) | `mileage_slicer: mileage_category_7bucket` | 7 integer buckets | `retail_values_7bucket/` |
| **v3 — Smart buckets** (Project 2) | `mileage_slicer: mileage_label` | 4-tier label | `retail_values_smart/` |

Cascade chain, geographic grain, combine logic are identical across the three runs — only the slicing dimension differs. Comparison is then a straightforward dataset-vs-dataset diff.

Downstream consumers (client_matcher, Drive comparison) read from matched_results once and get all the mileage classifications for free; they read from whichever retail-values dataset the comparison study selects.

---

## Context

The Patrol R2.08M investigation exposed two distinct problems:

1. **Bucket boundaries are arbitrary and inflict dead-bucket damage.** The 12-bucket scheme produces unstable retail values at boundaries that empty out as cohorts age.
2. **Fixed km thresholds are model-agnostic.** 50,000 km is "barely broken in" on a Hilux but "very used" on a Porsche 911.

This sprint addresses both. Project 1 changes the bucket boundaries. Project 2 introduces model-relative classification independent of buckets.

---

## Project 1 — Bucket Boundary Resize

### What changes

Replace the 12-bucket scheme with a 7-bucket scheme:

| New bucket | Range (km) |
| --- | --- |
| 0 | 0 – 30,000 |
| 1 | 30,000 – 60,000 |
| 2 | 60,000 – 90,000 |
| 3 | 90,000 – 120,000 |
| 4 | 120,000 – 150,000 |
| 5 | 150,000 – 180,000 |
| 6 | 180,000 + |

**No other logic changes.** Cascade chain, geographic grain, combine logic — all unchanged. Output is a parallel retail-values dataset for A/B comparison.

Uniform 30k intervals are stakeholder-specified. We note for the record that uniform intervals are empirically suboptimal (depreciation is concentrated in 30–60k km and flattens in 120k+); this concern belongs to Project 2's domain.

### Code-impact map

The 7-bucket function is a new bucketing function, **added alongside the existing 12-bucket function**, not replacing it. Both run; matched_results gains a new `mileage_category_7bucket` column alongside the existing `mileage_category`.

| File | Lines | Change |
| --- | --- | --- |
| `retail_values/calculate.py` | 269–286 | Add a SECOND DuckDB SQL macro `mileage_cat_7bucket(km)` next to the existing `mileage_cat(km)`. Both used; `mileage_cat` produces `mileage_category` (status quo), `mileage_cat_7bucket` produces `mileage_category_7bucket` (new). |
| `client_matcher/key_builder.py` | 11–12 | Add a SECOND Python `MILEAGE_BINS_7BUCKET` list, `MILEAGE_LABELS_7BUCKET = range(7)`, and a `calculate_mileage_category_7bucket()` function. Existing function untouched. |

A regression test (`tests/test_mileage_bucket_consistency.py`) asserts the new SQL macro output matches the new Python `calculate_mileage_category_7bucket` on boundary edges (29999/30000/30001 etc.). The existing 12-bucket consistency test (if any) continues unchanged.

**Boundary semantics**: half-open intervals `[lower, upper)` — left-inclusive, right-exclusive. The DuckDB macro uses `< 30000`, `< 60000`, etc. (NOT `BETWEEN`, which is inclusive on both ends).

**Field-name-stable consumers** (no edits): `retail_values/combine.py`, `client_matcher/retail_joiner.py`, `client_matcher/column_mapper.py`, `client_matcher/xlsx_writer.py`, `mm_matcher_analytics/views/retail/05_spread.sql`, `mm_matcher_analytics/views/retail/06_bias.sql`, `client_matcher/config/clients/drive/config.yaml`.

**Documentation needing find-and-replace**: `mm_matcher_analytics/README.md`, `docs/RETAIL_VALUES_IMPROVEMENT_SPRINT.md`, `docs/MULTISOURCE_PIPELINE_PLAN.md`, `docs/SQL_DECOUPLING_SPRINT.md`, `docs/RETAIL_VALUE_CALCULATION_FLOW.md`, `client_matcher/IMPLEMENTATION_PLAN.md`, `README.md`.

### Migration steps

1. Branch `feature/mileage-7bucket`.
2. Edit the two source-of-truth files — add the new functions alongside the existing ones + add regression test.
3. Apply the new bucket function as a matched_results enrichment column (`mileage_category_7bucket`).
4. Run retail-values calculation with `mileage_slicer: mileage_category_7bucket` config → writes to parallel blob path `retail_values_7bucket/`. Status-quo `retail_values/` path is untouched.
5. Re-run client matcher (it consumes the enriched matched_results plus whichever retail-values dataset is selected).
6. **A/B comparison report** (`docs/analyses/2026-XX_bucket_resize_ab.md`): per (MMCode, RegYear) cell, compute `delta_pct` vs status quo. Histogram, ranked top-N divergent cells, Drive-comparison metric shift, Patrol bucket-0 specific check.
7. Update documentation.
8. **Decision gate**: stakeholder review.

Note: Project 1's matched_results column addition AND Project 2's matched_results columns can land together in Phase D rather than as separate enrichment passes. The two projects produce different columns; they don't conflict.

### Acceptance checks

- **Year-aggregate stability**: `retail_value_year_N` must be byte-identical between old and new.
- **Patrol bucket-0 cell**: should now pick `cascade_window ∈ {45, 90, 180, 365}` instead of `'all'`.
- **Drive comparison metrics**: median ratio stays near 1.00, % within ±10% does not decrease, % >20% above Drive should decrease.

### Effort

~1 day total (2h code + test, 1h rebuild, 4h A/B report, 1h docs).

---

## Project 2 — Model-Relative Mileage Curves

### Problem statement

Fixed-km thresholds cannot answer "is this car high mileage for what it is?" because the answer is model-dependent. The SA industry-aggregate of ~20,000 km/year masks large per-model differences driven by buyer behaviour, fleet/private split, and use case.

**Deliverable:** per-(Make, Model) fitted mileage-age curve (QR coefficients at three terciles), from which downstream consumers compute (a) **`Low` / `Medium` / `High` / `out_of_scope`** label and (b) **continuous percentile rank 0–100** for any listing. Output is independent of retail values — a separate analytical product.

### Data philosophy: two-tier mental model

The matched_results data has two distinct uses in this sprint. Conflating them caused errors in earlier drafts.

| Purpose | Mental model | What's in scope |
| --- | --- | --- |
| **Building the curves** | Clean, trustworthy observations only. Quantile regression's robustness handles outliers within reason but it cannot fix placeholder values masquerading as odometer readings. | Drop suspicious / placeholder mileages; cap > 2M km; require positive age. ~62k listings excluded out of ~2.9M (~2%). |
| **Labelling the output** | EVERY listing in matched_results gets a label. The label is a primary output of this sprint; no listing can be left without one. | All listings — including the ones excluded from the fit — receive a Low/Med/High/out-of-scope label and a continuous rank. A `mileage_data_quality` field surfaces the data-quality concerns to downstream consumers. |

### Research findings summary

#### Industry methodology survey

| Authority | Method | Key parameters |
| --- | --- | --- |
| **Manheim Market Report (US)** | Simple linear regression of price-on-mileage per (model year, make, body class). ~20 market classes. Outliers removed where deviation exceeds ±2.6 SD on **both** price and mileage **jointly**. 24-month rolling weight. | Linear regression per class; joint 2.6 SD outlier filter |
| **Kelley Blue Book (US)** | "Typical mileage by age" tables refreshed regularly. Methodology not publicly disclosed. | Typical mileage by age; transaction-data driven |
| **TransUnion South Africa** | Multi-factor adjustment. ~12M vehicles, ~60k new records/month. Methodology not publicly disclosed. | Multi-factor; expert review |
| **Lightstone Auto (SA)** | Bank-financed transaction data (~2.8M records). Adjusts for mileage + condition. | Transaction-anchored |
| **Academic — Hong et al. (2020)** | Quantile regression on lifetime mileage. Finds that average annual mileages decline with age. | QR at p25/p50/p75 |
| **Lam (2015)** | Adaptive splines per make-model — note: price-vs-mileage, not mileage-vs-age. | Splines on price-vs-mileage |

#### SA benchmarks (for sanity-checking output)

| Benchmark | Value |
| --- | --- |
| Typical SA annual mileage (pre-2020) | 20,000–25,000 km/year |
| Average sold vehicle (2025) | 73,646 km |
| Implied 2025 annual rate (sold market avg) | ~14,700 km/year |

The pre-COVID 20-25k vs 2025 ~14.7k gap suggests SA driving has shifted downward. We publish `median_annual_km` per (Make, Model) as a diagnostic.

#### Labelling convention

Industry convention for "Low / Medium / High" is **terciles (p33 / p66)** — equal cell counts. We adopt this for the discrete label. We add a fourth label `out_of_scope` for genuinely unclassifiable cases.

### Methodology comparison

| Methodology | Verdict |
| --- | --- |
| **A. Empirical percentiles + smoothing + Bayesian shrinkage** (v1 draft) | Rejected. Three layered approximations; hand-picked smoothing window and shrinkage prior; no closed-form SEs. |
| **B. Linear+quadratic quantile regression** of `log(mileage) ~ age + age²`, fit per (Make, Model) | **Selected.** Validated qualitatively by Hong et al. (2020). Smooth curves by construction. Within-Make+Model age pooling = the shrinkage that matters. iid SEs via `statsmodels` (pyqreg fallback documented below). |
| **C. GAMs / spline-based QR** | Rejected for v1. v2 candidate if linear+quadratic insufficient. |
| **D. Full hierarchical Bayesian** | Rejected (tooling cost). v2 candidate. |

**Library choice: `statsmodels.regression.quantile_regression.QuantReg`** with `cov_type='iid'`. The original v6 plan called for `pyqreg` (cluster-robust SEs), but pyqreg failed to install on the prod environment (Python 3.11 + numpy 2.x + Cython 3 compatibility issues; pyqreg is unmaintained since March 2022). Falling back to statsmodels keeps the algorithm identical (linear+quadratic QR via dual-simplex linear program) and produces the same β coefficients; the only difference is the standard errors are iid rather than cluster-robust.

**Implication of iid SEs:** Standard errors are slightly optimistic because cross-source duplicate listings of the same physical vehicle (no VIN to dedupe across autotrader / cars.co.za) introduce residual within-cluster correlation that iid SEs ignore. The **β coefficient point estimates are unaffected** — only the inference / pass criteria become slightly more permissive. Phase C SE-based validation acknowledges this: a failed slope-SE check is still genuinely a failure (the threshold catches real noise); a passed check is less conservative than cluster-robust would have been. Mitigation is the cross-validation in Phase C check 2 which doesn't depend on the SE adjustment.

### Recommended approach

#### Data preparation (curve-fitting inputs)

1. Load `matched_results` (all 3 sources): one row per Car_ID with `(Make, Model, RegYear, MinDate, SaleDate, Mileage, Source, Dealership, Price)`.
2. **Sanity-check `MinDate` semantics in Phase A** — confirm first-observed-on-platform, not re-listing date.
3. **Dedupe** at `(Source, Dealership, MMCode, RegYear, Mileage)` — keep earliest `MinDate`. **Price deliberately excluded** to avoid the price-drop duration bias.
4. **Drop rows where the underlying record is fundamentally unusable for ANY purpose** (Price = 0 placeholder, Make or Model NULL).
5. **For curve fitting only — exclude suspicious mileage patterns:**
   - **Placeholder values** (likely "we don't know" rather than real odometer): `mileage IN {0, 1, 10, 100, 101}` — ~62k listings combined.
   - **Fat-finger / typo patterns**: `mileage IN {1234, 12345, 123456, 1234567, 1234567890, 1111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, 9999999}` — ~400 listings.
   - **Extreme typos**: `mileage > 2,000,000` — ~12 listings.
6. **Age computation — mathematically consistent midpoint method** (in months):

    ```python
    def estimate_age_months(observation_date, regyear):
        """Best-estimate age using the midpoint of the possible-registration range.

        We only know RegYear (an integer year), not the exact registration date.
        The car was registered some day in RegYear, no later than the observation
        date. The unbiased best estimate of age is the midpoint of:
          - earliest possible age (registered Jan 1 of RegYear) — gives age_max
          - latest possible age (registered as late as possible) — gives age_min

        This naturally handles current-year observations (where the upper bound
        on registration date IS the observation date) and cross-year observations
        (where the upper bound is Dec 31 of RegYear).
        """
        earliest_reg = date(regyear, 1, 1)
        latest_reg   = min(observation_date, date(regyear, 12, 31))
        age_max_days = (observation_date - earliest_reg).days
        age_min_days = max(0, (observation_date - latest_reg).days)
        return (age_max_days + age_min_days) / 2 / 30.44
    ```

    `observation_date = MinDate` for both listings and sales (rationale: our sale records carry listing-time mileage paired with a sale-date metadata column; the mileage was observed when the listing first appeared, not at the moment of sale).

7. **Age handling for fitting**:
   - **Negative ages must be investigated, not auto-dropped.** An observation date preceding the registration year is impossible in reality, so any case where `age_months < 0` indicates one of three failure modes worth root-causing:
     - **Wrong MinDate** — possibly a re-listing date used in place of first-observed; or a corruption from the snapshot pipeline.
     - **Wrong RegYear** — matcher returned an incorrect RegYear, or source data has the wrong year on the listing.
     - **Time-zone / date-cast bugs** — unlikely but possible.

     Phase A includes a discovery query to count negative-age cases, group them by Source × Make × RegYear, sample 20 random examples, and identify the failure mode. The fix depends on what we find: if systematic (e.g., one source's MinDate consistently wrong), report upstream and fix. If scattered random errors, drop them from the fit with a documented count and mark them as `out_of_scope` at labelling output. Decision deferred until the discovery query has run.

   - **Drop where `age_months > 300`** (>25 years; collectible territory). Cars older than this still get labelled at output (with `out_of_scope`), but don't influence the fits.

   - **No minimum-age cutoff** — young vehicles are kept in the fit. A 3-month-old vehicle with 18k km informs the upper tercile at very young ages; a 3-month-old with 1k km informs the lower tercile.

#### Curve fitting

For each (Make, Model) with `n_observations ≥ 30` AND `observed_age_range_years ≥ 4`:

$$\log(\text{mileage}) = \beta_0^{(\tau)} + \beta_1^{(\tau)} \cdot \text{age}_c + \beta_2^{(\tau)} \cdot \text{age}_c^2 + \varepsilon$$

where $\text{age}_c = \text{age\_months}/12 - 8$ (centered on age 8 years for numerical conditioning), fitted for $\tau \in \{0.33, 0.50, 0.66\}$ via `statsmodels.regression.quantile_regression.QuantReg(y, X).fit(q=tau, cov_type='iid')`. SEs and `Cov(β₁, β₂)` come from `.cov_params()`.

Store `(β₀, β₁, β₂, β₀_se, β₁_se, β₂_se, Cov(β₁, β₂))` per (Make, Model, τ).

**Conditional fallback to linear-only:**

- If `observed_age_range_years < 4` but `n ≥ 30`: fit linear-only (drop the `age²` term).
- After fitting quadratic: check monotonicity. If predicted curve is non-monotonic in the [0, 25 year] window (typically `β₂ < 0` with the parabolic peak inside the window), **refit linear-only** for that specific (Make, Model, τ). Set `β₂ = 0` and record `quadratic_fit = False`.

If `n < 30`: do not fit at (Make, Model) grain. Mark for parent-hierarchy fallback (see below).

#### Brand-tier × body-class segmentation (the v6 fallback hierarchy)

Two reference taxonomies, hand-curated as one-off configuration assets:

**Brand tier** — `config/brand_tiers.yaml`. **The mapping below is a STARTING POINT only — it MUST be reviewed and fine-tuned by someone with SA-market knowledge before being committed.** Brand positioning in SA differs from global perception: Haval in SA is positioned more premium than a typical Toyota or VW model, Chery is moving up-market with the Tiggo range, and Mahindra has a Bakkie-utility flavour distinct from cheap-passenger budget makes. Treat the starter list as a draft to argue with, not as the final answer.

| Tier | Starter list (subject to SA-market review) |
| --- | --- |
| `ultra_luxury` | Ferrari, Lamborghini, McLaren, Bentley, Rolls-Royce, Bugatti, Pagani, Aston Martin, Maserati, Lotus, Koenigsegg |
| `luxury` | Mercedes-Benz, BMW, Audi, Porsche, Land Rover, Range Rover, Lexus, Volvo, Jaguar, Alfa Romeo, Mini, Tesla, Genesis, Infiniti, Cadillac, Acura |
| `mainstream` | Toyota, Volkswagen, Ford, Nissan, Honda, Mazda, Hyundai, Kia, Renault, Subaru, Mitsubishi, Peugeot, Citroen, Chevrolet, Opel, Fiat, SEAT, Skoda |
| `budget` | Suzuki, Chery, Haval, GWM, Mahindra, MG, JAC, BYD, Tata, Datsun, Daihatsu, Proton |
| `commercial` | Hino, UD Trucks, FAW, Iveco, Foton |

**Body class** — `config/body_class.yaml` — derived from `mmcodes.BodyType`:

| Class | Source BodyTypes |
| --- | --- |
| `passenger` | HATCHBACK, SEDAN, COUPE, CABRIOLET, STATION WAGON |
| `utility` | SUV, CROSSOVER, MPV |
| `bakkie` | SINGLE CAB BAKKIE, DOUBLE CAB BAKKIE |
| `commercial_body` | PANEL VAN, BUS, TRUCK |

**Fallback hierarchy (5 levels):**

| Level | Grouping key | What it means |
| --- | --- | --- |
| 1 | `(Make, Model)` | The specific nameplate |
| 2 | `Make` | All models of this make pooled — captures this-brand-buyer driving behavior |
| 3 | `(brand_tier, body_class)` | e.g., `(luxury, passenger)` or `(ultra_luxury, passenger)` |
| 4 | `brand_tier` | All vehicles of this tier regardless of body |
| 5 | Global | Last resort |

Each level fits the same QR procedure at coarser grain at build time. The first level with `n ≥ 30` is the cell's curve. The parent's coefficients are **denormalized into the (Make, Model) row** so lookups are always O(1).

**Trace examples:**

| Vehicle | Level reached | Compared against |
| --- | --- | --- |
| Toyota Corolla (n=50k+) | 1 | Itself |
| Alfa Romeo Giulia Quadrifoglio (sparse) | 2 | All Alfa Romeo listings (Giulia + Stelvio + Mito + …) |
| Ferrari F40 (n=0) | 3 | `(ultra_luxury, passenger)` — Ferrari/Lambo/McLaren/Bentley/Aston passenger cars |
| Tata Indica (sparse) | 2 | All Tata listings |
| Niche budget hatch never seen | 3 | `(budget, passenger)` — Tata + Chery + Suzuki + Datsun hatches/sedans |
| Toyota Hilux 2.8 GD-6 Legend | 1 | Itself (n=lots) |
| Niche commercial bakkie | 3 | `(mainstream, bakkie)` — all Hilux + Ranger + D-Max + NP200 etc. |

#### Per-listing classification

```python
def classify_mileage(make, model, model_year, current_date, mileage, data_quality):
    age_m = estimate_age_months(current_date, model_year)

    # Out-of-scope cases — vehicle exists outside our comparison universe
    if age_m < 0:
        return ('out_of_scope', None, 'negative_age', data_quality)
    if age_m > 25 * 12:
        return ('out_of_scope', None, 'age_out_of_range', data_quality)

    coefs = lookup(make, model)                       # denormalized row from fallback
    age_c = age_m / 12 - 8
    log_p33 = coefs.beta0_33 + coefs.beta1_33*age_c + coefs.beta2_33*age_c**2
    log_p50 = coefs.beta0_50 + coefs.beta1_50*age_c + coefs.beta2_50*age_c**2
    log_p66 = coefs.beta0_66 + coefs.beta1_66*age_c + coefs.beta2_66*age_c**2
    p33_km, p50_km, p66_km = (exp(log_p33), exp(log_p50), exp(log_p66))

    # Anomaly threshold with physical floor — protects fleet/Uber vehicles
    is_anomaly = mileage > max(p66_km * 2.5, 80000)

    label = 'Low' if mileage < p33_km else ('Medium' if mileage < p66_km else 'High')
    rank = log_anchored_rank(mileage, p33_km, p50_km, p66_km)

    # Combine fit-grain provenance with anomaly / data-quality info
    label_reason = coefs.fit_grain                    # make_model / make_only / tier_body / tier / global
    data_quality_out = data_quality if data_quality != 'ok' else (
        'anomaly_high' if is_anomaly else 'ok'
    )
    return (label, rank, label_reason, data_quality_out)


def log_anchored_rank(m, p33, p50, p66):
    """Continuous rank in [0, 100], anchored exactly at p33/p50/p66.
    Inside [p33, p66]: linear interpolation in log-mileage space.
    Outside: log-normal-based asymptotic extrapolation."""
    log_m, log_p33, log_p50, log_p66 = log(m), log(p33), log(p50), log(p66)
    if log_p33 <= log_m <= log_p50:
        return 33 + 17 * (log_m - log_p33) / (log_p50 - log_p33)
    if log_p50 < log_m <= log_p66:
        return 50 + 16 * (log_m - log_p50) / (log_p66 - log_p50)
    if log_m < log_p33:
        sigma_lower = (log_p50 - log_p33) / 0.4399
        z = (log_m - log_p50) / sigma_lower
        return max(0.0, normal_cdf(z) * 100.0)
    # log_m > log_p66
    sigma_upper = (log_p66 - log_p50) / 0.4125
    z = (log_m - log_p50) / sigma_upper
    return min(100.0, normal_cdf(z) * 100.0)
```

### Bias considerations

| Bias | Mechanism | Mitigation |
| --- | --- | --- |
| **Duplicate-listing inflation** | Same listing republished feeds the model multiple times. | Dedupe at `(Source, Dealership, MMCode, RegYear, Mileage)` at ingestion. |
| **Price-drop duration bias** | Including Price in dedup key would over-weight slow-moving (high-mileage) stock. | Explicitly exclude Price from dedup. |
| **Cross-source duplicates** | Same physical car on autotrader + cars.co.za with different Car_IDs (no VIN). | Acknowledged residual bias. With statsmodels iid SEs, the within-cluster correlation is not adjusted for — SEs are slightly optimistic. β point estimates unaffected. Could be revisited if pyqreg or similar becomes installable later. |
| **Listing-vs-sales duration bias** | High-mileage cars may linger longer before selling. | Pool equally; publish `n_from_listings` / `n_from_sales` as diagnostic. |
| **Survivorship bias** | High-mileage cars may transact off-platform. | Document. Not solvable from our data. |
| **Geographic bias** | Gauteng dominates. | National-average for v1. Per-province in v2 if needed. |
| **Time-period bias** | Pre/post-COVID driving regimes differ. | Phase C validation with fixed age-bin stratification. |
| **Fleet-vs-private use** | Fleet Polo Vivo (~60k km/yr) vs retiree Polo Vivo (~8k km/yr). | Data-driven `fleet_heavy` flag (`median_annual_km > 30,000`); hard-coded sanity list as build-gate. |
| **Generation changes within nameplate** | 2008 Hilux vs 2024 Hilux pooled together. | Phase C variant-stratification check; add generation handling in v1.5 if material. |
| **Cross-tier brand contamination** | A Make spanning multiple tiers (Mercedes commercial vans vs C-Class) gets one tier assignment. | Mostly invisible — niche models stop at Level 2 (Make-only), not Level 3 (Tier). |

### Output product

A new dataset at `analytics/mileage_curves/`:

#### `mileage_curve_fits.parquet` — one row per (Make, Model, τ)

| Column | Type | Description |
| --- | --- | --- |
| `Make` | string | |
| `Model` | string | |
| `tau` | float | 0.33, 0.50, or 0.66 |
| `beta_0`, `beta_1`, `beta_2` | float | Intercept, age_c slope, age_c² coefficient (β₂ = 0 if linear-only) |
| `beta_0_se`, `beta_1_se`, `beta_2_se` | float | iid SEs from `statsmodels.QuantReg.fit().cov_params()` (see Library choice for caveat) |
| `cov_beta1_beta2` | float | Covariance from `cov_params()` — used for delta-method slope SE in validation |
| `n_observations` | int | Sample size used (after dedup, mileage filter, age filter) |
| `n_from_listings`, `n_from_sales` | int | Diagnostic |
| `observed_age_range_years` | float | max(age) - min(age) for this nameplate |
| `fit_grain` | string | `make_model` / `make_only` / `tier_body` / `tier` / `global` — when parent grain used, the parent's coefficients are denormalized into this row |
| `brand_tier` | string | `ultra_luxury` / `luxury` / `mainstream` / `budget` / `commercial` |
| `body_class` | string | `passenger` / `utility` / `bakkie` / `commercial_body` |
| `median_annual_km` | float | `exp(predicted_log_p50_at_age_5) / 5`. Auto-validates against benchmarks. |
| `confidence_flag` | string | `green` (n ≥ 60, fit at make_model), `amber` (30 ≤ n < 60, fit at make_model), `red` (fell through to parent grain) |
| `fleet_heavy` | bool | Data-driven: `median_annual_km > fleet_heavy_threshold_km` |
| `quadratic_fit` | bool | True if quadratic term used; False if linear-only |

**Methodology config embedded in parquet metadata.** No separate sidecar file.

#### Per-listing classification — four output columns

Wired into `client_matcher/column_mapper.py`:

- `mileage_label` ∈ {`Low`, `Medium`, `High`, `out_of_scope`}
- `mileage_percentile_rank` ∈ [0, 100] or NULL for `out_of_scope`
- `mileage_label_reason` — which fallback grain produced the curve: `make_model` / `make_only` / `tier_body` / `tier` / `global` / `negative_age` / `age_out_of_range`
- `mileage_data_quality` ∈ {`ok`, `placeholder_value`, `suspicious_pattern`, `anomaly_high`}

`mileage_data_quality` values:

- `ok` — mileage looks like a real odometer reading
- `placeholder_value` — exact match against {0, 1, 10, 100, 101} (likely "we don't know" indicator)
- `suspicious_pattern` — exact match against fat-finger patterns or > 2,000,000 km
- `anomaly_high` — passed the `max(p66 × 2.5, 80000)` threshold

Listings with `mileage_data_quality != 'ok'` were excluded from curve fitting but still receive a label at output time. Consumers filtering anomalies just check the flag.

### Validation strategy

#### Sanity checks (must all pass)

| Check | Pass criterion |
| --- | --- |
| **Quadratic-aware monotonicity (post-fallback)** | After the programmatic linear-fallback safeguard, `β₁ + 2·β₂·age_c > 0` across the full age window for **100%** of fits. Pre-fallback failure rate published as diagnostic. |
| **Quantile ordering** | At every age in {1, 3, 5, 8, 12} and every fit, `predicted_p33 ≤ p50 ≤ p66`. |
| **Median annual rate cross-Make+Model** | Median across green-flagged fits of `median_annual_km` should land in 10,000–25,000 km/year. |
| **Fit success rate** | ≥80% of (Make, Model) cells with n ≥ 30 produce a successful fit at `make_model` grain. |
| **Fleet-heavy sanity coverage (build-gate)** | Every (Make, Model) in `fleet_heavy_sanity_list` MUST be flagged `fleet_heavy = True`. 100% pass; build halts otherwise. |
| **Brand-tier mapping coverage** | Every Make appearing in matched_results with n ≥ 100 must have a brand_tier assignment in `config/brand_tiers.yaml`. Unmapped makes default to `mainstream` with a warning logged. |
| **Body-class mapping coverage** | Every BodyType appearing in mmcodes master must map to a body_class. 100% required. |

#### Quantitative validation

1. **Effective-slope SE via delta method.** With quadratic basis, slope at age 5 is `β₁ + 2·β₂·(5-8) = β₁ - 6·β₂`. SE computed via:

    ```python
    slope_se = sqrt(Var(β₁) + 36·Var(β₂) + 2·(-6)·Cov(β₁, β₂))
    ```

    **Pass:** median across green fits of `slope_se / |slope_at_5|` < 0.20.

2. **Cluster-aware CV with pinball loss.** 5-fold CV with `cv_random_seed` pinned. Cluster key = the dedup tuple `(Source, Dealership, MMCode, RegYear, Mileage)`. **Pass:** for ≥80% of green fits, `holdout_pinball(model) < 0.95 × holdout_pinball(constant_baseline)`.

3. **Beat-the-naive-baseline.** Naive: `expected_mileage = age × 20,000 km` with ±20% bands. **Pass:** our curve matches empirical label at ≥80% of held-out points; baseline lower.

4. **Cross-model differentiation visual.** Plot p33 / p50 / p66 curves for Toyota Hilux, Porsche 911, Suzuki Swift, BMW 3-Series, Mercedes-Benz E-Class. **Pass:** curves visibly differ.

5. **Cross-tier differentiation visual.** Plot the per-tier curves: ultra_luxury vs luxury vs mainstream vs budget vs commercial (each pooled across passenger body class). **Pass:** ultra_luxury and luxury curves are visibly lower (in mileage per unit age) than mainstream/budget. If they overlap, the tier taxonomy isn't doing useful work and we recheck.

6. **Variant heterogeneity check.** Toyota Hilux — stratify into 3-4 variant clusters (Single Cab vs Double Cab; petrol vs diesel). **Pass:** if cluster-level p50 slopes differ from pooled by `< 30%` relative, pooling is defensible.

7. **Make+Model vs Master_Model.** Fit at both grains for 50 random nameplates. **Pass:** median `|Δβ₁| / |β₁| < 0.10`.

8. **Drive comparison with pattern detection.** Apply `classify_mileage` to Drive's asset list. **Trigger:** if disagreement on a recognisable subset exceeds 40%, flag for v1.5.

9. **Time-period stratification with fixed age bins.** Refit on cohorts (2018–2020, 2021–2023, 2024–2026). Compare `median predicted mileage at fixed age bins` (3-5, 6-8, 9-11). **Pass:** if `|Δp50 at fixed age|/p50 < 15%` across cohorts, no recency weight needed.

#### First-run diagnostics (published with the validation report)

| Diagnostic | What to look for |
| --- | --- |
| `pre_fallback_monotonicity_failure_rate` per τ | Expected < 1%. If 5%+: raise `min_age_range_for_quadratic` to 5 or 6. |
| `fit_grain` distribution | Expected: 70–85% `make_model`, 10–20% `make_only`, 3–10% `tier_body`, <2% `tier`, <0.5% `global`. |
| `median_annual_km` distribution | Cross-fit median in 10k–25k. Tails (<5k or >60k) → data-quality investigation. |
| `cluster_size_distribution` post-dedup | Most clusters size 1–2. ≥10 suggests republishing not caught by dedup. |
| `anomaly_rate` per (Make, Model) | Expected < 2%. ≥10% on a nameplate → `p66 × 2.5` miscalibrated for that nameplate. |
| `drive_disagreement_rate` per subset | Calibrates the 40% trigger threshold. |
| `tier_assignment_coverage` | Every Make with n ≥ 100 maps to a brand_tier. Unmapped makes warning-logged. |

### Reproducibility & configurability

All parameters in `config/mileage_curves.yaml` (config embedded in parquet metadata at build time):

```yaml
mileage_curves:
  data_prep:
    mileage_ceiling_km: 2000000
    placeholder_mileages: [0, 1, 10, 100, 101]
    fat_finger_mileages: [1234, 12345, 123456, 1234567, 1234567890,
                          1111111, 222222, 333333, 444444, 555555,
                          666666, 777777, 888888, 999999, 9999999]
    age_min_months: 0                          # no floor; only drop negative ages
    age_max_months: 300                        # 25 years
    dedupe_key: [Source, Dealership, MMCode, RegYear, Mileage]
    dedupe_retain: earliest_min_date
    zero_price_handling: drop
  curve_fitting:
    library: statsmodels                       # v9: pyqreg failed to install on Py 3.11 + numpy 2.x + Cython 3 (unmaintained since 2022)
    se_type: iid                               # cluster-robust unavailable in statsmodels — residual cross-source-duplicate bias documented
    tau_levels: [0.33, 0.50, 0.66]
    age_center_years: 8                        # subtract from age (years) before squaring
    age_basis: linear_plus_quadratic
    min_age_range_for_quadratic: 4.0
    monotonicity_check_range_years: [0, 25]
    min_n_for_make_model_fit: 30
  fallback_hierarchy:                          # tried in order, first with n>=30 wins
    - make_model
    - make_only
    - tier_body
    - tier
    - global
  taxonomies:
    brand_tiers_path: config/brand_tiers.yaml
    body_class_path: config/body_class.yaml
    default_tier_for_unmapped_make: mainstream
  classification:
    label_cuts: [0.33, 0.66]
    rank_method: log_anchored
    anomaly_multiple_of_p66: 2.5
    anomaly_absolute_floor_km: 80000
  confidence_flags:
    green_min_n: 60
    amber_min_n: 30
  diagnostics:
    median_annual_km_definition: predicted_p50_at_age_5_div_5
    fleet_heavy_threshold_km: 30000
    fleet_heavy_sanity_list:
      - {make: Volkswagen, model: Polo Vivo}
      - {make: Toyota, model: Quantum}
      - {make: Toyota, model: Hiace}
  validation:
    cv_folds: 5
    cv_random_seed: 42
  update_cadence: monthly
```

### Limitations

1. **Not a price model.** Mileage classification only.
2. **Generation pooling.** All Hilux generations pool together. Defensible for driving-behaviour norms.
3. **Survivorship bias unaddressed.** True p66 may exceed our observed p66.
4. **Brand-tier mapping is hand-curated.** Some makes (Mercedes-Benz, Toyota, Suzuki) span tiers. We assign the dominant tier; niche models stop at Level 2 (Make-only) before reaching Level 3 (Tier).
5. **Odometer fraud undetected.** Anomaly flag is our only defense beyond the global filter.
6. **Fleet-vs-private use unmodelled.** Data-driven flag for known-heavy nameplates; otherwise residual noise.
7. **Continuous rank uses log-normal tail extrapolation.** Inside [p33, p66] is exact; outside is smooth and bounded but precision degrades in the tails.
8. **Pre-COVID benchmark drift.** SA driving appears to have shifted downward; the `median_annual_km` diagnostic surfaces this per nameplate.

### Open questions for external review

1. **Brand-tier mapping curation.** Starting list provided in `config/brand_tiers.yaml`. Reviewer / SA-market expert may want to revise placements (e.g., is Genesis luxury or near-luxury? Is Mini luxury or premium-mainstream?).
2. **Body-class commercial split.** Should `commercial_body` separate PANEL VAN from BUS/TRUCK? Probably yes for v2 if commercial classifications materially differ.
3. **Make+Model vs Master_Model.** Recommended: Make+Model. Phase C validates if Master_Model changes β₁ materially.
4. **Anomaly threshold (`max(p66 × 2.5, 80000)`).** Starting calibration; may need per-tier overrides after first run.

### Implementation phases

| Phase | Work | Effort |
| --- | --- | --- |
| **A** | Data prep: load from DuckDB; dedup; placeholder/fat-finger flag (DON'T drop from output, only from fit); global mileage filter; midpoint age computation; **investigate negative-age cases (discovery query, root-cause)**; persist filtered fit-dataset + full labelling-dataset + `dedupe_tuple_id` column | 1 day |
| **B.1** | **Build the two reference taxonomy datasets — methodically, with 100% coverage enforced:** Step 1 — enumerate the full target lists programmatically: `SELECT DISTINCT Make FROM matched_results ORDER BY Make` (~50–80 entries) and `SELECT DISTINCT BodyType FROM mmcodes ORDER BY BodyType` (~15 entries). Step 2 — for each Make, assign one of 5 tiers; for each BodyType, assign one of 4 classes. **Every value must be mapped — no defaults, no missing entries, no surprises.** Step 3 — review with an SA-market expert (especially for the brand tier — see note below). Step 4 — write `config/brand_tiers.yaml` and `config/body_class.yaml` and commit. Build halts if any Make/BodyType is unmapped. | 0.5 day |
| **B.2** | Curve fitting: per-(Make, Model) QR at 3 τ levels using `statsmodels.QuantReg` with iid SEs (pyqreg fallback applied — see Library choice); fit parent levels (`make_only`, `tier_body`, `tier`, `global`); apply fallback hierarchy with parent-coefficient denormalization; write `mileage_curve_fits.parquet` with embedded config metadata | 0.5 day |
| **C** | Validation: 7 sanity checks + 9 quantitative checks (incl. cross-tier differentiation, taxonomy coverage); output validation report | 1.5 days |
| **D** | **Matched-results enrichment + consumer integration:** Implement `classify_mileage()`. Add a matched-results post-processing step that adds the new columns: `mileage_category_7bucket` (from Project 1's bucket function), `mileage_label`, `mileage_percentile_rank`, `mileage_label_reason`, `mileage_data_quality` (from Project 2's classifier). Wire the new columns into `client_matcher/column_mapper.py`. Re-run on Drive sample. | 0.5 day |
| **E** | **Retail-values rebuild — config switch, not code rewrite.** Add a `mileage_slicer` config parameter to `retail_values/calculate.py` controlling which column drives the per-(MMCode, RegYear, slicer) grouping. Run three times: with `mileage_category` → `retail_values/` (unchanged), with `mileage_category_7bucket` → `retail_values_7bucket/`, with `mileage_label` → `retail_values_smart/`. Produces all three parallel datasets from the same enriched matched_results. | 1 day |
| **F** | Documentation: `docs/MILEAGE_CURVES_METHODOLOGY.md`, README updates, ROADMAP closeout | 0.5 day |
| | **Total** | **~5.5 days** |

---

## Cross-Project Considerations

### Architecture summary — enriched matched_results + three retail-values builds

Sprint's headline deliverable: matched_results enriched with all three mileage classifications as columns once, then three parallel retail-values rebuilds driven by config switching the slicing column.

**Enriched matched_results** carries:

- `mileage_category` (status quo 12-bucket; unchanged)
- `mileage_category_7bucket` (Project 1 output added as a column)
- `mileage_label`, `mileage_percentile_rank`, `mileage_label_reason`, `mileage_data_quality` (Project 2 outputs added as columns)

**Three retail-values outputs** in parallel blob paths:

1. **`retail_values/` (status quo)** — slices on `mileage_category`. Untouched. Continues as the production baseline.
2. **`retail_values_7bucket/` (Project 1)** — slices on `mileage_category_7bucket`. Same code, config flag flipped.
3. **`retail_values_smart/` (Project 2 Phase E)** — slices on `mileage_label`. Same code, config flag flipped.

All three feed parallel client_matcher / Drive-comparison runs. The team chooses which becomes the production default after the comparison.

### Project independence

The two projects share no code or schema. They CAN run in any order or in parallel — neither blocks the other. The shared artefact is the comparison machinery (each project produces a parallel blob path that gets benchmarked against status quo and against each other).

### Sequencing recommendation

Run both in parallel if capacity allows. Project 1 is smaller (1 day); Project 2 is larger (5.5 days). If only one person available, Project 1 first because it exercises the rebuild + A/B machinery we want for Project 2 Phase E anyway.

### Project 1: ship with explicit deprecation framing

Decision (v5, retained): ship Project 1 as a 1-day deliverable. The 7-bucket integer becomes deprecated when the Project 2 smart-label version proves itself. Consumers migrate to `mileage_label` + `mileage_percentile_rank` within Q3.

### What this sprint does NOT solve

- Dead-bucket cascade fallback (Patrol R2.08M root cause) — tracked separately.
- Cross-source-tier-mixing in retail values combine — separate matter.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Bucket boundaries diverge between SQL macro and Python after a future change | Med | High | Regression test in `tests/test_mileage_bucket_consistency.py` |
| `pyqreg` cannot be added to build environment (REALIZED v9) | — | Med | Fallback to `statsmodels` applied. iid-SE caveat documented in Library choice. β point estimates unaffected; only SE-based inference is less conservative. |
| Quadratic curve violates monotonicity within age window | Med | Low | Programmatic per-curve refit to linear-only |
| Linear+quadratic age basis insufficient for materially non-linear shape | Low | Med | Phase C visual inspection; escalate to spline basis in v1.5 |
| Variant heterogeneity within nameplate masks intra-model differences | Med | Med | Phase C variant-stratification check |
| Generation drift pooled | Low | Med | Phase C stratification; add generation cohorts in v1.5 if material |
| Drive comparison reveals systematic disagreement | Med | Med | Phase C pattern detection trigger |
| Time-period bias (pre/post-COVID) | Med | Med | Phase C fixed-age-bin stratification |
| Anomaly threshold miscalibrated | Med | Low | Configurable; recalibrate from first-run distribution |
| Sparse cells fall through to parent grain with reduced precision | Low | Low | `mileage_label_reason` makes fallback visible |
| Brand-tier mapping incorrect for a niche make | Low | Low | Niche makes stop at Level 2 (Make-only) before tier matters |
| Body-class mapping breaks if new BodyType values appear in mmcodes | Low | Low | Phase C coverage check fails the build until updated |

---

## Design decisions to preserve (locked)

These are deliberate-but-easy-to-undo choices that emerged from review rounds 1–6.

| Decision | Why it's locked |
| --- | --- |
| **Half-open intervals `[lower, upper)` with explicit `<`** | `BETWEEN` is inclusive on both ends and would double-count boundary values. |
| **`Price` deliberately excluded from the dedup key** | Including Price would over-weight slow-moving (high-mileage) stock via the price-drop duration bias. |
| **Midpoint age estimation method** | Mathematically consistent everywhere. Self-correcting at the current-year boundary. Identical to mid-year-anchor for cars > 1 year old. |
| **Anomaly absolute floor at 80,000 km** | Protects legitimate Uber/fleet vehicles from being incorrectly flagged. |
| **Anomalies get a label + flag, not undefined** | A 100k km 1-year-old Uber is genuinely "very high mileage for what it is." Drop-label discards usable signal. |
| **Log-anchored rank with log-normal tail extrapolation** | Linear-in-km extrapolation saturated too aggressively. |
| **Programmatic monotonicity safeguard with per-curve linear fallback** | Catches the parabolic-reversal failure mode robustly. |
| **`mileage_label_reason` + `mileage_data_quality` as first-class output columns** | Provenance and data-quality flagging are critical to consumer trust; not derivable post-hoc. |
| **Two-tier data philosophy (filter for fitting; label for output)** | The matched_results data has two distinct uses; conflating them caused earlier errors. |
| **Brand-tier × body-class fallback (not body-type alone)** | Body type alone would compare a Quadrifoglio to a Corolla. Tier captures market-segment / use-case correlation that body type doesn't. |
| **Brand-tier mapping is a one-off curation asset, not an automatic derivation** | No reliable automatic mapping exists; hand-curation is cheap (50–80 makes) and maintainable. |
| **Negative-age cases must be investigated, not auto-dropped** | A negative age means our data is internally inconsistent in a way that has a cause. Auto-dropping hides root-cause failure modes (wrong MinDate, wrong RegYear, time-zone bug). Phase A includes a discovery query before any decision is made. |
| **Project 2 produces a retail-values rebuild (Phase E), not just a labels-only product** | The smart-bucket retail values is the third parallel dataset users need to compare against the status quo and the simple-resize. Without it, Project 2 stops short of demonstrating its end-to-end value. |
| **Three retail-values outputs run in parallel; status quo is preserved** | A/B/C comparison requires all three to coexist in blob until the team chooses a production default. Project 1 and Project 2 Phase E write to parallel paths, never overwriting `retail_values/`. |
| **Mileage classifications land as columns on matched_results, not as separate parallel datasets** | matched_results is the single source of truth. All three slicing schemes (12-bucket, 7-bucket, smart-label) sit alongside each other as columns. Retail-values rebuilds are config switches, not code rewrites. Downstream consumers read once and get everything. |
| **Brand-tier mapping requires SA-market expert review before commit** | Global brand positioning differs from SA. Haval is positioned more premium in SA than the budget mapping suggests; Chery is moving up-market with the Tiggo range; Mahindra leans bakkie-utility distinct from cheap-passenger budget makes. The starter list in this doc is a draft to argue with, not the final answer. |
| **Taxonomy build is enumeration-first, 100% coverage enforced** | Every distinct Make in matched_results gets a tier; every distinct BodyType in mmcodes gets a class. No defaults, no missing entries. The build fails if anything is unmapped — surfaces new makes / body types immediately. |

---

## Pre-work checklist (commence after these are confirmed)

Methodology / design checks (already locked in via reviews 1–8):

- [x] M1–M5 fixes applied (v5)
- [x] D1 (label-and-flag for anomalies), D2 (Project 1 ship-with-deprecation) decided
- [x] v6 segmentation upgrade (brand-tier × body-class) and data-philosophy split (filter for fit; label for output)
- [x] v7 Phase E retail-values rebuild added; negative-age investigation step added
- [x] v8 matched_results-enrichment architecture (one source of truth, three slicing columns)

Environment + curation prerequisites (must complete before Phase B.2 runs):

- [x] **D3 — pyqreg attempted; failed.** Numpy 2.x + Cython 3 + Python 3.11 incompatibility on unmaintained 2022-vintage package. Fallback to `statsmodels` applied. Methodology doc updated with iid-SE caveat. Verify statsmodels available:

    ```powershell
    python -c "from statsmodels.regression.quantile_regression import QuantReg; print('statsmodels QuantReg ok')"
    ```

- [ ] **Brand-tier mapping reviewed by SA-market expert.** `config/brand_tiers.yaml` is the v1 starter list; needs review against SA dealer/buyer intuition (Haval positioning, Chery up-market move, Mahindra bakkie-utility flavour).
- [ ] **Body-class mapping covers 100% of distinct `BodyType` values** in `mmcodes` master (build will halt if any unmapped).
- [ ] **Brand-tier mapping covers 100% of distinct Makes** with n ≥ 100 in matched_results (build will halt if any unmapped).

Operational readiness:

- [ ] Validation report includes `cluster_size_distribution` and the other first-run diagnostics.
- [ ] Parallel blob paths (`retail_values_7bucket/`, `retail_values_smart/`) provisioned for writes.

---

## Sources

- [Kelley Blue Book FAQ — used vehicle valuation methodology](https://www.kbb.com/faq/used-cars/)
- [B2B KBB — Vehicle Valuation FAQs for Dealers](https://b2b.kbb.com/kbb-vehicle-values/faq/)
- [Manheim Used Vehicle Value Index — Summary Methodology (PDF)](https://site.manheim.com/wp-content/uploads/sites/2/2023/07/Used-Vehicle-Summary-Methodology.pdf)
- [Manheim Used Vehicle Value Index In-Depth Methodology Overview (LinkedIn)](https://www.linkedin.com/pulse/manheim-used-vehicle-value-index-in-depth-methodology-babak)
- [TransUnion South Africa — Auto Vehicle Valuations](https://www.transunion.co.za/product/auto-vehicle-valuations)
- [Lightstone Auto — vehicle valuation product](https://lightstoneauto.co.za/automotive.aspx)
- [Hong et al. (2020) — Driving propensity and vehicle lifetime mileage: A quantile regression approach](https://www.sciencedirect.com/science/article/abs/pii/S0301479720314249)
- [Weymar & Finkbeiner (2016) — Statistical analysis of empirical lifetime mileage data for automotive LCA](https://link.springer.com/article/10.1007/s11367-015-1020-6)
- [Lam (2015) — Car depreciation and regression splines (price-vs-mileage)](https://longhowlam.wordpress.com/2015/04/17/car-depreciation-and-regression-splines-3/)
- [AutoTrader South Africa — What mileage is good for a used car in SA?](https://www.autotrader.co.za/cars/news-and-advice/buying-a-car/what-mileage-is-good-for-a-used-car-in-south-africa/11525)
- [Dealerfloor — How affordability reshaped South Africa's used car market in 2025](https://dealerfloor.co.za/industry-news/how-affordability-reshaped-south-africas-used-car-market-in-2025)
- [pyqreg — Cluster-robust quantile regression in Python (PyPI)](https://pypi.org/project/pyqreg/)
- [statsmodels — Quantile Regression](https://www.statsmodels.org/stable/examples/notebooks/generated/quantile_regression.html)
- [Hagemann (2015) — Cluster-robust bootstrap inference for quantile regression](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1359774)
- [Ialongo (2018) — Confidence intervals for quantiles from grouped data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5828320/)
- [Koenker & Machado (1999) — Goodness of fit and related inference for quantile regression](https://www.tandfonline.com/doi/abs/10.1080/01621459.1999.10474138)
