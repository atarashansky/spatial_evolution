# Round-7 pre-registration (written before any round-7 data are generated)

Filed in response to the round-6 panel's two dominant charges: (T3) the census-based derivation of the ridge is identification-dependent — the competing population "opposite-clone Moore contacts" was selected because it reproduces the ridge, making the inversion a consistency check rather than a derivation; (T2) the headline ridge value 0.44 (the s_d = 0.01 crossing) sits below the pooled constant 0.55–0.56 and is an effect-size dependence, not a constant.

## A. A-priori identification of the competing population, derived from the update rule

The kernel implements a death–Birth (dB) Moran rule: a site dies (Bernoulli deaths) and the vacancy is refilled from its full Moore contact set (8 sites, both clones), source chosen fitness-proportionally (`_run_strip`, source-draw block: `tot = Σ fitness over all contacts`; `r < acc` selection).

**Claim, stated before inspecting any round-7 census.** The interface moves only through cross-clone conversion events: a boundary site of one clone dies and is refilled from an opposite-clone contact. Same-clone refills do not move the interface. Therefore the individuals whose fitness participates in an *interface-moving* competition are the dying boundary site's **opposite-clone contacts**; the same-clone contacts enter the conversion probability p = W_opp·n_opp/(W_opp·n_opp + W_same·n_same) only as a denominator that dilutes conversions uniformly, not as competitors whose least-loaded class must be maintained. The least-loaded class is dynamically decisive only where its members can convert — i.e., in the opposite-clone contact set of boundary sites.

**Prediction 1 (identification, fixed in advance).** The population N_eff whose census sets the ridge is the mean number of opposite-clone contacts of a boundary site, and it is a property of the *stencil*, not of the load: it should be reproduced under any contact geometry with the identification unchanged.

**Prediction 2 (the falsification test — von Neumann intervention).** Under the four-site (von Neumann) stencil the mean opposite-clone contact count of a boundary site is smaller than under Moore (round-5 measurement: 1.63 opposite / 2.37 same for vN vs 2.90 / 5.10 for Moore). With the identification FIXED to opposite-clone contacts and θ₀ FIXED to the Moore-census value (0.404), the census inversion must predict the von Neumann ridge with no free parameter. The vN ridge was measured independently in the stencil test (freeze/flow retest under vN). The independently measured target (round-5 stencil test, Supplementary Fig. S18, 192×32, log-linear estimator, each stencil normalised by its own neutral anchor) is the von Neumann ridge U_d*/s_d = 0.30 [0.27, 0.40] (Moore on the matched pool: 0.43 [0.28, 0.54]; ratio 0.71). PASS criterion: the inversion's predicted vN ridge (with its bootstrap interval) is consistent with the measured 0.30 [0.27, 0.40]; FAIL: the intervals are disjoint, in which case the identification is falsified and will be reported as such (as the click-count landmark was in round 6).

**Prediction 3 (θ₀ portability).** The vN front census f₀(U_d/s_d) is exponential with the SAME θ₀ = 0.404 (± its interval) — θ₀ is a property of the ratchet under this update rule, not of the stencil. FAIL: vN θ₀ differs from Moore θ₀ outside their joint intervals.

## B. Census across effect size (T3 second ask)

Repeat the least-loaded-class census and the identification at s_d = 0.0025 and s_d = 0.04 (192×32, Moore). Prediction: θ₀ and the opposite-clone contact count are unchanged (both are s_d-free by construction of the rule), and the inversion's predicted ridge tracks the fixed-s_d ridge measured at those rungs (0.76 [wide] at 0.0025; 0.66 at 0.04) at least within the joint intervals. If the inversion predicts a constant ridge while the measured fixed-s_d ridge varies, the discrepancy is reported as a limitation of the census picture, not tuned away.

## C. Ridge-vs-s_d characterization and the headline constant (T2)

1. Homogeneity test across the five fixed-s_d ridge estimates with their own bootstrap variances (Cochran's Q / random-effects τ²), mirroring the area-intensivity test.
2. Crossing-level sensitivity: U_d*/s_d as a function of the crossing level (0.3, 0.4, 0.5, 0.6, 0.7) at each sampled s_d.
3. Minimum detectable slope of ridge/s_d vs log s_d at 80% power, stated instead of "flat but not perfectly flat".
4. Decision rule adopted IN ADVANCE: the quantitative headline for the ridge becomes the pooled (random-effects) constant with its heterogeneity-inflated interval; the s_d = 0.01 crossing is reported as the most precisely measured rung, not as "the" criterion. Every tissue placement is recomputed against the pooled interval AND the shape/aspect correction jointly, using joint worst-case (μ_d low × s_d high and vice versa) rather than independently marginalised bounds.

## D. What would change the paper, and how

- If Prediction 2 fails: the census inversion is downgraded from "derivation" to "post-hoc consistency check", stated in those words, and the constants ledger reverts the ridge prefactor to "measured, not derived".
- If Prediction 2 passes: the derivation stands as a genuine out-of-sample prediction, and the paper says the identification was fixed before the vN test.
- If C.1 rejects homogeneity: the ridge is described as an s_d-dependent profile with its measured shape, not a constant, and tissue margins use the profile at each tissue's own s_d.

Seeds for all round-7 ensembles are drawn from a fresh base (12,000,000) disjoint from every prior campaign, recorded per run.
