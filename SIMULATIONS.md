# Simulations and computations in Shepard (1987)

Source: Shepard, R. N. (1987). Toward a universal law of generalization for
psychological science. *Science*, 237(4820), 1317–1323.

The paper is mostly theoretical. Its "simulations" are numerical integrations,
a Monte Carlo study, and nonmetric MDS fits to published data. Each item below
lists what was computed, the figure or equation it belongs to, and what we would
need to reproduce it.

## Core model (used by S3–S6)

- Stimulus at the origin **0** turns out to be consequential. The consequential
  region is convex, centrally symmetric, of unknown size *s* with prior p(s)
  (mean μ), and every location is equally likely.
- Generalization to x (Eq. 2): g(x) = ∫ p(s) · m(s,x)/m(s) ds, where m(s,x) is
  the overlap of the regions centred on 0 and on x.
- 1-D case (Eqs. 5–8): g(d) = ∫_d^∞ p(s)(s − d)/s ds; g′ < 0, g″ = p(d)/d ≥ 0.
- Erlang prior (Eq. 9): p(s) = (2/μ)² s e^(−2s/μ) gives g(d) = exp(−2d/μ) (Eq. 10).

---

## S1. Monte Carlo recovery of metric distances from rank orders

- **Where:** p. 1318, text (ref. 19); no figure.
- **What:** Random configurations of n points in 2-D. Run nonmetric MDS on only
  the rank order of the n(n−1)/2 distances, then correlate the recovered
  distances with the true ones.
- **Reported:** n = 10 gives a mean r of 0.998; n = 45 gives r > 0.999999.
- **To reproduce:** `sklearn.manifold.MDS(metric=False)`, many random trials,
  sweep n. Plot mean r (or 1 − r) against n.

## S2. Nonmetric MDS gives the gradient of generalization (Fig. 1)

- **Where:** Fig. 1, panels A–L, with 12 data sets: sizes, lightness and
  saturation, positions, spectral hues (pigeon and human), consonant and vowel
  phonemes, Morse code, shapes, and others.
- **What:** Convert confusion probabilities p_ij into generalization measures
  g_ij = [(p_ij·p_ji)/(p_ii·p_jj)]^½. Run nonmetric MDS, plot g_ij against the
  fitted distance d_ij, and fit exp(−k·d) using only the slope parameter.
- **Data:** Most original matrices are not in the paper. Data we can get:
  - Ekman (1954) colour similarity (14 hues), widely reproduced.
  - Rothkopf (1957) Morse code confusions (36×36), widely available.
  - Miller & Nicely (1955) consonant confusions, available.
  - Peterson & Barney (1952) vowels, available as raw formants, not confusions.
- **Side results in the text:**
  - Hue: the 1-D continuum bends into a circle in 2-D (Newton's colour circle).
  - Pitch: octave generalization makes the line a helix in 3-D.
  - Orientation of near-symmetric shapes: a Möbius band in 4-D.
  These could be shown as synthetic demos: generate g from the curved geometry,
  then run MDS and see that it recovers the geometry.

## S3. Overlap geometry of consequential regions (Fig. 2)

- **Where:** Fig. 2A and 2B.
- **What:**
  - 2A: the set of centres c of regions that contain both 0 and x is the
    intersection of the two regions centred on 0 and on x.
  - 2B: for a non-convex but centrally symmetric region, trace the locus of
    centres that keep a fixed overlap (about 20%) with the region at 0. This
    locus comes out close to a Euclidean ellipse.
- **To reproduce:** Rasterise the shapes on a grid, compute overlap areas for
  displacement vectors, and draw the iso-overlap contour. An interactive version
  would let the user pick the shape and the overlap level.

## S4. Robustness of g(d) to the size prior p(s) (Fig. 3)

- **Where:** Fig. 3A–F, Eq. 6.
- **What:** Integrate Eq. 6 numerically for six priors, each with mean μ = 1,
  and overlay exp(−d/μ)-type decay:

  | Panel | p(s) | Closed form shown in figure |
  |---|---|---|
  | A | rectangular on [0, 2μ] | 1 − d/2μ + (d/2μ) log(d/2μ), d ≤ 2μ |
  | B | triangular, decreasing, on [0, 3μ] | 1 − (d/3μ)² + 2(d/3μ)² log(d/3μ)… |
  | C | exponential | exp(−d/μ) − (d/μ) Ei(d/μ) |
  | D | triangular, increasing, on [0, 3μ/2] | 1 − 2(2d/3μ) + (2d/3μ)² |
  | E | parabolic on [0, 2μ] | 1 − 3(d/2μ) + 3(d/2μ)² − (d/2μ)³ |
  | F | Erlang (shape 2) | exp(−2d/μ), exact |

- **To reproduce:** Use `scipy.integrate.quad`, with a choice-of-prior widget
  and a slider for μ. Check the numerical curves against the closed forms in the
  figure.

## S5. Bayesian / max-entropy derivation of the Erlang prior

- **Where:** pp. 1320–1321, Eqs. 9–10.
- **What:** The max-entropy prior q(s) with a fixed mean is exponential. Bayes
  with the size-principle likelihood m(s) = s gives
  p(s) ∝ s·q(s), which is Erlang(2), and that yields an exactly exponential g.
- **To reproduce:** Show q(s) → p(s) → g(d) step by step. Optional Monte Carlo:
  sample regions from q, keep the ones that contain 0, and estimate how often
  they contain d.

## S6. Equal-generalization contours in 2-D (Fig. 4)

- **Where:** Fig. 4A–D. Contours for g = 0.1 … 0.9 in one quadrant, with the
  L2 circle and L1 diamond drawn for reference.
  - **A:** square region, one shared size s (correlated dimensions), rectangular
    p(s). Contours are mostly circular (L2-like).
  - **B:** circular region, rectangular p(s). Contours are exactly circular (L2).
  - **C:** rectangular region with independent sizes s and t, both with
    rectangular priors. Contours are rhombic (L1-like).
  - **D:** rectangular region with independent Erlang s and t. Then
    g = exp(−2(|x|+|y|)/μ), which is exactly an L1 rhombus with exponential
    decay.
- **To reproduce:** Evaluate the 2-D Eq. 2 integral on a grid. Where it
  factorises (C, D) it is the product of the 1-D g values. A and B need a 1-D
  integral over s of the overlap fraction. Draw contours with Altair (a
  rect-heatmap plus contour lines precomputed with skimage or matplotlib).

## S7. Proposed extensions (text only, p. 1322)

The paper sketches these but does not carry them out. Each could be an
exploratory demo:

1. **Multiple trials / discrimination learning.** Update the posterior over
   candidate regions from positive and negative examples. This leads to
   Tenenbaum & Griffiths (2001).
2. **Soft (Gaussian) consequence** instead of sharp region boundaries. The paper
   says this still gives an exponential g and the L1/L2 metrics.
3. **Negatively correlated extents** along the dimensions. This gives
   star-shaped contours (Minkowski r < 1) that violate the triangle inequality.
4. **Discrimination reaction time** inversely related to distance, through a
   race of candidate-region "events".
5. **Noise or delayed tests** move g toward a Gaussian (inflected) shape and the
   contours toward ellipses.

---

## Suggested priority for the marimo notebook

| Priority | Item | Why |
|---|---|---|
| ★★★ | S4 (Fig. 3) | Central result, cheap, and works well with interactive widgets |
| ★★★ | S6 (Fig. 4) | The L1/L2 metric story, a striking contour plot |
| ★★☆ | S5 | Explains *why* the curve is exponential; ties S4 together |
| ★★☆ | S1 | Short Monte Carlo that gives the MDS step credibility |
| ★★☆ | S3 (Fig. 2B) | A good geometry visual, more fiddly to build |
| ★☆☆ | S2 (Fig. 1) | Needs external confusion data (Ekman, Rothkopf, and Miller–Nicely are feasible) |
| ★☆☆ | S7 | Open-ended extensions |
