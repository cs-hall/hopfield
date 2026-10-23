# Simulations in Hopfield (1982)

Source: Hopfield, J. J. (1982). Neural networks and physical systems with
emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.

The paper has one data figure (Fig. 2). Most results are Monte Carlo numbers
reported in the text. Each item below gives what was simulated, where it
appears, what was reported, and how to reproduce it.

## Core model (used by every item)

- **Units:** N binary neurons, V_i ∈ {0, 1}. Thresholds U_i = 0 unless stated.
- **Dynamics (Eq. 1):** asynchronous and random. Each neuron is picked at mean
  rate W and set to V_i = 1 if Σ_j T_ij V_j > U_i, else 0. One unit of time
  1/W is about N single-neuron updates (one "sweep").
- **Storage (Eq. 2):** T_ij = Σ_s (2V_i^s − 1)(2V_j^s − 1), with T_ii = 0.
- **Energy (Eqs. 7–8):** E = −½ Σ_{i≠j} T_ij V_i V_j, and
  ΔE = −ΔV_i Σ_j T_ij V_j ≤ 0 when T is symmetric.
- **Memories** are random binary vectors with p(1) = ½. The complement of each
  memory is also stable, so n memories give 2n nominal attractors.
- **Stopping rule:** stop when a full pass finds no neuron that wants to flip,
  or at a cap of about 50 sweeps, and label the run "not converged".

One vectorised numpy engine (many trials at once, random update order, optional
clamped bits, optional thresholds) covers all the items below.

---

## S1. Energy descent and stable states (Eqs. 7–8)

- **Where:** p. 2556, "The model has stable limit points."
- **What:** With symmetric T, every asynchronous update leaves E the same or
  lowers it, so the run must end at a local minimum. The paper links this to
  Ising models and spin glasses.
- **To reproduce:** Trace E(t) for several runs on a Hebbian T and on a random
  symmetric T, and check that ΔE ≤ 0 at every step. A second view: enumerate
  every state for small N (≈ 16) and count the local minima.

## S2. Random asymmetric T_ij (N = 30, N = 100)

- **Where:** p. 2556, Eq. 9.
- **What:** T_ij ~ U(−1, 1) with no symmetry. Start from random states.
- **Reported (N = 30):**
  - Runs never wander ergodically. They settle within about 4/W.
  - In 50 trials on one matrix, every run ends in one of 2–3 end states.
  - Simple cycles A → B → A → B appear now and then.
  - Some runs wander chaotically within a short Hamming distance of one state.
    The entropy measure ln M = −Σ p_i ln p_i (Eq. 9) gives M ≈ 25.
  - N = 100 behaves the same way qualitatively.
- **To reproduce:** For each run, record the end state and how long it took.
  Classify the outcome as fixed point, cycle, or local wandering. For wandering
  runs, collect visit frequencies over a long window and compute M. An optional
  slider for T = T_sym + λ·T_antisym shows the "effective temperature" argument
  on p. 2556.

## S3. Capacity: error distribution of recalled memories (Fig. 2)

- **Where:** Fig. 2, and the p. 2556 text "About 0.15 N states…".
- **What:** N = 100 with n = 5, 10, 15 random memories. Start at each stored
  memory, run until stationary, and histogram the Hamming distance to the
  original. Average over memories and over many matrices.
- **Bins:** 0, 1, …, 9, then 10–19, 20–29, 30–39, 40–49, and > 49.
- **Reported:**
  - n = 5: almost always exact.
  - n = 10: about 0.6 of states have zero errors.
  - n = 15: about half end with fewer than 5 errors. The rest land 20–40 bits
    away.
- **Also:** Repeat with N = 30 to show that the critical n scales with N. Sweep
  n/N and plot the fraction recalled with fewer than 5 errors. The curve should
  drop near n ≈ 0.15 N.
- **Note:** The y-axis in Fig. 2 is not linear, so match the panels by eye.

## S4. Signal-to-noise theory (Eqs. 3, 4, 10)

- **Where:** p. 2556.
- **What:** The effective field on a bit of a stored memory has mean ±N/2 and
  Gaussian crosstalk noise σ = √((n − 1)N/2). The single-bit error probability
  is P = Φ̄((N/2)/σ).
- **Reported:** For n = 10, N = 100, P = 0.0091, and P(no errors in 100 bits)
  ≈ e^(−0.91) ≈ 0.40. The simulation gave 0.6.
- **To reproduce:**
  - Plot the analytic P against n.
  - Overlay the measured one-step instability: the fraction of bits whose field
    has the wrong sign at the memory.
  - Overlay the measured final zero-error fraction from S3.
  - The gap between 0.40 and 0.6 is worth showing. The theory counts bits that
    are unstable at the start. The simulation counts states that are still
    exact after the network has relaxed.

## S5. ±1 coding / thresholds double capacity

- **Where:** p. 2557, top of the left column.
- **What:** In 0/1 coding the field is ½ Σ T_ij μ_j + ½ Σ T_ij, and the second
  term adds noise. Choosing thresholds U_i = ½ Σ_j T_ij, which is the same as
  using μ = ±1 states with threshold 0, removes that term. At a fixed error
  rate this doubles capacity.
- **To reproduce:** Rerun the S3 sweep in both codings and overlay the two
  capacity curves.

## S6. Basins of attraction and content-addressable recall (N = 30, n = 5)

- **Where:** p. 2557.
- **S6a, random starts:**
  - Reported: about 85% of runs end in assigned memories (10 including
    complements), 10% end in spurious stable states, and 5% end in states very
    near a memory.
  - Reported: the 10 states differ in likelihood by about a factor of 20.
  - To reproduce: tabulate the end-state categories and the frequency of each
    memory.
- **S6b, distance-dependent recall:**
  - Corrupt a memory by d random flips. Measure P(end state is the nearest
    memory) as a function of d.
  - Reported: above 0.9 for d ≤ 5, falling smoothly to about 0.2 (2× chance)
    at d = 12.
- **S6c, recall from a subpart (intro example):** Give part of a memory, either
  with the rest random or with the known bits clamped. The paper suggests
  clamping for hardware. Compare recall rates with and without clamping as the
  known fraction grows.

## S7. Clipped synapses: T_ij → sign(T_ij)

- **Where:** p. 2557.
- **What:** Replace T with its sign.
- **Reported:**
  - At N = 100, clipped storage with n = 9 has about the same error level as
    standard storage with n = 12.
  - Theory: the signal-to-noise ratio drops by √(2/π), so capacity drops by a
    factor of 2/π.
  - Shannon information stored peaks near n = 13 at about N²/8 bits. This is
    for the ±1 algorithm with clipped T.
- **To reproduce:**
  - Overlay the S3-style curves for clipped and standard T.
  - Compute information as n·N·(1 − H₂(p_err)) from the measured per-bit error
    rate and plot it against n to find the peak.

## S8. Overload and forgetting through bounded synapses

- **Where:** p. 2557.
- **What:**
  - Unbounded T: storing past capacity makes every memory irretrievable.
  - Bounded T_ij ∈ {0, ±1, ±2, ±3}, with increments that saturate: only recent
    memories stay stable, so the network forgets gracefully. The paper calls
    ±3 a suitable depth for N = 100.
- **To reproduce:** Store a long stream of memories one at a time. After each
  one, test the stability of every earlier memory. Plot recall against memory
  age for both schemes, with a slider for the bound depth.

## S9. One-way connections and fail-soft behaviour

- **Where:** p. 2557.
- **What:** For each pair, keep only one direction: if T_ij ≠ 0 then T_ji = 0.
- **Reported:**
  - Errors increase, but the network still forms stable minima.
  - Theory: the signal-to-noise ratio drops by 1/√2, and the simulations agree.
  - As more synapses fail, the network degrades slowly ("soft" failure).
- **To reproduce:**
  - Compare capacity curves for full T and one-way T.
  - Sweep the fraction of randomly deleted synapses, 0 → 0.9, and plot recall.
    The curve should fall smoothly rather than collapse.

## S10. Similar memories merge

- **Where:** p. 2557.
- **What:** N = 100, n = 8. Seven memories are random. The eighth is placed a
  Hamming distance h = 30, 20, or 10 from one of the seven. Random pairs are
  normally 50 ± 5 apart.
- **Reported:**
  - h = 30: both memories usually stable.
  - h = 20: the minima are distinct but displaced.
  - h = 10: the minima are often fused.
- **To reproduce:** Start at each of the two close memories. Record where each
  run ends and how far it is from each memory. Sweep h continuously, 0–50.

## S11. Familiarity through the 00…0 state (thresholds)

- **Where:** p. 2557.
- **What:** The all-zero state is always stable. With U = 0 it has much higher
  energy than the stored memories and is rarely reached. A uniform threshold
  U > 0 makes 00…0 a common attractor for unfamiliar inputs, which amounts to
  saying "not recognised".
- **To reproduce:** Use two kinds of starting state: noisy memories (familiar)
  and random states (unfamiliar). Use a slider for U. Plot P(end at 00…0) for
  each kind and the resulting ROC.

## S12. Familiarity under heavy overload (N = 100, n = 500)

- **Where:** p. 2558.
- **What:** With 25× overload, no stored memory is stable.
- **Reported:** The initial processing rate tells familiar from unfamiliar
  states most of the time. The rate is the number of neuron flips in time
  1/(2W), about N/2 updates, and it is faster for unfamiliar states.
- **To reproduce:**
  - Start from stored memories and from random states.
  - Count flips over the first N/2 updates.
  - Plot both distributions and the d′ or AUC.

## S13. Completion from correlated memories (Eqs. 11–12)

- **Where:** p. 2558.
- **What:**
  - Store many correlated memories, which gives a mean correlation C_ij ≠ 0.
  - Then store a new partial state X on only k < N neurons (Eq. 11).
  - Recall fills in the missing bits from the sign of Σ_{j≤k} C_ij X_j
    (Eq. 12). This is a form of generalization.
  - The paper says this works best when the correlated set is stored weakly
    and X is stored at normal strength.
- **To reproduce:** The paper does not specify this setup. One design:
  - Generate memories from a few prototypes plus noise.
  - Store them with a weight ε < 1.
  - Store X on its first k bits.
  - Measure how well the remaining bits match the class prototype, as a
    function of ε and k.

## S14. Time sequences from asymmetric terms (Eq. 13)

- **Where:** p. 2558.
- **What:** Add ΔT_ij = A Σ_s (2V_i^(s+1) − 1)(2V_j^s − 1). This is
  nonsymmetric, so a minimum can become metastable and hand the state on to
  the next memory in the sequence.
- **Reported:** With A tuned, the network stays near V_s for a while and then
  moves to V_(s+1). Sequences longer than 4 states could not be produced, and
  even short ones were not followed faithfully.
- **To reproduce:** Plot the overlap of the state with each V_s over time,
  with sliders for A and sequence length. Count how many transitions are
  followed in order. Expect failure beyond about 4 states, because the dynamics
  have no delay term.

## S15. Nonlinear choice vs a linear associator

- **Where:** p. 2555.
- **What:** A linear net given the mixed stimulus 0.6 S₁ + 0.4 S₂ outputs the
  mixture 0.6 O₁ + 0.4 O₂. The Hopfield net "makes a choice" and returns S₁.
- **To reproduce:** Sweep the mixing weight from 0 to 1. Plot the overlap with
  S₁ and with S₂ for the linear readout T·x and for the attractor end state.
  The linear curve should be a ramp and the attractor curve a step near 0.5.

Fig. 1 (the firing-rate curve and its step and linear approximations) is
conceptual. It needs no simulation, but it could be drawn as a static
illustration.

---

## Suggested priority for the marimo notebook

| Priority | Item | Why |
|---|---|---|
| ★★★ | S3 (Fig. 2) + S4 | The one data figure and its theory; the capacity result is ≈ 0.15 N |
| ★★★ | S6 | Content-addressable memory and basins, the paper's headline claim |
| ★★★ | S1 | Lyapunov/energy argument; short and builds intuition |
| ★★☆ | S5, S7, S9 | Robustness variants; they reuse the S3 sweep code |
| ★★☆ | S15 | One-panel contrast that motivates the nonlinearity |
| ★★☆ | S10, S11, S12 | Categorization and familiarity; cheap to run |
| ★☆☆ | S2 | Asymmetric dynamics; classifying cycles and wandering is fiddly |
| ★☆☆ | S8 | Forgetting with bounded synapses; needs long storage streams |
| ★☆☆ | S13, S14 | Loosely specified, and S14 is a reported partial failure |
