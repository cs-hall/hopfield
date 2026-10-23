# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=5.4",
#     "marimo",
#     "numpy>=2.0",
#     "pandas>=2.2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import math
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    return alt, math, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Hopfield (1982): storing a pattern in the connection matrix

    The network has $N$ binary neurons, each either silent ($V_i = 0$) or
    firing ($V_i = 1$). Every pair of neurons is joined by a connection of
    strength $T_{ij}$, the input neuron $i$ receives from neuron $j$. A pattern
    of activity is stored by adding to $T$ (Eq. 2):

    $$T_{ij} = \sum_s (2V_i^s - 1)(2V_j^s - 1), \qquad T_{ii} = 0.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pair_i = mo.ui.switch(value=True, label="neuron i firing")
    pair_j = mo.ui.switch(value=False, label="neuron j firing")
    return pair_i, pair_j


@app.cell(hide_code=True)
def _(INK, MID, NEG, POS, alt, mo, pair_i, pair_j, pd, signed):
    # One storage step for a single pair: the increment Eq. 2 adds to T_ij
    _vi, _vj = int(pair_i.value), int(pair_j.value)
    _xi, _xj = 2 * _vi - 1, 2 * _vj - 1
    pair_dT = _xi * _xj

    _nodes = pd.DataFrame(
        {
            "x": [0, 1],
            "name": ["i", "j"],
            "state": ["firing (1)" if v else "silent (0)" for v in (_vi, _vj)],
            "ink": ["white" if v else INK for v in (_vi, _vj)],
        }
    )
    _edge = pd.DataFrame(
        {"x": [0], "x2": [1], "label": [f"\u0394T = {signed(pair_dT)}"]}
    )
    _scale = alt.Scale(domain=[-0.35, 1.35])
    _y = alt.value(30)
    pair_chart = (
        alt.layer(
            alt.Chart(_edge)
            .mark_rule(strokeWidth=5, color=POS if pair_dT > 0 else NEG)
            .encode(x=alt.X("x:Q", scale=_scale, axis=None), x2="x2:Q", y=_y),
            alt.Chart(_edge)
            .mark_text(dy=-16, fontSize=13, fontWeight="bold", color=INK)
            .encode(x=alt.value(130), y=_y, text="label:N"),
            alt.Chart(_nodes)
            .mark_circle(size=900, opacity=1, stroke=INK, strokeWidth=1)
            .encode(
                x=alt.X("x:Q", scale=_scale),
                y=_y,
                fill=alt.Fill(
                    "state:N",
                    scale=alt.Scale(
                        domain=["firing (1)", "silent (0)"], range=[POS, MID]
                    ),
                    legend=None,
                ),
            ),
            alt.Chart(_nodes)
            .mark_text(fontSize=14, fontWeight="bold")
            .encode(
                x=alt.X("x:Q", scale=_scale),
                y=_y,
                text="name:N",
                color=alt.Color("ink:N", scale=None),
            ),
        )
        .properties(width=260, height=60)
        .configure_view(stroke=None)
    )

    _meaning = (
        "Same state, so the connection is **strengthened** (excitatory): each "
        "neuron pushes the other toward the state they share."
        if pair_dT > 0
        else "Opposite states, so the connection is **weakened** (inhibitory): "
        "each neuron pushes the other toward the opposite state."
    )
    mo.hstack(
        [
            mo.vstack([pair_i, pair_j], gap=0.5),
            pair_chart,
            mo.md(
                f"$\\Delta T_{{ij}} = (2V_i - 1)(2V_j - 1) = ({_xi:+d})({_xj:+d}) = {pair_dT:+d}$"
                f"\n\n{_meaning}"
            ),
        ],
        justify="start",
        align="center",
        gap=2,
        wrap=True,
    )
    return


@app.cell
def _(n_neurons):
    N = n_neurons.value
    # Heatmap cell size in px, shrinking as the network grows
    cell_px = max(3, min(23, 300 // N))

    # Diverging pair: blue = +1 / firing, red = −1, warm gray = 0 / silent
    POS = "#2a78d6"
    NEG = "#e34948"
    MID = "#f0efec"
    INK = "#0b0b0b"
    MUTED = "#52514e"
    # Categorical order for one line per stored pattern
    SERIES = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ]
    return INK, MID, MUTED, N, NEG, POS, SERIES, cell_px


@app.cell
def _(INK, MID, MUTED, NEG, POS, alt, np, pd):
    def signed(v):
        v = int(v)
        if v > 0:
            return f"+{v}"
        if v < 0:
            return f"−{-v}"
        return "0"

    def ink_for(value, vmax):
        if value == 0:
            return MUTED
        return "white" if abs(value) > vmax / 2 else INK

    def matrix_chart(
        M, title, vmax=None, header=None, outline=None, notes=None, cell=24
    ):
        """Heatmap of an N×N matrix, optionally framed by the vector x along
        the top (x_j) and left (x_i) so each cell reads as their product."""
        n = M.shape[0]
        detailed = n <= 25  # cell values and tick labels only for small networks
        names = [str(k + 1) for k in range(n)]
        vmax = vmax or max(1, int(np.abs(M).max()))
        rows = [
            {
                "i": names[i],
                "j": names[j],
                "value": int(M[i, j]),
                "note": notes[i][j]
                if notes
                else f"T[{i + 1},{j + 1}] = {signed(M[i, j])}",
                "outline": outline is not None and bool(outline[i, j]),
            }
            for i in range(n)
            for j in range(n)
        ]
        x_order, y_order = names, names
        if header is not None:
            rows += [
                {
                    "i": "xⱼ",
                    "j": names[j],
                    "value": int(header[j]),
                    "note": f"x_{j + 1} = {signed(header[j])}",
                    "outline": False,
                }
                for j in range(n)
            ]
            rows += [
                {
                    "i": names[i],
                    "j": "xᵢ",
                    "value": int(header[i]),
                    "note": f"x_{i + 1} = {signed(header[i])}",
                    "outline": False,
                }
                for i in range(n)
            ]
            x_order, y_order = ["xᵢ"] + names, ["xⱼ"] + names
        df = pd.DataFrame(rows)
        df["label"] = df["value"].map(signed)
        df["ink"] = [ink_for(v, vmax) for v in df["value"]]

        base = alt.Chart(df).encode(
            x=alt.X(
                "j:O",
                sort=x_order,
                title="j (sending neuron)",
                axis=alt.Axis(
                    orient="top",
                    labelAngle=0,
                    ticks=False,
                    domain=False,
                    labels=detailed,
                ),
            ),
            y=alt.Y(
                "i:O",
                sort=y_order,
                title="i (receiving neuron)",
                axis=alt.Axis(ticks=False, domain=False, labels=detailed),
            ),
        )
        rect = base.mark_rect(
            stroke="white", strokeWidth=1.5 if detailed else 0
        ).encode(
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(
                    domain=[-vmax, 0, vmax],
                    range=[NEG, MID, POS],
                    interpolate="lab",
                ),
                legend=alt.Legend(title="value", tickCount=3),
            ),
            tooltip=[alt.Tooltip("note:N", title=" ")],
        )
        text = base.mark_text(fontSize=max(7, min(11, cell // 2))).encode(
            text="label:N", color=alt.Color("ink:N", scale=None)
        )
        layers = [rect, text] if detailed else [rect]
        if outline is not None:
            layers.append(
                alt.Chart(df[df["outline"]])
                .mark_rect(filled=False, stroke=INK, strokeWidth=2)
                .encode(x=alt.X("j:O", sort=x_order), y=alt.Y("i:O", sort=y_order))
            )
        return (
            alt.layer(*layers)
            .resolve_scale(color="independent")
            .properties(
                width=cell * len(x_order), height=cell * len(y_order), title=title
            )
        )

    def rows_chart(
        R, row_names, title, signed_values=False, distances=None, cell=24
    ):
        """Each row is a pattern over the neurons: 0/1 (gray/blue) or ±1 (red/blue)."""
        n = R.shape[1]
        df = pd.DataFrame(
            [
                {"row": row_names[r], "j": str(j + 1), "value": int(R[r, j])}
                for r in range(R.shape[0])
                for j in range(n)
            ]
        )
        df["label"] = (
            df["value"].map(signed) if signed_values else df["value"].astype(str)
        )
        df["ink"] = [("white" if v != 0 else MUTED) for v in df["value"]]
        domain, colors = (
            ([-1, 1], [NEG, POS]) if signed_values else ([0, 1], [MID, POS])
        )
        base = alt.Chart(df).encode(
            x=alt.X(
                "j:O",
                sort=[str(j + 1) for j in range(n)],
                title="neuron",
                axis=alt.Axis(
                    orient="top", labelAngle=0, ticks=False, domain=False
                ),
            ),
            y=alt.Y(
                "row:N",
                sort=row_names,
                title=None,
                axis=alt.Axis(ticks=False, domain=False),
            ),
        )
        rect = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
            color=alt.Color(
                "value:O", scale=alt.Scale(domain=domain, range=colors), legend=None
            ),
            tooltip=[
                alt.Tooltip("j:O", title="neuron"),
                alt.Tooltip("value:Q", title="value"),
            ],
        )
        text = base.mark_text(fontSize=max(7, min(11, cell // 2))).encode(
            text="label:N", color=alt.Color("ink:N", scale=None)
        )
        grid = (
            (rect + text)
            .resolve_scale(color="independent")
            .properties(width=cell * n, height=cell * len(row_names))
        )
        if distances is None:
            return grid.properties(title=title)
        # Optional column of per-row values (e.g. Hamming distance) on the right
        ddf = pd.DataFrame(
            {"row": row_names, "col": "Hamming", "d": [int(v) for v in distances]}
        )
        dbase = alt.Chart(ddf).encode(
            x=alt.X(
                "col:N",
                title="distance",
                axis=alt.Axis(
                    orient="top", labelAngle=0, ticks=False, domain=False
                ),
            ),
            y=alt.Y("row:N", sort=row_names, axis=None),
        )
        dcol = (
            dbase.mark_rect(fill=MID, stroke="white", strokeWidth=1.5).encode(
                tooltip=[
                    alt.Tooltip(
                        "d:Q", title="Hamming distance from current pattern"
                    )
                ]
            )
            + dbase.mark_text(fontSize=11, color=INK).encode(text="d:Q")
        ).properties(width=64, height=cell * len(row_names))
        return alt.hconcat(grid, dcol, spacing=8).properties(title=title)

    def network_chart(V, T, size=320, highlight=None):
        """Neurons on a ring (filled = firing); lines are connections, blue for
        excitatory (T > 0), red for inhibitory (T < 0), width ∝ |T|."""
        n = len(V)
        # Shrink nodes so neighbours on the ring don't overlap
        node_size = min(650, 0.5 * (2.6 * size / n) ** 2)
        # Thin and fade the edges as the number of connections grows
        edge_width = max(0.5, 5 * min(1, 13 / n))
        edge_opacity = max(0.08, 0.55 * min(1, 13 / n))
        angle = np.pi / 2 - 2 * np.pi * np.arange(n) / n
        px, py = np.cos(angle), np.sin(angle)
        edges = pd.DataFrame(
            [
                {
                    "x": px[i],
                    "y": py[i],
                    "x2": px[j],
                    "y2": py[j],
                    "w": abs(int(T[i, j])),
                    "sign": "excitatory (+)" if T[i, j] > 0 else "inhibitory (−)",
                    "note": f"T[{i + 1},{j + 1}] = T[{j + 1},{i + 1}] = {signed(T[i, j])}",
                }
                for i in range(n)
                for j in range(i + 1, n)
                if T[i, j] != 0
            ],
            columns=["x", "y", "x2", "y2", "w", "sign", "note"],
        )
        nodes = pd.DataFrame(
            {
                "x": px,
                "y": py,
                "name": [str(k + 1) for k in range(n)],
                "state": ["firing (1)" if v else "silent (0)" for v in V],
                "ink": ["white" if v else INK for v in V],
                "hl": [k == highlight for k in range(n)],
            }
        )
        scale = alt.Scale(domain=[-1.2, 1.2])
        wmax = max(1, int(np.abs(T).max()))
        lines = (
            alt.Chart(edges)
            .mark_rule(opacity=edge_opacity)
            .encode(
                x=alt.X("x:Q", scale=scale, axis=None),
                y=alt.Y("y:Q", scale=scale, axis=None),
                x2="x2:Q",
                y2="y2:Q",
                color=alt.Color(
                    "sign:N",
                    scale=alt.Scale(
                        domain=["excitatory (+)", "inhibitory (−)"],
                        range=[POS, NEG],
                    ),
                    legend=alt.Legend(title="connection"),
                ),
                strokeWidth=alt.StrokeWidth(
                    "w:Q",
                    scale=alt.Scale(domain=[0, wmax], range=[0, edge_width]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("note:N", title=" ")],
            )
        )
        dots = (
            alt.Chart(nodes)
            .mark_circle(size=node_size, opacity=1, stroke=INK)
            .encode(
                strokeWidth=alt.condition("datum.hl", alt.value(4), alt.value(1)),
                x=alt.X("x:Q", scale=scale),
                y=alt.Y("y:Q", scale=scale),
                fill=alt.Fill(
                    "state:N",
                    scale=alt.Scale(
                        domain=["firing (1)", "silent (0)"], range=[POS, MID]
                    ),
                    legend=alt.Legend(title="neuron"),
                ),
                tooltip=[
                    alt.Tooltip("name:N", title="neuron"),
                    alt.Tooltip("state:N", title="state"),
                ],
            )
        )
        labels = (
            alt.Chart(nodes)
            .mark_text(fontSize=12 if n <= 16 else 9, fontWeight="bold")
            .encode(
                x=alt.X("x:Q", scale=scale),
                y=alt.Y("y:Q", scale=scale),
                text="name:N",
                color=alt.Color("ink:N", scale=None),
            )
        )
        return (
            alt.layer(*([lines, dots, labels] if n <= 25 else [lines, dots]))
            .resolve_scale(color="independent")
            .properties(width=size, height=size)
            .configure_view(stroke=None)
        )

    return matrix_chart, network_chart, rows_chart, signed


@app.cell
def _(mo):
    get_stored, set_stored = mo.state([])
    return get_stored, set_stored


@app.cell(hide_code=True)
def _(mo):
    get_net, set_net = mo.state(
        {"cue": None, "state": None, "t": 0, "last": None, "history": None}
    )
    return get_net, set_net


@app.cell
def _(mo):
    mo.md(r"""
    ## The network

    Tick the neurons that should fire, or get a random pattern.

    Pick a pattern below and press **Store** to add it to $T$.

    The network updates one neuron at a time asynchronously. The basic unit of time is $\frac{1}{W}$ (the average number of steps it takes for a neuron to update again). We expect about N single-neuron updates to happen in that time.
    """)
    return


@app.cell
def _(mo):
    n_neurons = mo.ui.number(start=5, stop=12, step=1, value=12, label="Neurons")
    clip_level = mo.ui.number(
        start=0, stop=50, step=1, value=0, label="Clip T to ± (0 = off)"
    )
    mo.hstack([n_neurons, clip_level], justify="start", gap=2)
    return clip_level, n_neurons


@app.cell
def _(mo):
    random_button = mo.ui.button(
        label="Random pattern", value=0, on_click=lambda v: v + 1
    )
    return (random_button,)


@app.cell
def _(N, mo, np, random_button):
    start = np.random.default_rng(1982 + random_button.value).integers(0, 2, N)
    neuron_boxes = mo.ui.array(
        [
            mo.ui.checkbox(value=bool(b), label=str(k + 1))
            for k, b in enumerate(start)
        ]
    )
    return (neuron_boxes,)


@app.cell
def _(N, clip_level, get_stored, neuron_boxes, np):
    V = np.array(neuron_boxes.value, dtype=int)

    # Patterns stored at a different network size are kept but ignored here
    P = np.array([p for p in get_stored() if len(p) == N], dtype=int).reshape(-1, N)

    # Eq. 2, adding one pattern at a time. With a clip level > 0, each T_ij saturates
    # at ±clip_level as it is incremented (the paper's bounded-synapse scheme),
    # so later patterns can overwrite earlier ones.
    T = np.zeros((N, N), dtype=int)
    for p in P:
        T = T + np.outer(2 * p - 1, 2 * p - 1)
        if clip_level.value:
            T = np.clip(T, -clip_level.value, clip_level.value)
    np.fill_diagonal(T, 0)
    return P, T, V


@app.cell(hide_code=True)
def _(np):
    def update_neurons(S, T, n_updates, rng, U=0):
        """Asynchronous dynamics (Eq. 1). Each neuron attempts updates at the same
        mean rate W, independently and at random times, so the sequence of
        attempts is a uniformly random neuron each time (with replacement), one
        at a time. N attempts take about 1/W. A neuron sets V_i = 1 if its input
        sum_j T_ij V_j is above the threshold U and V_i = 0 if below; on a tie
        it keeps its state. Also returns the state after every update."""
        S = S.copy()
        last = None
        trail = []
        for _ in range(n_updates):
            i = int(rng.integers(len(S)))
            h = int(T[i] @ S)
            old = int(S[i])
            if h > U:
                S[i] = 1
            elif h < U:
                S[i] = 0
            last = {"neuron": i, "input": h, "old": old, "new": int(S[i])}
            trail.append(S.copy())
        return S, last, np.array(trail)

    dyn_rng = np.random.default_rng()
    return dyn_rng, update_neurons


@app.cell(hide_code=True)
def _(N, V, get_net):
    net = get_net()
    if net["cue"] == tuple(V) and len(net["state"]) == N:
        S, t_updates, last_update = net["state"], net["t"], net["last"]
        history = net["history"]
    else:
        S, t_updates, last_update = V, 0, None
        history = V[None, :]
    return S, history, last_update, t_updates


@app.cell
def _(V, mo, set_stored):
    store_button = mo.ui.button(
        label="Store pattern",
        kind="success",
        on_click=lambda _: set_stored(lambda s: s + [V.copy()]),
    )
    clear_button = mo.ui.button(
        label="Clear T", kind="danger", on_click=lambda _: set_stored([])
    )
    return clear_button, store_button


@app.cell
def _(
    P,
    S,
    T,
    cell_px,
    clip_level,
    last_update,
    matrix_chart,
    mo,
    network_chart,
):
    mo.hstack(
        [
            network_chart(
                S, T, highlight=last_update["neuron"] if last_update else None
            ),
            matrix_chart(
                T,
                f"Connection matrix T ({len(P)} pattern(s) stored"
                + (
                    f", clipped to ±{clip_level.value})"
                    if clip_level.value
                    else ")"
                ),
                cell=cell_px,
            ),
        ],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell
def _(
    N,
    S,
    T,
    V,
    dyn_rng,
    history,
    last_update,
    mo,
    np,
    set_net,
    t_updates,
    update_neurons,
):
    def _advance(k):
        new_S, last, trail = update_neurons(S, T, k, dyn_rng)
        set_net(
            {
                "cue": tuple(V),
                "state": new_S,
                "t": t_updates + k,
                "last": last,
                "history": np.vstack([history, trail]),
            }
        )

    step_button = mo.ui.button(
        label="Update 1 neuron", on_click=lambda _: _advance(1)
    )
    sweep_button = mo.ui.button(
        label=f"Update {N} neurons (\u2248 1/W)", on_click=lambda _: _advance(N)
    )
    reset_button = mo.ui.button(
        label="Reset to cue",
        on_click=lambda _: set_net(
            {"cue": None, "state": None, "t": 0, "last": None, "history": None}
        ),
    )

    # A state is stable when no neuron's input disagrees with its current value
    _h = T @ S
    n_unstable = int((((_h > 0) & (S == 0)) | ((_h < 0) & (S == 1))).sum())
    if last_update is None:
        _last = "no updates yet"
    else:
        _change = (
            f"flipped {last_update['old']} \u2192 {last_update['new']}"
            if last_update["old"] != last_update["new"]
            else f"stayed {last_update['new']}"
        )
        _last = f"last: neuron {last_update['neuron'] + 1}, input {last_update['input']:+d}, {_change}"
    _stable = (
        "**stable**: no neuron would change"
        if n_unstable == 0
        else f"{n_unstable} neuron(s) would change if picked"
    )
    mo.hstack(
        [
            step_button,
            sweep_button,
            reset_button,
            mo.md(
                f"t = {t_updates} updates ({t_updates / N:.2f} / W) \u00b7 {_last} \u00b7 {_stable}"
            ),
        ],
        justify="start",
        align="center",
        gap=1,
        wrap=True,
    )
    return


@app.cell
def _(
    N,
    P,
    S,
    cell_px,
    clear_button,
    mo,
    neuron_boxes,
    random_button,
    rows_chart,
    store_button,
):
    # Above 13 neurons the checkboxes and pattern grid get unwieldy, so show
    # only the buttons and a list of Hamming distances to the stored patterns
    show_detail = N <= 13
    distances = (P != S).sum(axis=1)  # from the network's current state

    pattern_controls = mo.vstack(
        [
            mo.md("**Cue / pattern to store**"),
            neuron_boxes.hstack(gap=0.75, justify="start")
            if show_detail
            else mo.md(""),
            mo.hstack([random_button, store_button, clear_button], justify="start"),
        ],
        gap=0.75,
    )
    if not len(P):
        stored_view = mo.md("*No patterns stored yet.*")
    elif show_detail:
        stored_view = rows_chart(
            P,
            [f"pattern {k + 1}" for k in range(len(P))],
            "Stored patterns V^s",
            distances=distances,
            cell=cell_px,
        )
    else:
        stored_view = mo.md(
            f"**Hamming distance from the current network state** (out of {N})\n\n"
            "| pattern | distance |\n|---:|---:|\n"
            + "\n".join(f"| {k + 1} | {d} |" for k, d in enumerate(distances))
        )
    mo.hstack(
        [pattern_controls, stored_view],
        justify="start",
        align="start",
        gap=3,
        wrap=True,
    )
    return


@app.cell(hide_code=True)
def _(MUTED, N, P, POS, SERIES, T, alt, history, mo, np, pd):
    # Energy (Eq. 7) and Hamming distance to each stored pattern along the run,
    # measured with the current T. Time is in units of 1/W (N updates).
    trace_time = np.arange(len(history)) / N
    energy = -0.5 * np.einsum("ti,ij,tj->t", history, T, history)
    energy_df = pd.DataFrame(
        {"time": trace_time, "update": np.arange(len(history)), "E": energy}
    )
    _x = alt.X("time:Q", title="time (units of 1/W)")
    energy_chart = (
        alt.Chart(energy_df)
        .mark_line(strokeWidth=2, interpolate="step-after", color=POS)
        .encode(x=_x, y=alt.Y("E:Q", title="energy E", scale=alt.Scale(zero=False)))
        + alt.Chart(energy_df)
        .mark_point(filled=True, size=60, opacity=0)
        .encode(
            x=_x,
            y="E:Q",
            tooltip=["update:Q", alt.Tooltip("time:Q", format=".2f"), "E:Q"],
        )
        + alt.Chart(energy_df.tail(1))
        .mark_point(filled=True, size=50, color=POS, stroke="white", strokeWidth=2)
        .encode(x=_x, y="E:Q")
    ).properties(width=320, height=200, title="Energy")

    if len(P):
        _names = [f"pattern {k + 1}" for k in range(len(P))]
        _dist = (history[:, None, :] != P[None, :, :]).sum(axis=2)
        hamming_df = pd.DataFrame(
            [
                {
                    "time": trace_time[t],
                    "update": t,
                    "pattern": _names[k],
                    "distance": int(_dist[t, k]),
                }
                for t in range(len(history))
                for k in range(len(P))
            ]
        )
        _color = alt.Color(
            "pattern:N",
            sort=_names,
            scale=alt.Scale(
                domain=_names,
                range=[
                    SERIES[k] if k < len(SERIES) else MUTED for k in range(len(P))
                ],
            ),
            legend=alt.Legend(title=None),
        )
        _y = alt.Y(
            "distance:Q", title="Hamming distance", scale=alt.Scale(domain=[0, N])
        )
        hamming_chart = (
            alt.Chart(hamming_df)
            .mark_line(strokeWidth=2, interpolate="step-after")
            .encode(x=_x, y=_y, color=_color)
            + alt.Chart(hamming_df)
            .mark_point(filled=True, size=60, opacity=0)
            .encode(
                x=_x,
                y=_y,
                color=_color,
                tooltip=["pattern:N", "update:Q", "distance:Q"],
            )
        ).properties(width=320, height=200, title="Distance to stored patterns")
    else:
        hamming_chart = mo.md("*Store a pattern to track distances.*")

    mo.hstack([energy_chart, hamming_chart], justify="start", gap=2, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The amount of "energy" in the system is just the amount of disagreement between the current state and the stored weights. More disagreement (two neurons firing that want to inhibit each other) leads to higher energy. This metric can never go up (neurons always update in line with the connection weights).

    But is a lower energy state always to a stored pattern?

    The distance to a stored pattern is shown on the right. Here we see the distance sometimes goes down to 0, but sometimes finds a stable location elsewhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation

    The same dynamics at a larger scale. $n$ random patterns are stored with Eq. 2. The network starts from
    the first pattern with some bits flipped and runs the asynchronous rule. The
    traces show the energy and the Hamming distance over time.
    Fig. 2 of the paper uses $N = 100$ with $n = 5, 10, 15$.

    Note how variable the spread of the stored patterns and the final pattern "chosen" are across runs.
    """)
    return


@app.cell
def _(mo):
    sim_neurons = mo.ui.number(
        start=10, stop=1000, step=10, value=100, label="Neurons N"
    )
    sim_patterns = mo.ui.number(
        start=1, stop=200, step=1, value=10, label="Stored patterns n"
    )
    sim_flips = mo.ui.number(
        start=0, stop=500, step=1, value=10, label="Bits flipped in cue"
    )
    sim_duration = mo.ui.number(
        start=1, stop=50, step=1, value=10, label="Run length (1/W)"
    )
    sim_new_run = mo.ui.button(label="New run", value=0, on_click=lambda v: v + 1)
    mo.hstack(
        [sim_neurons, sim_patterns, sim_flips, sim_duration, sim_new_run],
        justify="start",
        align="end",
        gap=1.5,
        wrap=True,
    )
    return sim_duration, sim_flips, sim_neurons, sim_new_run, sim_patterns


@app.cell(hide_code=True)
def _(np):
    def simulate(T, S0, P, n_updates, rng, U=0):
        """Run the asynchronous rule for n_updates single-neuron updates.

        Keeps the input h = T @ S up to date incrementally, so each update costs
        O(N + n) rather than O(N^2). The energy changes by dE = -dV_i * h_i
        (Eq. 8) and each pattern's Hamming distance changes by +/-1 per flip.
        Returns the energy and the distances (n_updates + 1 rows) and the final state.
        """
        S = S0.astype(np.int64).copy()
        h = T @ S
        E = -0.5 * S @ h
        d = (P != S).sum(axis=1)
        energies = np.empty(n_updates + 1)
        dists = np.empty((n_updates + 1, len(P)), dtype=np.int64)
        energies[0], dists[0] = E, d
        for t, i in enumerate(rng.integers(len(S), size=n_updates), start=1):
            hi = h[i]
            new = 1 if hi > U else (0 if hi < U else S[i])
            if new != S[i]:
                dv = new - S[i]
                E -= dv * hi
                h += dv * T[i]  # T is symmetric, so row i is column i
                d += np.where(P[:, i] == new, -1, 1)
                S[i] = new
            energies[t] = E
            dists[t] = d
        return energies, dists, S

    return (simulate,)


@app.cell(hide_code=True)
def _(
    np,
    sim_duration,
    sim_flips,
    sim_neurons,
    sim_new_run,
    sim_patterns,
    simulate,
):
    sim_rng = np.random.default_rng(sim_new_run.value)
    sim_N = sim_neurons.value
    sim_P = sim_rng.integers(0, 2, (sim_patterns.value, sim_N))
    sim_X = 2 * sim_P - 1
    sim_T = sim_X.T @ sim_X
    np.fill_diagonal(sim_T, 0)

    # Cue: pattern 1 with some bits flipped
    sim_cue = sim_P[0].copy()
    sim_cue[sim_rng.choice(sim_N, min(sim_flips.value, sim_N), replace=False)] ^= 1

    sim_energy, sim_dist, sim_final = simulate(
        sim_T, sim_cue, sim_P, sim_duration.value * sim_N, sim_rng
    )
    return sim_N, sim_P, sim_T, sim_dist, sim_energy, sim_final


@app.cell(hide_code=True)
def _(
    MUTED,
    POS,
    alt,
    mo,
    np,
    pd,
    sim_N,
    sim_P,
    sim_T,
    sim_dist,
    sim_energy,
    sim_final,
):
    # Thin to ~500 time points per line so the charts stay light
    _stride = max(1, len(sim_energy) // 500)
    _idx = np.r_[np.arange(0, len(sim_energy), _stride), len(sim_energy) - 1]
    _time = _idx / sim_N
    _x = alt.X("time:Q", title="time (units of 1/W)")

    sim_energy_df = pd.DataFrame(
        {"time": _time, "update": _idx, "E": sim_energy[_idx]}
    )
    sim_energy_chart = (
        alt.Chart(sim_energy_df)
        .mark_line(strokeWidth=2, interpolate="step-after", color=POS)
        .encode(
            x=_x,
            y=alt.Y("E:Q", title="energy E", scale=alt.Scale(zero=False)),
            tooltip=["update:Q", alt.Tooltip("time:Q", format=".2f"), "E:Q"],
        )
        .properties(width=320, height=200, title="Energy")
    )

    # The cued pattern in blue; the other stored patterns in gray
    _n = len(sim_P)
    sim_dist_df = pd.DataFrame(
        {
            "time": np.repeat(_time, _n),
            "update": np.repeat(_idx, _n),
            "pattern": np.tile(np.arange(1, _n + 1), len(_idx)),
            "distance": sim_dist[_idx].ravel(),
        }
    )
    sim_dist_df["kind"] = np.where(
        sim_dist_df["pattern"] == 1, "cued pattern (1)", "other stored patterns"
    )
    _y = alt.Y(
        "distance:Q", title="Hamming distance", scale=alt.Scale(domain=[0, sim_N])
    )
    _color = alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=["cued pattern (1)", "other stored patterns"], range=[POS, MUTED]
        ),
        legend=alt.Legend(title=None, orient="bottom"),
    )
    _tip = ["pattern:N", "update:Q", "distance:Q"]
    sim_dist_chart = (
        alt.Chart(sim_dist_df[sim_dist_df["pattern"] != 1])
        .mark_line(strokeWidth=1, opacity=0.5, interpolate="step-after")
        .encode(x=_x, y=_y, color=_color, detail="pattern:N", tooltip=_tip)
        + alt.Chart(sim_dist_df[sim_dist_df["pattern"] == 1])
        .mark_line(strokeWidth=2, interpolate="step-after")
        .encode(x=_x, y=_y, color=_color, tooltip=_tip)
    ).properties(width=320, height=200, title="Distance to stored patterns")

    _h = sim_T @ sim_final
    _stable = not (
        ((_h > 0) & (sim_final == 0)) | ((_h < 0) & (sim_final == 1))
    ).any()
    _d1 = int(sim_dist[-1, 0])
    mo.vstack(
        [
            mo.hstack(
                [sim_energy_chart, sim_dist_chart],
                justify="start",
                gap=2,
                wrap=True,
            ),
            mo.md(
                f"Final state: {_d1} bit(s) from pattern 1 ({sim_N - _d1} from its complement), "
                f"nearest stored pattern at distance {int(sim_dist[-1].min())}; "
                + (
                    "**stable**."
                    if _stable
                    else "still changing, so try a longer run."
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reproducing Fig. 2

    The paper's capacity test: store $n$ random memories, start the network
    exactly at each one, let it run until no neuron would change, and count the
    errors (the Hamming distance from the starting memory). The counts are
    pooled over every memory of many random matrices. The paper reports that
    with $N = 100$, $n = 5$ memories are almost always exact, about 0.6 of
    $n = 10$ memories are exact, and $n = 15$ is where recall breaks down.
    """)
    return


@app.cell
def _(mo):
    fig2_neurons = mo.ui.number(
        start=20, stop=500, step=10, value=100, label="Neurons N"
    )
    fig2_patterns = mo.ui.text(
        value="5, 10, 15", label="Stored patterns n (comma-separated)"
    )
    fig2_matrices = mo.ui.number(
        start=1, stop=500, step=1, value=30, label="Random matrices per n"
    )
    fig2_resample = mo.ui.button(
        label="New sample", value=0, on_click=lambda v: v + 1
    )
    mo.hstack(
        [fig2_neurons, fig2_patterns, fig2_matrices, fig2_resample],
        justify="start",
        align="end",
        gap=1.5,
        wrap=True,
    )
    return fig2_matrices, fig2_neurons, fig2_patterns, fig2_resample


@app.cell(hide_code=True)
def _(np):
    def settle(T, S0, rng, max_sweeps=50, U=0):
        """Run the asynchronous rule until no neuron would change (checked after
        every N updates). Gives up after max_sweeps * N updates. Returns the
        final state and whether it is stable."""
        S = S0.astype(np.int64).copy()
        n = len(S)
        h = T @ S
        for _ in range(max_sweeps):
            for i in rng.integers(n, size=n):
                hi = h[i]
                new = 1 if hi > U else (0 if hi < U else S[i])
                if new != S[i]:
                    h += (new - S[i]) * T[i]
                    S[i] = new
            if not (((h > U) & (S == 0)) | ((h < U) & (S == 1))).any():
                return S, True
        return S, False

    return (settle,)


@app.cell(hide_code=True)
def _(fig2_matrices, fig2_neurons, fig2_patterns, fig2_resample, np, settle):
    fig2_N = fig2_neurons.value
    fig2_ns = [int(v) for v in fig2_patterns.value.replace(",", " ").split()]
    fig2_rng = np.random.default_rng(1982 + fig2_resample.value)

    # For each n: the error count of every memory, and how many never settled
    fig2_errors, fig2_unsettled = {}, {}
    for _n in fig2_ns:
        _errs, _bad = [], 0
        for _ in range(fig2_matrices.value):
            _P = fig2_rng.integers(0, 2, (_n, fig2_N))
            _T = (2 * _P - 1).T @ (2 * _P - 1)
            np.fill_diagonal(_T, 0)
            for _mem in _P:
                _final, _ok = settle(_T, _mem, fig2_rng)
                _errs.append(int((_final != _mem).sum()))
                _bad += not _ok
        fig2_errors[_n] = np.array(_errs)
        fig2_unsettled[_n] = _bad
    return fig2_N, fig2_errors, fig2_unsettled


@app.cell(hide_code=True)
def _(POS, alt, fig2_N, fig2_errors, fig2_unsettled, math, mo, np, pd):
    # The paper's bins: single counts 0-9, then decades, then everything >= N/2
    _half = fig2_N // 2
    fig2_bins = [(k, k, str(k)) for k in range(10)]
    fig2_bins += [
        (lo, min(lo + 9, _half - 1), f"{lo}–{min(lo + 9, _half - 1)}")
        for lo in range(10, _half, 10)
    ]
    fig2_bins += [(_half, fig2_N, f">{_half - 1}")]
    _order = [b[2] for b in fig2_bins]

    def _eq10_no_errors(n, N):
        """Eq. 10: chance that all N bits of a memory start out stable."""
        if n < 2:
            return 1.0
        sigma = np.sqrt((n - 1) * N / 2)
        p_bit = 0.5 * math.erfc((N / 2) / (sigma * np.sqrt(2)))
        return (1 - p_bit) ** N

    _panels, _stats = [], {}
    for _n, _errs in fig2_errors.items():
        _df = pd.DataFrame(
            {
                "bin": [b[2] for b in fig2_bins],
                "probability": [
                    ((_errs >= lo) & (_errs <= hi)).mean()
                    for lo, hi, _ in fig2_bins
                ],
                "count": [
                    int(((_errs >= lo) & (_errs <= hi)).sum())
                    for lo, hi, _ in fig2_bins
                ],
            }
        )
        _panels.append(
            alt.Chart(_df)
            .mark_bar(
                color=POS,
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4,
                width=alt.RelativeBandSize(0.7),
            )
            .encode(
                x=alt.X(
                    "bin:O",
                    sort=_order,
                    title="Nerr = number of errors in state",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "probability:Q",
                    title="probability",
                    scale=alt.Scale(type="sqrt", domain=[0, 1]),
                    axis=alt.Axis(values=[0.05, 0.1, 0.2, 0.5, 1]),
                ),
                tooltip=[
                    "bin:O",
                    alt.Tooltip("probability:Q", format=".3f"),
                    "count:Q",
                ],
            )
            .properties(width=440, height=120, title=f"n = {_n}, N = {fig2_N}")
        )
        _stats[_n] = [
            str(len(_errs)),
            f"{(_errs == 0).mean():.2f}",
            f"{(_errs < 5).mean():.2f}",
            f"{_eq10_no_errors(_n, fig2_N):.2f}",
            str(fig2_unsettled[_n]),
        ]

    # Transposed summary: one narrow column per n so it fits beside the panels
    _rows = [
        "memories tested",
        "P(0 errors)",
        "P(< 5 errors)",
        "Eq. 10: P(all bits start stable)",
        "never settled",
    ]
    _ns = list(_stats)
    _table = (
        "| | " + " | ".join(f"n = {_n}" for _n in _ns) + " |\n"
        "|---|"
        + "---:|" * len(_ns)
        + "\n"
        + "\n".join(
            f"| {_label} | " + " | ".join(_stats[_n][_r] for _n in _ns) + " |"
            for _r, _label in enumerate(_rows)
        )
    )

    mo.hstack(
        [
            alt.vconcat(*_panels, spacing=12),
            mo.md(
                _table
                + "\n\nThe y-axis uses a square-root scale, which matches the spacing of the paper's ticks."
            ),
        ],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An error in a Nobel Prize-winning article

    Eq. 10 predicts that 40% of memories will be stable states (given N=100 and n=10). Their simulations (and ours) show 60%. Why is that?

    Each bit is unstable with probability 0.009 (from the paper). So if each bit is independent then 40% of memories should be stable.
    However, each bit is not independent. The average error rate is 0.009 per bit, but that is stronger for memories that are very close to other memories. So if some memories have more errors than expected, others must have fewer (so the average still works out).
    """)
    return


@app.cell(hide_code=True)
def _(np):
    # Counterexample: the same unstable-bit statistics, with and without each
    # memory's neighbourhood. N = 100, n = 10 as in the paper's example.
    _rng = np.random.default_rng(10)
    counter_N, counter_n = 100, 10
    _bad = []
    for _ in range(2000):
        _P = _rng.integers(0, 2, (counter_n, counter_N))
        _T = (2 * _P - 1).T @ (2 * _P - 1)
        np.fill_diagonal(_T, 0)
        _H = _P @ _T  # input to every bit of every stored memory (T is symmetric)
        _bad.append(((_H > 0) & (_P == 0)) | ((_H < 0) & (_P == 1)))
    counter_unstable = np.concatenate(
        _bad
    )  # one row per memory, one column per bit
    counter_p_bit = counter_unstable.mean()

    # Real memories keep their own N bits together
    counter_real = counter_unstable.sum(axis=1)
    # Synthetic memories take each of their N bits from a different random memory
    _flat = counter_unstable.ravel()
    counter_synth = _flat[
        _rng.integers(len(_flat), size=(len(counter_real), counter_N))
    ].sum(axis=1)
    return counter_N, counter_n, counter_p_bit, counter_real, counter_synth


@app.cell(hide_code=True)
def _(
    MUTED,
    SERIES,
    alt,
    counter_N,
    counter_n,
    counter_p_bit,
    counter_real,
    counter_synth,
    math,
    mo,
    np,
    pd,
):
    _sigma = np.sqrt((counter_n - 1) * counter_N / 2)
    _p_eq10 = 0.5 * math.erfc((counter_N / 2) / (_sigma * np.sqrt(2)))
    _binom_var = counter_N * counter_p_bit * (1 - counter_p_bit)

    _table = mo.md(
        f"""
    | | P(no unstable bits) | variance of # unstable bits |
    |---|---:|---:|
    | Real memories (a memory's {counter_N} bits together) | **{(counter_real == 0).mean():.2f}** | {counter_real.var():.2f} |
    | Synthetic memories (each bit from a different memory) | **{(counter_synth == 0).mean():.2f}** | {counter_synth.var():.2f} |
    | Independent bits: $(1 - p)^N$ with measured $p = {counter_p_bit:.4f}$ | {(1 - counter_p_bit) ** counter_N:.2f} | {_binom_var:.2f} |
    | Eq. 10: $(1 - P)^N$ with $P = {_p_eq10:.4f}$ | {(1 - _p_eq10) ** counter_N:.2f} | |

    Both kinds of memory have the same average of {counter_real.mean():.2f} unstable bits.
    In real memories the errors clump onto memories with close neighbours, so more
    memories come out perfect. Synthetic memories break up those neighbourhoods, and
    Eq. 10's number comes back.
    """
    )

    _cap = 6  # counts of 6 or more share the last bar
    _kinds = ["real", "synthetic"]
    _counts_df = pd.DataFrame(
        [
            {
                "unstable bits": k,
                # Numeric x with a manual offset per group, so the mean lines can
                # sit at fractional positions on the same axis
                "x": k + (-0.18 if _kind == "real" else 0.18),
                "memories": _kind,
                "fraction": (np.minimum(_c, _cap) == k).mean(),
            }
            for _kind, _c in [("real", counter_real), ("synthetic", counter_synth)]
            for k in range(_cap + 1)
        ]
    )
    _means_df = pd.DataFrame(
        {"memories": _kinds, "mean": [counter_real.mean(), counter_synth.mean()]}
    )
    _color = alt.Color(
        "memories:N",
        sort=_kinds,
        scale=alt.Scale(domain=_kinds, range=SERIES[:2]),
        legend=alt.Legend(title=None, orient="top"),
    )
    _x_scale = alt.Scale(domain=[-0.6, _cap + 0.6], nice=False)
    _bars = (
        alt.Chart(_counts_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=14)
        .encode(
            x=alt.X(
                "x:Q",
                scale=_x_scale,
                title="unstable bits per memory",
                axis=alt.Axis(
                    values=list(range(_cap + 1)),
                    labelExpr=f"datum.value == {_cap} ? '{_cap}+' : datum.value",
                    grid=False,
                ),
            ),
            y=alt.Y("fraction:Q", title="fraction of memories"),
            color=_color,
            tooltip=[
                "memories:N",
                "unstable bits:Q",
                alt.Tooltip("fraction:Q", format=".3f"),
            ],
        )
    )
    # Mean of each group: real solid, synthetic dashed so both show when they coincide
    _mean_lines = (
        alt.Chart(_means_df)
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("mean:Q", scale=_x_scale),
            color=_color,
            strokeDash=alt.StrokeDash(
                "memories:N",
                sort=_kinds,
                scale=alt.Scale(domain=_kinds, range=[[1, 0], [6, 4]]),
                legend=None,
            ),
            tooltip=[
                "memories:N",
                alt.Tooltip("mean:Q", format=".2f", title="mean"),
            ],
        )
    )
    _mean_label = (
        alt.Chart(_means_df.head(1))
        .mark_text(align="left", dx=6, dy=4, baseline="top", color=MUTED)
        .encode(
            x=alt.X("mean:Q", scale=_x_scale),
            y=alt.value(0),
            text=alt.value(f"mean = {counter_real.mean():.2f} (both)"),
        )
    )
    _chart = (_bars + _mean_lines + _mean_label).properties(
        width=360,
        height=200,
        title="Real memories have more zeros and a longer tail",
    )
    mo.hstack([_chart, _table], justify="start", align="center", gap=2)
    return


if __name__ == "__main__":
    app.run()
