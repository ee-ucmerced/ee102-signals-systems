# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Impulse Train via Fourier Series", layout="wide")

st.title("Virtually manipulate to show how sum of sinusoids can approximate impulses?")

with st.expander("The underlying theory:", expanded=True):
    st.markdown(r"""
        In this virtual manipulative, we consider an impulse train:

        $$x(t)=\sum_{m \in \mathbb{Z}}\delta(t-mT),\quad \omega_0=\frac{2\pi}{T}.$$

        Its Fourier series coefficients are

        $$a_k=\frac{1}{T}\ \text{for all}\ k.$$

        The N-term partial sum using the real cosine form is

        $$x_N(t)=\frac{1}{T}\left(1+2\sum_{k=1}^{N}\cos(k\omega_0 t)\right).$$
        """)

with st.sidebar:
    st.header("Controls")
    T = st.number_input("Period T", min_value=0.05, max_value=10.0, value=1.0, step=0.05, format="%.3f")
    N = st.slider("Number of harmonics N", min_value=0, max_value=200, value=5, step=1)
    periods = st.slider("Time window (in periods)", min_value=1, max_value=6, value=3, step=1)
    s_per_T = st.select_slider("Samples per period", options=[256, 512, 1024, 2048, 4096], value=2048)
    show_components = st.checkbox("Show component sinusoids (first few k)", value=True)
    max_comps = st.slider("How many k-components to show", 1, 8, 2) if show_components else 0

# ------------------------- Time grid -------------------------
w0 = 2*np.pi / T
half_win = periods * T / 2
t = np.linspace(-half_win, half_win, int(periods * s_per_T) + 1)

# ------------------------- Partial sum x_N(t) -------------------------
# x_N(t) = (1/T)*(1 + 2 * sum_{k=1..N} cos(k w0 t))
if N == 0:
    xN = np.full_like(t, 1.0 / T)
else:
    ks = np.arange(1, N+1)[:, None]          # shape (N, 1)
    cos_matrix = np.cos(ks * w0 * t[None, :])  # shape (N, len(t))
    xN = (1.0 / T) * (1.0 + 2.0 * np.sum(cos_matrix, axis=0))

# ------------------------- Impulse train for visualization -------------------------
# Visual stems at mT in the window (unit height for display)
m_min = int(np.floor((-half_win) / T)) - 1
m_max = int(np.ceil(( half_win) / T)) + 1
imp_times = T * np.arange(m_min, m_max + 1)
imp_times = imp_times[(imp_times >= t[0]) & (imp_times <= t[-1])]

# ------------------------- Coefficients a_k -------------------------
# |a_k| = 1/T, phase = 0
k_plot = np.arange(-N, N+1)
ak_mag = np.full_like(k_plot, 1.0/T, dtype=float)

# ========================= LAYOUT =========================
col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.4, 2.2])

# ----- Left: Input impulse train -----
with col1:
    st.subheader("$x(t)$: Impulse Train")
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.plot(t, 0*t, lw=1.0, color="black")
    for ti in imp_times:
        ax.vlines(ti, 0, 1.0, colors="tab:red", linewidth=2)
        ax.plot([ti], [1.0], "o", color="tab:red", ms=4)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel("visual stems", fontsize=14)
    ax.set_title(r"$\sum_m \delta(t-mT)$", fontsize=16)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ----- Middle-left: |a_k| -----
with col2:
    st.subheader("Fourier Coefficients $a_k$")
    fig, ax = plt.subplots(figsize=(4, 2.4))
    # Plain stem without use_line_collection
    markerline, stemlines, baseline = ax.stem(k_plot, ak_mag)
    plt.setp(markerline, marker='o', color='C0')
    plt.setp(stemlines, color='C0', linewidth=1.5)
    plt.setp(baseline, color='k', linewidth=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$|a_k|$")
    ax.set_title(r"$|a_k| = 1/T$", fontsize=16)
    ax.set_ylim(0, (1.0/T)*1.2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ----- Middle-right: component sinusoids (first few k) -----
with col3:
    st.subheader("Component Sinusoids")
    if show_components and N > 0:
        ks_show = np.arange(1, min(N, max_comps) + 1)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for k in ks_show:
            comp = (2.0 / T) * np.cos(k * w0 * t)  # amplitude for k-th cosine term
            ax.plot(t, comp, lw=1.2, label=fr"$(2/T) cos({k}\omega_0 t)$")
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("t")
        ax.set_title("First few cosine components (scaled)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, ncol=1, loc="upper right", frameon=True)
        st.pyplot(fig, clear_figure=True)
    elif N == 0:
        st.info("N = 0 implies that only the DC term 1/T exists.")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(t, np.full_like(t, 1.0/T), lw=1.6)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("t")
        ax.set_title("DC component")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Components hidden. (Enable in sidebar)")

# ----- Rightmost: running partial sum x_N(t) -----
with col4:
    st.subheader("Sum of Sinusoids Approximates Impulses")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(t, xN, lw=2.0,
            label=r"$x_N(t) = 1/T ( 1 + 2\sum_{k=1}^{N} \cos(k\omega_0 t) )$")
    for ti in imp_times:
        ax.axvline(ti, color="tab:red", alpha=0.25, lw=1)
    ax.set_xlim(t[0], t[-1])
    ypad = 0.1 * max(1.0, np.max(np.abs(xN)))
    ax.set_ylim(np.min(xN) - ypad, np.max(xN) + ypad)
    ax.set_xlabel("t")
    ax.set_ylabel("partial sum")
    ax.set_title(f"Constructive interference at t = mT (N = {N})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    st.pyplot(fig, clear_figure=True)

# show these notes:
with st.expander("Try the following observations", expanded=True):
    st.markdown(r"""
- Find out the regions of constructive and destructive interference. Near where the impulses are at integer multiples of T you should see constructive interference from the cosines for different k whereas, away from those points, the cosines are out of phase and tend to cancel out.
- Observe the Gibbs phenomenon: near the impulses, the partial sum overshoots and oscillates before settling down.
- As N grows, peaks get taller and narrower, but the area per period stays 1. Try this out!
- How do you get your regular dirac delta function $$\delta(t)$$ from this periodic impulse train? (Hint: let $$T \to \infty$$.) Can you visualize how that looks?
""")
