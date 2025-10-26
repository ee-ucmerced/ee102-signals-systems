# streamlit_app.py
# EE102: Visualizing FT pairs and reconstruction via partial inverse-FT sums
# Run: streamlit run streamlit_app.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Fourier Transform Visualizer")

# --------------------------
# Helpers
# --------------------------
def sinc(x):
    y = np.ones_like(x)
    m = x != 0
    y[m] = np.sin(x[m]) / x[m]
    return y

def gaussian_window(t, tau):
    # L2-normalized-ish window for visualization (not strict unit energy)
    return np.exp(-(t / tau) ** 2)

def numerical_ft(t, x, w):
    # X(w) = ∫ x(t) e^{-jwt} dt  (Riemann sum)
    dt = t[1] - t[0]
    return np.trapz(x[:, None] * np.exp(-1j * np.outer(t, w)), t, axis=0)

def inverse_ft_partial(w, Xw, t, K):
    # Use only 2K+1 centered frequency samples to synthesize
    # x(t) ~ (Δw / 2π) Σ X(w_m) e^{j w_m t}
    dw = w[1] - w[0]
    mid = len(w) // 2
    idx = np.arange(mid - K, mid + K + 1)
    return (dw / (2 * np.pi)) * np.sum(Xw[idx][None, :] * np.exp(1j * np.outer(t, w[idx])), axis=1)

def energy_time(t, x):
    # ∫ |x(t)|^2 dt
    return np.trapz(np.abs(x) ** 2, t)

def plot_time(ax, t, x, title, xlabel="t", ylabel="x(t)"):
    ax.plot(t, x.real, lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

def plot_freq(ax1, ax2, w, X, title="Fourier Transform X(ω)"):
    ax1.plot(w, np.abs(X), lw=2)
    ax1.set_title(title + " (magnitude)")
    ax1.set_xlabel("ω")
    ax1.set_ylabel("|X(ω)|")
    ax1.grid(True)
    ax2.plot(w, np.angle(X), lw=2)
    ax2.set_title("Phase ∠X(ω)")
    ax2.set_xlabel("ω")
    ax2.set_ylabel("rad")
    ax2.grid(True)

# --------------------------
# Layout
# --------------------------
left, mid, right = st.columns([1.05, 1.15, 1.2])
st.sidebar.header("Global Settings")

# time / frequency grids
T = st.sidebar.slider("Time half-span T (view window is [-T, T])", 2.0, 20.0, 8.0, 0.5)
N_t = st.sidebar.select_slider("Time samples", options=[2048, 4096, 8192], value=4096)
W = st.sidebar.slider("Frequency half-span Ω (plot window is [-Ω, Ω])", 10.0, 200.0, 80.0, 5.0)
N_w = st.sidebar.select_slider("Frequency samples", options=[1024, 2048, 4096], value=2048)

t = np.linspace(-T, T, N_t)
w = np.linspace(-W, W, N_w)

# --------------------------
# Left panel: input signal
# --------------------------
with left:
    st.subheader("Input signal x(t)")
    sig = st.selectbox("Choose signal", ["Delta (with optional shift)", "Rect (width & amp)", "Step u(t - t0)", "Sine", "Cosine"])

    # Defaults
    x = np.zeros_like(t, dtype=float)
    x_label = "x(t)"
    finite_energy = True   # flag for energy bar logic
    analytic_X = None

    if sig == "Delta (with optional shift)":
        A = st.number_input("Amplitude A", value=1.0, step=0.1)
        t0 = st.number_input("Shift t0", value=0.0, step=0.1)
        # Draw as an impulse stem; for internal numeric FT/IFT we use a very narrow Gaussian proxy
        sigma_d = st.slider("Impulse visualization width σ (only for drawing)", 0.005, 0.2, 0.02, 0.005)
        x = np.zeros_like(t)
        # Visualization proxy (do not use for energy bars)
        x_vis = A * np.exp(-(t - t0) ** 2 / (2 * sigma_d ** 2)) / (sigma_d * np.sqrt(2 * np.pi))
        finite_energy = False  # true delta is a distribution; energy undefined

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(t, x_vis, lw=2)
        ax.vlines([t0], 0, [np.max(x_vis)], color="C1", linestyles="--", label="δ(t - t0) visual stem")
        ax.set_xlim(-T, T)
        ax.set_title("Impulse at t0 (Gaussian-drawn proxy)")
        ax.set_xlabel("t")
        ax.set_ylabel("x(t)")
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)

        # Analytic FT: X(ω) = A e^{-j ω t0}
        analytic_X = A * np.exp(-1j * w * t0)

    elif sig == "Rect (width & amp)":
        A = st.number_input("Amplitude A", value=1.0, step=0.1)
        d = st.slider("Width d", 0.2, 6.0, 2.0, 0.1)
        t0 = st.number_input("Center shift t0", value=0.0, step=0.1)
        x = A * ((np.abs(t - t0) <= d / 2).astype(float))
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_time(ax, t, x, "Rectangular pulse")
        st.pyplot(fig, use_container_width=True)

        # Analytic FT: A * d * sinc(ω d/2) * e^{-j ω t0}
        analytic_X = A * d * sinc(0.5 * w * d) * np.exp(-1j * w * t0)

    elif sig == "Step u(t - t0)":
        A = st.number_input("Amplitude A", value=1.0, step=0.1)
        t0 = st.number_input("Step time t0", value=0.0, step=0.1)
        use_window = st.checkbox("Use Gaussian window (finite-energy view for plots and numeric FT)", value=True)
        tau = st.slider("Window time-scale τ (if windowed)", 0.5, 10.0, 3.0, 0.5) if use_window else 3.0
        x = A * (t >= t0).astype(float)
        if use_window:
            x = x * gaussian_window(t - t0, tau)
            finite_energy = True
        else:
            finite_energy = False

        fig, ax = plt.subplots(figsize=(6, 3))
        plot_time(ax, t, x, "Step (windowed for finite-energy view)" if use_window else "Ideal step (energy not finite)")
        st.pyplot(fig, use_container_width=True)

        # For display, compute numerical FT (ideal step has distributional FT)
        analytic_X = numerical_ft(t, x, w)

    elif sig == "Sine":
        A = st.number_input("Amplitude A", value=1.0, step=0.1)
        w0 = st.slider("ω0 (rad/s)", 0.5, 20.0, 6.0, 0.5)
        phi = st.slider("Phase φ (rad)", -np.pi, np.pi, 0.0, 0.1)
        use_window = st.checkbox("Use Gaussian window (finite-energy view)", value=True)
        tau = st.slider("Window time-scale τ (if windowed)", 0.5, 10.0, 3.0, 0.5) if use_window else 3.0
        base = A * np.sin(w0 * t + phi)
        x = base * gaussian_window(t, tau) if use_window else base
        finite_energy = use_window

        fig, ax = plt.subplots(figsize=(6, 3))
        plot_time(ax, t, x, "Sine (windowed for finite-energy view)" if use_window else "Sine (not finite-energy)")
        st.pyplot(fig, use_container_width=True)

        analytic_X = numerical_ft(t, x, w)

    elif sig == "Cosine":
        A = st.number_input("Amplitude A", value=1.0, step=0.1)
        w0 = st.slider("ω0 (rad/s)", 0.5, 20.0, 6.0, 0.5)
        phi = st.slider("Phase φ (rad)", -np.pi, np.pi, 0.0, 0.1)
        use_window = st.checkbox("Use Gaussian window (finite-energy view)", value=True)
        tau = st.slider("Window time-scale τ (if windowed)", 0.5, 10.0, 3.0, 0.5) if use_window else 3.0
        base = A * np.cos(w0 * t + phi)
        x = base * gaussian_window(t, tau) if use_window else base
        finite_energy = use_window

        fig, ax = plt.subplots(figsize=(6, 3))
        plot_time(ax, t, x, "Cosine (windowed for finite-energy view)" if use_window else "Cosine (not finite-energy)")
        st.pyplot(fig, use_container_width=True)

        analytic_X = numerical_ft(t, x, w)

# --------------------------
# Middle panel: FT
# --------------------------
with mid:
    st.subheader("Fourier Transform X(ω)")
    if analytic_X is None:
        Xw = numerical_ft(t, x, w)
    else:
        Xw = analytic_X

    fig, (axm, axp) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    plot_freq(axm, axp, w, Xw, title="X(ω)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# --------------------------
# Right panel: Reconstruction and Energy Bars
# --------------------------
with right:
    st.subheader("Reconstruction from X(ω)")
    Kmax = (len(w) // 2) - 1
    K = st.slider("Number of frequency points per side (partial sum size = 2K+1)", 1, int(Kmax), int(min(50, Kmax)))

    x_rec = inverse_ft_partial(w, Xw, t, K)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t, x_rec.real, lw=2, label="Reconstruction (partial ω-sum)")
    ax.plot(t, x.real, lw=2, ls="--", label="Original x(t)")
    ax.set_title("x(t) vs. partial reconstruction")
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude")
    ax.grid(True)
    ax.legend(loc="upper right")
    st.pyplot(fig, use_container_width=True)

    # Optional multi-axis illustration: show selected cosine packets that sum to x_rec
    st.markdown("**How the pieces add up:** below are example real parts of a few terms being summed.")
    pick = st.slider("Show first M cosine/exponential terms (visual only)", 1, min(12, 2*K+1), 5)
    mid_idx = len(w) // 2
    idx = np.arange(mid_idx - K, mid_idx + K + 1)
    chosen = idx[:pick] if pick <= len(idx) else idx
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    for m in chosen:
        ax2.plot(t, (w[1]-w[0])/(2*np.pi) * (Xw[m] * np.exp(1j*w[m]*t)).real, lw=1, alpha=0.7)
    ax2.set_title(f"Selected {pick} terms of the partial sum (real parts shown)")
    ax2.set_xlabel("t")
    ax2.grid(True)
    st.pyplot(fig2, use_container_width=True)

    # Energy bars
    st.markdown("**Energy progress (Parseval idea)**")
    if finite_energy:
        E_true = energy_time(t, x)
        E_recon = energy_time(t, x_rec.real)
        ratio = float(np.clip(E_recon / E_true if E_true > 0 else 0.0, 0.0, 1.0))
        st.write(f"Estimated original signal energy (time-domain): **{E_true:.3f}**")
        st.progress(ratio, text=f"Recovered energy ≈ {100*ratio:.1f}% of original")
        st.progress(1.0, text="Reference bar = original energy (100%)")
    else:
        st.info("Energy bar disabled: this signal is not finite-energy in the ideal sense (e.g., δ, ideal step, pure sin/cos).")
        st.progress(0.0, text="Recovered energy (N/A)")
        st.progress(0.0, text="Reference bar (N/A)")

# Footer notes
st.caption(
    "Notes: X(ω) uses analytic forms when available (e.g., rect→sinc). "
    "Reconstruction uses a uniform ω-grid and partial inverse-FT sums. "
    "For non–finite-energy signals, an optional Gaussian window is used for visualization."
)
