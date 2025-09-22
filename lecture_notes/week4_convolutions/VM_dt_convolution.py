# file: conv_discrete_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Discrete-Time Convolution", layout="wide")

def make_signal(kind, n, **p):
    x = np.zeros_like(n, dtype=float)
    if kind == "Impulse δ[n-n0]":
        n0, A = p["n0"], p["A"]
        x[n == n0] = A
    elif kind == "Rectangular pulse":
        A, start, width = p["A"], p["start"], p["width"]
        x[(n >= start) & (n < start + width)] = A
    elif kind == "Step u[n-n0]":
        A, n0 = p["A"], p["n0"]
        x[n >= n0] = A
    elif kind == "Ramp (finite)":
        start, width, A = p["start"], p["width"], p["A"]
        idx = (n >= start) & (n < start + width)
        x[idx] = A * (n[idx] - start)
    elif kind == "Exponential a^n u[n]":
        a, A = p["a"], p["A"]
        x[n >= 0] = A * (a ** (n[n >= 0]))
    elif kind == "Random sparse":
        np.random.seed(p["seed"])
        x = np.zeros_like(n, dtype=float)
        k = p["k"]
        pos = np.random.choice(n, size=min(k, len(n)), replace=False)
        x[np.isin(n, pos)] = np.random.uniform(-1, 1, size=len(pos))
    return x

def make_h(kind, m, **p):
    h = np.zeros_like(m, dtype=float)
    if kind == "Impulse δ[n]":
        h[m == 0] = p["A"]
    elif kind == "Averager (length L)":
        L = p["L"]
        L = max(1, min(L, len(m)))
        center = 0
        half = L // 2
        idx = (m >= center - half) & (m < center - half + L)
        h[idx] = 1.0 / L
    elif kind == "Exponential decay α^n u[n]":
        alpha = p["alpha"]
        h[m >= 0] = (alpha ** (m[m >= 0]))
        h /= h.sum() if h.sum() != 0 else 1
    elif kind == "3-tap custom [b-1, 1, b+1] at 0":
        b = p["b"]
        h[m == -1] = b - 1
        h[m ==  0] = 1.0
        h[m == +1] = b + 1
        s = h.sum()
        if p["normalize"] and s != 0: h /= s
    return h

# Domain
N_total = st.sidebar.slider("Total discrete horizon (number of n points)", 31, 301, 101, step=2)
nmax = (N_total - 1) // 2
n = np.arange(-nmax, nmax + 1)

# Layout
c1, c2, c3 = st.columns(3, gap="large",
                        vertical_alignment="top")


# Panel 1: x[n]
with c1:
    st.subheader("Input x[n]")
    kind_x = st.selectbox("Form", ["Impulse δ[n-n0]", "Rectangular pulse", "Step u[n-n0]",
                                   "Ramp (finite)", "Exponential a^n u[n]", "Random sparse"])
    params_x = {}
    if kind_x == "Impulse δ[n-n0]":
        params_x["n0"] = st.slider("n0", int(n.min()), int(n.max()), 0)
        params_x["A"] = st.number_input("Amplitude A", value=1.0, step=0.1)
    elif kind_x == "Rectangular pulse":
        params_x["A"] = st.number_input("Amplitude A", value=1.0, step=0.1)
        params_x["start"] = st.slider("start", int(n.min()), int(n.max()), -5)
        params_x["width"] = st.slider("width", 1, 51, 11, step=2)
    elif kind_x == "Step u[n-n0]":
        params_x["A"] = st.number_input("Amplitude A", value=1.0, step=0.1)
        params_x["n0"] = st.slider("n0", int(n.min()), int(n.max()), 0)
    elif kind_x == "Ramp (finite)":
        params_x["A"] = st.number_input("Slope A", value=0.2, step=0.1)
        params_x["start"] = st.slider("start", int(n.min()), int(n.max()), -5)
        params_x["width"] = st.slider("width", 1, 51, 11, step=2)
    elif kind_x == "Exponential a^n u[n]":
        params_x["A"] = st.number_input("Scale A", value=1.0, step=0.1)
        params_x["a"] = st.slider("a", -1.0, 1.0, 0.7, step=0.05)
    elif kind_x == "Random sparse":
        params_x["k"] = st.slider("Nonzeros k", 1, 20, 5)
        params_x["seed"] = st.number_input("Seed", value=0, step=1)
    x = make_signal(kind_x, n, **params_x)
    fig1, ax1 = plt.subplots()
    ax1.stem(n, x)
    ax1.set_xlabel("n")
    ax1.set_ylabel("x[n]")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

# Panel 2: h[n]
with c2:
    st.subheader("Impulse response h[n]")
    kind_h = st.selectbox("Form", ["Impulse δ[n]", "Averager (length L)",
                                   "Exponential decay α^n u[n]", "3-tap custom [b-1, 1, b+1] at 0"])
    m = n  # same index range centered at 0
    params_h = {}
    if kind_h == "Impulse δ[n]":
        params_h["A"] = st.number_input("Gain A", value=1.0, step=0.1)
    elif kind_h == "Averager (length L)":
        params_h["L"] = st.slider("L (odd recommended)", 1, min(51, len(m)), 5, step=2)
    elif kind_h == "Exponential decay α^n u[n]":
        params_h["alpha"] = st.slider("α", 0.0, 1.0, 0.8, step=0.01)
    elif kind_h == "3-tap custom [b-1, 1, b+1] at 0":
        params_h["b"] = st.slider("b", -2.0, 2.0, 0.0, step=0.1)
        params_h["normalize"] = st.checkbox("Normalize sum to 1", value=True)
    h = make_h(kind_h, m, **params_h)
    fig2, ax2 = plt.subplots()
    ax2.stem(m, h)
    ax2.set_xlabel("n")
    ax2.set_ylabel("h[n]")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# Panel 3: y[n] = (x*h)[n]
with c3:
    st.subheader("Output y[n] = (x*h)[n]")
    mode = st.selectbox("Convolution support", ["full", "same", "valid"], index=0)
    y = np.convolve(x, h, mode=mode)
    # Build index for plotting
    if mode == "full":
        k = np.arange(n[0] + m[0], n[-1] + m[-1] + 1)
    elif mode == "same":
        k = n
    else:  # valid
        k = np.arange(n[0] + m[0] + (len(m)-1), n[-1] + m[-1] - (len(m)-1) + 1)
    fig3, ax3 = plt.subplots()
    ax3.stem(k, y)
    ax3.set_xlabel("n")
    ax3.set_ylabel("y[n]")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

# Helpful initial state (obvious convolution):
# Try x: "Rectangular pulse" (A=1, width ~11), h: "Averager (length L)" (L=5).
