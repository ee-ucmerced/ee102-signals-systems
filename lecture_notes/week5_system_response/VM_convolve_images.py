# file: conv_colorstrip_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Color Strip Smoothing via Convolution", layout="wide")

# Generate / load 1D color strip (RGB)
st.sidebar.subheader("Input color strip")
N = st.sidebar.slider("Number of pixels", 50, 800, 200, step=10)
mode = st.sidebar.selectbox("Strip type", ["Random", "Two-color halves", "Gradient + noise"])

rng = np.random.default_rng(0)
if mode == "Random":
    x = rng.random((N, 3))
elif mode == "Two-color halves":
    c1 = np.array([0.9, 0.1, 0.1])  # red-ish
    c2 = np.array([0.1, 0.1, 0.9])  # blue-ish
    x = np.vstack([np.tile(c1, (N//2, 1)), np.tile(c2, (N - N//2, 1))])
else:
    grad = np.linspace(0, 1, N)[:, None]
    base = np.hstack([grad, 1-grad, 0.5*np.ones_like(grad)])
    noise = 0.15 * rng.standard_normal((N, 3))
    x = np.clip(base + noise, 0, 1)

# Panel layout
c1, c2, c3 = st.columns(3, gap="large")

def kernel_box(L):
    k = np.ones(L, dtype=float)
    return k / k.sum()

def kernel_gauss(L, sigma):
    n = np.arange(L) - (L-1)/2
    k = np.exp(-0.5*(n/sigma)**2)
    s = k.sum()
    return k / (s if s != 0 else 1)

def kernel_custom3(a, b, c):
    k = np.array([a, b, c], dtype=float)
    s = k.sum()
    return k / (s if s != 0 else 1)

# Panel 1: show input strip
with c1:
    st.subheader("Input strip (x)")
    fig1, ax1 = plt.subplots(figsize=(6, 1.2))
    ax1.imshow(x[None, :, :], aspect="auto")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("x (colors along horizontal axis)")
    st.pyplot(fig1)

# Panel 2: design h[n] (3 parameters)
with c2:
    st.subheader("Design impulse response h[n]")
    kshape = st.selectbox("Kernel shape", ["Box", "Gaussian", "3-tap custom"])
    if kshape == "Box":
        L = st.slider("Length L", 1, 101, 9, step=2)
        h = kernel_box(L)
    elif kshape == "Gaussian":
        L = st.slider("Length L", 3, 101, 15, step=2)
        sigma = st.slider("σ", 0.5, 20.0, 4.0, step=0.5)
        h = kernel_gauss(L, sigma)
    else:
        a = st.slider("a", 0.0, 2.0, 0.5, step=0.05)
        b = st.slider("b", 0.0, 2.0, 1.0, step=0.05)
        c = st.slider("c", 0.0, 2.0, 0.5, step=0.05)
        h = kernel_custom3(a, b, c)

    fig2, ax2 = plt.subplots()
    n = np.arange(len(h)) - (len(h)-1)//2
    ax2.stem(n, h)
    ax2.set_xlabel("n (pixel index)")
    ax2.set_ylabel("h[n]")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# Panel 3: output y = x*h (channel-wise) + uniformity metric
with c3:
    st.subheader("Output y = x * h (goal: all-same color)")
    # Convolve each RGB channel along the pixel axis
    y = np.zeros_like(x)
    for ch in range(3):
        y[:, ch] = np.convolve(x[:, ch], h, mode="same")

    # Show
    fig3, ax3 = plt.subplots(figsize=(6, 1.2))
    ax3.imshow(np.clip(y[None, :, :], 0, 1), aspect="auto")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("y (after filtering)")
    st.pyplot(fig3)

    # Uniformity score: mean squared deviation from the strip mean color
    mean_color = y.mean(axis=0, keepdims=True)
    mse = float(np.mean((y - mean_color) ** 2))
    st.metric("Uniformity loss (lower is better)", f"{mse:.6f}")
    st.write("Tip: box/gaussian with larger support tends to flatten colors. Fine-tune to approach a single color.")

# Initial “obvious” settings:
# - Use "Two-color halves" + Box kernel (L~21) or Gaussian (L~31, σ~6) to see strong smoothing.
