# file: conv_colorstrip_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Color Strip via Convolution", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def kernel_box(L: int):
    k = np.ones(L, dtype=float)
    return k / k.sum()

def kernel_gauss(L: int, sigma: float):
    n = np.arange(L) - (L - 1) / 2
    k = np.exp(-0.5 * (n / sigma) ** 2)
    s = k.sum()
    return k / (s if s != 0 else 1.0)

def to_rgb_from_gray(g):
    """g: (N,) in [0,1] -> (N,3) replicated RGB"""
    return np.repeat(g[:, None], 3, axis=1)

def parse_numeric_list(txt: str):
    # Split on commas/space/newlines; keep digits, dots, minus
    raw = txt.replace(",", " ").split()
    vals = []
    for tok in raw:
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return np.array(vals, dtype=float)

def fit_to_length(arr: np.ndarray, N: int, pad_value: float = 0.0):
    if len(arr) == N:
        return arr
    if len(arr) > N:
        return arr[:N]
    if len(arr) == 0:
        return np.full(N, pad_value, dtype=float)
    # pad with last value
    return np.concatenate([arr, np.full(N - len(arr), arr[-1], dtype=float)])

# -----------------------------
# Sidebar: strip source
# -----------------------------
st.sidebar.subheader("Input color strip")

# Global N control: 25..100 (default 50)
N = st.sidebar.slider("Number of pixels (N)", 25, 100, 50, step=1)

strip_type = st.sidebar.selectbox(
    "Strip type",
    [
        "Manual (B/W 0–1)",          # default
        "Manual (0–255 grayscale)",
        "Black & white (random)",
        "Gradient + noise",
        "Random RGB",
    ],
    index=0,  # default selection
)

rng = np.random.default_rng(0)

# -----------------------------
# Build x (shape N x 3 in [0,1])
# -----------------------------
if strip_type == "Manual (B/W 0–1)":
    st.sidebar.caption("Enter N values in [0,1] (comma/space/newline separated).")
    default_bw = " ".join(["0"] * (N // 2) + ["1"] * (N - N // 2))
    txt = st.sidebar.text_area("Values (B/W)", value=default_bw, height=120)
    g = parse_numeric_list(txt)
    g = np.clip(fit_to_length(g, N, pad_value=0.0), 0.0, 1.0)
    x = to_rgb_from_gray(g)

elif strip_type == "Manual (0–255 grayscale)":
    st.sidebar.caption("Enter N integer values in [0,255] (comma/space/newline separated).")
    default_255 = " ".join(["0"] * (N // 2) + ["255"] * (N - N // 2))
    txt = st.sidebar.text_area("Values (0–255)", value=default_255, height=120)
    g255 = parse_numeric_list(txt)
    g255 = np.clip(np.rint(fit_to_length(g255, N, pad_value=0.0)), 0, 255)
    g = (g255 / 255.0).astype(float)
    x = to_rgb_from_gray(g)

elif strip_type == "Black & white (random)":
    g = rng.integers(0, 2, size=N).astype(float)
    x = to_rgb_from_gray(g)

elif strip_type == "Gradient + noise":
    grad = np.linspace(0, 1, N)[:, None]
    base = np.hstack([grad, 1 - grad, 0.5 * np.ones_like(grad)])
    noise = 0.15 * rng.standard_normal((N, 3))
    x = np.clip(base + noise, 0, 1)

else:  # "Random RGB"
    x = rng.random((N, 3))

# -----------------------------
# Layout
# -----------------------------
c1, c2, c3 = st.columns([1.0, 1.0, 1.0], gap="large")

# -----------------------------
# Panel 1: Input strip (x) + pixel values
# -----------------------------
with c1:
    st.subheader("Input strip x[n]")
    fig1, ax1 = plt.subplots(figsize=(6, 1.2))
    ax1.imshow(x[None, :, :], aspect="auto")
    ax1.set_xticks([])
    ax1.set_yticks([])
    st.pyplot(fig1, use_container_width=True)

    # Show actual pixel values at the bottom
    with st.expander("Pixel values", expanded=True):
        if strip_type in ["Manual (B/W 0–1)", "Manual (0–255 grayscale)", "Black & white (random)"]:
            # grayscale values
            if strip_type == "Manual (0–255 grayscale)":
                st.write(g255.astype(int).tolist())
            else:
                st.write((x[:, 0]).round(4).tolist())
        else:
            # compact RGB table
            arr = np.column_stack([np.arange(N), x[:, 0], x[:, 1], x[:, 2]])
            st.dataframe(
                {
                    "idx": arr[:, 0].astype(int),
                    "R": np.round(arr[:, 1], 4),
                    "G": np.round(arr[:, 2], 4),
                    "B": np.round(arr[:, 3], 4),
                },
                use_container_width=True,
                hide_index=True,
            )

# -----------------------------
# Panel 2: Choose h[n]
# -----------------------------
with c2:
    st.subheader("Impulse response h[n]")

    h_choice = st.selectbox(
        "Choose h[n]",
        ["First difference", "Box blur", "Gaussian"],
        index=0,
    )

    # Ensure kernel length never exceeds N (prevents np.convolve broadcasting issues)
    if h_choice == "First difference":
        h = np.array([1.0, -1.0])  # h[n] = δ[n] − δ[n−1]
        L = len(h)
    elif h_choice == "Box blur":
        L = st.slider("Box length L (odd)", 3, min(101, N if N % 2 == 1 else N - 1), min(9, N if N % 2 == 1 else N - 1), step=2)
        L = max(3, min(L, N if N % 2 == 1 else N - 1))
        h = kernel_box(L)
    else:  # Gaussian
        L = st.slider("Gaussian length L (odd)", 5, min(101, N if N % 2 == 1 else N - 1), min(15, N if N % 2 == 1 else N - 1), step=2)
        L = max(5, min(L, N if N % 2 == 1 else N - 1))
        sigma = st.slider("σ", 0.5, 20.0, 4.0, step=0.5)
        h = kernel_gauss(L, sigma)

    # Plot h[n]
    fig2, ax2 = plt.subplots()
    n = np.arange(len(h)) - (len(h) - 1) // 2
    ax2.stem(n, h)
    ax2.set_xlabel("n")
    ax2.set_ylabel("h[n]")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, use_container_width=True)

    # Show equations for the discrete-time kernels (except Gaussian per request)
    if h_choice == "First difference":
        st.latex(r"h[n] = \delta[n] - \delta[n-1]")
    elif h_choice == "Box blur":
        st.latex(r"h[n] = \frac{1}{L}\sum_{k=0}^{L-1}\delta[n-k]")

# -----------------------------
# Panel 3: Output y = x * h
# -----------------------------
with c3:
    st.subheader("Output y[n] = (x * h)[n]")

    # Convolve each RGB channel along pixel axis
    # Mode='same' returns length max(len(x), len(h)); we ensured len(h) <= N to keep it N.
    y = np.zeros_like(x, dtype=float)
    for ch in range(3):
        y[:, ch] = np.convolve(x[:, ch], h, mode="same")

    # Normalize to [0,1] for display if needed
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-12:
        y_disp = np.clip(y, 0, 1)
    else:
        y_disp = (y - y_min) / (y_max - y_min)

    fig3, ax3 = plt.subplots(figsize=(6, 1.2))
    ax3.imshow(y_disp[None, :, :], aspect="auto")
    ax3.set_xticks([])
    ax3.set_yticks([])
    st.pyplot(fig3, use_container_width=True)
