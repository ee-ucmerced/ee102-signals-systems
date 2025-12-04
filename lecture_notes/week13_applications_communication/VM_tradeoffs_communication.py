import numpy as np
import streamlit as st
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time–Frequency Tradeoffs", layout="wide")

st.title("Time–Frequency Tradeoffs in Noisy Communication")

st.markdown(
    """
This app illustrates:

- A bandlimited tone transmitted over a noisy channel
- Additive white noise plus **optional bandpass (structured) noise**
- A **lowpass receive filter** with adjustable bandwidth \(B_s\)
- How shrinking \(B_s\) reduces noise power but lengthens the impulse response (time spreading)
- How this affects **output SNR** in the signal band
"""
)

# ---------------------------
# Sidebar: parameters
# ---------------------------
st.sidebar.header("Simulation parameters")

Fs = st.sidebar.slider("Sampling rate $F_s$ (Hz)", 2000, 20000, 8000, step=1000)
T = st.sidebar.slider("Duration $T$ (seconds)", 0.05, 0.5, 0.25, step=0.05)
A = st.sidebar.slider("Signal amplitude $A$", 0.1, 2.0, 1.0, step=0.1)
f0 = st.sidebar.slider("Tone frequency $f_0$ (Hz)", 100, int(Fs/4), 400, step=50)
noise_std = st.sidebar.slider("White noise std. dev.", 0.0, 1.0, 0.2, step=0.05)

# Channel bandwidth (for reference only here; we use it to define a "telephone-like" band)
B_ch = st.sidebar.slider(
    "Channel bandwidth $B_{\\text{ch}}$ (Hz)",
    min_value=500,
    max_value=int(Fs / 2) - 200,
    value=1500,
    step=100,
)

# LPF bandwidth for the signal band
max_Bs = min(B_ch - 100, int(Fs / 2) - 300)
if max_Bs <= 100:
    max_Bs = 100

B_s = st.sidebar.slider(
    "Receive LPF bandwidth $B_s$ (Hz)",
    min_value=100,
    max_value=max_Bs,
    value=min(600, max_Bs),
    step=50,
)

# Optional structured bandpass noise
include_bp = st.sidebar.checkbox("Add structured bandpass noise $n_{\\text{bp}}(t)$", value=True)

if include_bp:
    bp_low = st.sidebar.slider(
        "Bandpass noise low cutoff (Hz)",
        min_value=B_s + 50,
        max_value=B_ch - 100,
        value=min(B_s + 200, B_ch - 100),
        step=50,
    )
    bp_high = st.sidebar.slider(
        "Bandpass noise high cutoff (Hz)",
        min_value=bp_low + 50,
        max_value=B_ch,
        value=min(bp_low + 200, B_ch),
        step=50,
    )
    bp_std = st.sidebar.slider("Bandpass noise std. dev.", 0.0, 1.0, 0.2, step=0.05)
else:
    bp_low, bp_high, bp_std = None, None, 0.0

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)

# ---------------------------
# Generate signals
# ---------------------------
np.random.seed(seed)

N = int(T * Fs)
t = np.arange(N) / Fs

# Clean tone signal
x = A * np.cos(2 * np.pi * f0 * t)

# White noise
n_white = noise_std * np.random.randn(N)

# Structured bandpass noise: filter white noise through a bandpass FIR
if include_bp and bp_std > 0.0:
    n_bp_raw = np.random.randn(N)
    # Long-ish FIR to show time spreading as B changes
    num_bp = firwin(
        numtaps=513,
        cutoff=[bp_low / (Fs / 2), bp_high / (Fs / 2)],
        pass_zero=False,
    )
    n_bp = bp_std * lfilter(num_bp, 1.0, n_bp_raw)
else:
    n_bp = np.zeros_like(x)

# Total received signal
y = x + n_white + n_bp

# ---------------------------
# Receive Lowpass Filter for signal band
# ---------------------------
# LPF with cutoff B_s
num_s = firwin(numtaps=513, cutoff=B_s / (Fs / 2))  # normalized cutoff

# Filtered outputs
z = lfilter(num_s, 1.0, y)
x_filt = lfilter(num_s, 1.0, x)
noise_out = z - x_filt

# Powers and SNR
Ps = np.mean(x_filt**2)
Pn = np.mean(noise_out**2) if np.mean(noise_out**2) > 0 else np.nan
SNR_dB = 10 * np.log10(Ps / Pn) if Pn > 0 else np.nan

# For comparison: what if we used a "full channel" LPF ~ B_ch instead of B_s?
num_full = firwin(numtaps=513, cutoff=B_ch / (Fs / 2))
z_full = lfilter(num_full, 1.0, y)
x_full = lfilter(num_full, 1.0, x)
noise_full = z_full - x_full
Ps_full = np.mean(x_full**2)
Pn_full = np.mean(noise_full**2) if np.mean(noise_full**2) > 0 else np.nan
SNR_full_dB = 10 * np.log10(Ps_full / Pn_full) if Pn_full > 0 else np.nan

# ---------------------------
# Frequency-domain helpers
# ---------------------------
def compute_spectrum(sig, Fs):
    N = len(sig)
    # Real-sided spectrum
    spec = np.fft.rfft(sig * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1 / Fs)
    mag = np.abs(spec) + 1e-12
    return freqs, mag


freqs, Xmag = compute_spectrum(x, Fs)
_, Ymag = compute_spectrum(y, Fs)
_, Zmag = compute_spectrum(z, Fs)

# Filter responses
freqs_h, H_s_mag = compute_spectrum(num_s, Fs)
_, H_full_mag = compute_spectrum(num_full, Fs)

# Limit frequency axis to a bit beyond channel bandwidth
f_max_plot = min(Fs / 2, B_ch * 1.3)

# ---------------------------
# Layout: plots
# ---------------------------
time_cols = st.columns(2)

# Time-domain signals
with time_cols[0]:
    st.subheader("Time domain: signal, noisy, and filtered outputs")

    t_view = t  # full duration; you can truncate if you want a zoom

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_view, x[: len(t_view)], label="Clean signal $x(t)$", linewidth=1)
    ax.plot(t_view, y[: len(t_view)], label="Noisy $y(t)$", linewidth=0.7, alpha=0.7)
    ax.plot(t_view, z[: len(t_view)], label="Filtered $z(t)$ (LPF $B_s$)", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)

with time_cols[1]:
    st.subheader("Impulse responses: time–frequency tradeoff")

    # Show impulse responses of the two LPFs
    n_imp = np.arange(len(num_s)) / Fs

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(n_imp, num_full, label=f"$h_{{\\text{{full}}}}(t)$, cutoff $B_{{\\text{{ch}}}}$", linewidth=1)
    ax.plot(n_imp, num_s, label=f"$h_s(t)$, cutoff $B_s$", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("h(t)")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)

st.markdown(
    """
Narrower $B_s$ → **longer** impulse response $h_s(t)$, which means more time spreading / ringing,
but less integrated noise power in the passband.
"""
)

# Frequency-domain plots
freq_cols = st.columns(2)

with freq_cols[0]:
    st.subheader("Spectra: input and filtered output")

    fig, ax = plt.subplots(figsize=(6, 3))
    mask = freqs <= f_max_plot

    ax.plot(freqs[mask], Xmag[mask], label="$|X(f)|$ (clean)", linewidth=1)
    ax.plot(freqs[mask], Ymag[mask], label="$|Y(f)|$ (noisy)", linewidth=0.7, alpha=0.7)
    ax.plot(freqs[mask], Zmag[mask], label="$|Z(f)|$ after LPF $B_s$", linewidth=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)

with freq_cols[1]:
    st.subheader("Filter magnitude responses")

    fig, ax = plt.subplots(figsize=(6, 3))
    mask_h = freqs_h <= f_max_plot

    ax.plot(freqs_h[mask_h], H_full_mag[mask_h], label=f"$|H_{{\\text{{full}}}}(f)|$ (cutoff $B_{{\\text{{ch}}}}$)", linewidth=1)
    ax.plot(freqs_h[mask_h], H_s_mag[mask_h], label=f"$|H_s(f)|$ (cutoff $B_s$)", linewidth=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)
    st.pyplot(fig)

# ---------------------------
# SNR summary
# ---------------------------
st.subheader("SNR and bandwidth tradeoff")

col_snr1, col_snr2 = st.columns(2)

with col_snr1:
    st.markdown(
        f"""
**Estimated output SNR with LPF bandwidth $B_s = {B_s}\\,\\text{{Hz}}$**

- Signal power (after LPF): $P_\\text{{sig,out}} \\approx$ `{Ps:.4f}`
- Noise power (after LPF): $P_\\text{{noise,out}} \\approx$ `{Pn:.4f}`
- SNR$_\\text{{out}}$ (simulated): `{SNR_dB:.2f}` dB
"""
    )

with col_snr2:
    st.markdown(
        f"""
**If we instead used the wider channel bandwidth $B_\\text{{ch}} = {B_ch}\\,\\text{{Hz}}$**

- Signal power (after full-channel LPF): $P_\\text{{sig,full}} \\approx$ `{Ps_full:.4f}`
- Noise power (after full-channel LPF): $P_\\text{{noise,full}} \\approx$ `{Pn_full:.4f}`
- SNR$_\\text{{full}}$ (simulated): `{SNR_full_dB:.2f}` dB

Ratio (processing gain) $\\approx$ `{(SNR_dB - SNR_full_dB):.2f}` dB
"""
    )

st.markdown(
    r"""
**Key takeaway**

For a tone of power \(P_{\text{signal}} \approx A^2/2\) in a bandlimited channel
with additive white noise of PSD \(N_0/2\), the \emph{continuous-time} SNR after an
ideal lowpass filter of bandwidth \(B_s\) is

\[
\text{SNR}_\text{out} \approx \frac{A^2}{2 N_0 B_s}.
\]

This app approximates that situation in discrete time. As you decrease \(B_s\):

- The impulse response \(h_s(t)\) gets **longer** (more time spreading / ringing).
- The integrated noise power in the passband **decreases**, so SNR **increases**.
- If the structured bandpass noise is placed outside \([0, B_s]\), the receiver can
  remove it entirely by lowpass filtering, so it does not hurt SNR in the band of interest.

Once \(z(t)\) is bandlimited to \(B_s\), sampling at \(F_s \ge 2 B_s\) allows you
to reconstruct \(z(t)\) from its samples without changing this SNR in an ideal system.
"""
)
