# file: conv_audio_app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.io import wavfile

st.set_page_config(page_title="Audio Convolution Lab", layout="wide")

st.sidebar.write("Place `guitar_clean_2.wav` in the same folder.")
path = st.sidebar.text_input("Audio path", "guitar_clean_2.wav")

# Load audio (mono or first channel if stereo)
fs, x = wavfile.read(path)
x = x.astype(np.float32)
if x.ndim > 1:
    x = x[:, 0]
x /= (np.max(np.abs(x)) + 1e-12)

# Time axis preview length
seconds = st.sidebar.slider("Preview seconds", 0.5, min(10.0, len(x)/fs), 3.0, step=0.5)
N = int(seconds * fs)
x = x[:N]

c1, c2, c3 = st.columns(3, gap="large")

def moving_average(sig, L):
    if L <= 1: return sig.copy()
    w = np.ones(L, dtype=float) / L
    return np.convolve(sig, w, mode="same")

# Panel 1: input + cleaning (moving average)
with c1:
    st.subheader("Input x(t): Guitar (clean) + optional cleaning")
    Lclean = st.slider("Cleaning window (samples)", 1, int(0.050*fs), int(0.005*fs), step=1,
                       help="Moving-average length; larger = smoother")
    x_clean = moving_average(x, Lclean)
    # Auto y-limits
    ylim = float(np.max(np.abs(x_clean))) + 1e-3
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots()
    ax.plot(t, x, alpha=0.4, label="original")
    ax.plot(t, x_clean, label="cleaned")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    st.audio((x_clean/np.max(np.abs(x_clean)+1e-12)).astype(np.float32), sample_rate=fs)

# Panel 2: system impulse response h[n]
with c2:
    st.subheader("System h[n]: start as amplifier; add shaping")
    gain = st.slider("Gain", 0.1, 4.0, 1.0, step=0.1)
    shape = st.selectbox("Add linear shaping", ["None", "Averager (Lh)", "Echo (delay/decay)"])
    if shape == "None":
        h = np.array([gain], dtype=float)
    elif shape == "Averager (Lh)":
        Lh = st.slider("Lh (samples)", 1, int(0.020*fs), int(0.002*fs), step=1)
        h = np.ones(Lh) / Lh * gain
    else:
        delay_ms = st.slider("Echo delay [ms]", 10, 400, 120, step=10)
        decay = st.slider("Echo decay", 0.0, 0.95, 0.5, step=0.05)
        d = max(1, int(delay_ms * fs / 1000))
        h = np.zeros(d+1)
        h[0] = gain
        h[-1] = gain * decay

    # Optional nonlinearity: hard clipping (post-LTI)
    clip_on = st.checkbox("Apply clipping after convolution (nonlinear)", value=False)
    clip_level = st.slider("Clip level", 0.1, 1.0, 0.9, step=0.05) if clip_on else None

    fig2, ax2 = plt.subplots()
    n = np.arange(len(h))
    ax2.stem(n, h)
    ax2.set_xlabel("n (samples)")
    ax2.set_ylabel("h[n]")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# Panel 3: output y = x_clean * h (then optional clipping)
with c3:
    st.subheader("Output y(t) = (x_clean * h)(t)")
    y = np.convolve(x_clean, h, mode="same")
    if clip_on:
        y = np.clip(y, -clip_level, clip_level)
    # Normalize for audio playback
    y_play = y / (np.max(np.abs(y)) + 1e-12)
    t = np.arange(len(y)) / fs
    fig3, ax3 = plt.subplots()
    ax3.plot(t, y)
    ax3.set_xlim(0, t[-1] if len(t) else 1)
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("amplitude")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    st.audio(y_play.astype(np.float32), sample_rate=fs)
