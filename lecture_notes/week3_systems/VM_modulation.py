import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.io.wavfile as sio

# --- load audio (mono) ---
sr, data = sio.read("guitar_clean.wav")
x = data[:, 0] if data.ndim == 2 else data
x = x.astype(np.float64)
t = np.arange(x.size) / float(sr)

# --- initial modulation + save ---
f0 = 5.0
y = x * np.cos(2*np.pi*f0*t)
sio.write("guitar_modulated.wav", sr, y.astype(np.int16))

# --- plot + slider ---
fig, ax = plt.subplots(figsize=(10, 3))
fig.subplots_adjust(bottom=0.22)
line_out, = ax.plot(t, y, label=f"Output {f0:.2f} Hz", color="darkviolet")
ax.plot(t, x, label="Original", color="lime", alpha=0.6)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude"); ax.legend(loc="upper right")

axfreq = plt.axes([0.15, 0.08, 0.7, 0.04])
s_freq = Slider(axfreq, "f_mod [Hz]", valmin=0.1, valmax=20.0, valinit=f0, valstep=0.1)

def update(_):
    f = s_freq.val
    y = x * np.cos(2*np.pi*f*t)
    line_out.set_ydata(y)
    line_out.set_label(f"Output {f:.2f} Hz")
    ax.legend(loc="upper right")
    fig.canvas.draw_idle()
    sio.write("guitar_modulated.wav", sr, y.astype(np.int16))

s_freq.on_changed(update)
plt.show()
