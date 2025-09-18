import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.io.wavfile as sio

# load guitar signal wav file
sr, data = sio.read("guitar_clean.wav")
if data.ndim == 2:  # stereo
    print("Stereo signal detected; using one channel only.")
    x = data[:, 0]
else:
    x = data
x = x.astype(np.float64)
t = np.arange(x.size) / float(sr)

# modulate with complex exponential
f0 = 5.0
y = x * np.exp(1j*2*np.pi*f0*t).real
sio.write("guitar_modulated.wav", sr, y.astype(np.int16))

# --- plot + slider ---
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 3))

fig.subplots_adjust(bottom=0.22)
line_in, = ax[0].plot(t, x, label="Input", color="darkviolet")
# ax[0].plot(t, x, label="Input", color="lime", alpha=0.6)
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Amplitude")
ax[0].legend(loc="upper right")

ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Amplitude")
line_out, = ax[1].plot(t, y, label="Output", color="lime")
ax[1].legend(loc="upper right")
# ax[1].plot(t, y, label="Output", color="darkviolet", alpha=0.6)
axfreq = plt.axes([0.15, 0.08, 0.7, 0.04])
s_freq = Slider(axfreq, "f_mod [Hz]", valmin=0.1, valmax=20.0, valinit=f0, valstep=0.1)

def update(_):
    f = s_freq.val
    y = x * np.cos(2*np.pi*f*t)
    line_out.set_ydata(y)
    line_out.set_label(f"Output {f:.2f} Hz")
    ax[1].legend(loc="upper right")
    fig.canvas.draw_idle()
    sio.write("guitar_modulated.wav", sr, y.astype(np.int16))

s_freq.on_changed(update)
plt.show()
