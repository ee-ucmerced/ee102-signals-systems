import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update({
    "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13
})

def gauss_delta(t, sigma):
    # Unit area Gaussian: (1 / (sqrt(2π) σ)) * exp(-t^2 / (2σ^2))
    return (1.0 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*(t/sigma)**2)

t = np.linspace(-1.0, 1.0, 4000)
sigma0 = 0.2

fig, ax = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(bottom=0.25)

y = gauss_delta(t, sigma0)
(line,) = ax.plot(t, y, lw=2)
ax.axhline(0, color="k", lw=2)
ax.axvline(0, color="k", lw=2)
ax.grid(True, alpha=0.35)
ax.set_title(r"Gaussian Approximation of $\delta(t)$: $\sigma \to 0$")
ax.set_xlabel("t")
ax.set_ylabel("amplitude")

ax.set_xlim(t[0], t[-1])
ax.set_ylim(0, 1.2 * np.max(y))

ax_sig = plt.axes([0.15, 0.08, 0.7, 0.06])
s_sig = Slider(ax_sig, r'$\sigma$ (width)', valmin=0.0001, valmax=0.6, valinit=sigma0, valstep=0.005)
s_sig.label.set_fontsize(14); s_sig.valtext.set_fontsize(14)

def update(val):
    sigma = s_sig.val
    y = gauss_delta(t, sigma)
    line.set_ydata(y)
    ax.set_ylim(0, 1.2 * y.max())
    fig.canvas.draw_idle()

s_sig.on_changed(update)
plt.show()
