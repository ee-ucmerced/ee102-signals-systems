# %matplotlib widget
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update({
    "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13
})

def rect_delta(t, eps):
    return (1/eps) * (np.abs(t) <= eps/2)

# Domain and initial epsilon
t = np.linspace(-1.0, 1.0, 4000)
eps0 = 0.4

fig, ax = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(bottom=0.25)

y = rect_delta(t, eps0)
(line,) = ax.plot(t, y, lw=2)
ax.axhline(0, color="k", lw=2)
ax.axvline(0, color="k", lw=2)
ax.grid(True, alpha=0.35)
ax.set_title(r"Rectangular Approximation of $\delta(t)$: width $\varepsilon$, height $1/\varepsilon$")
ax.set_xlabel("t")
ax.set_ylabel("amplitude")

# Dynamic y-limit to keep the pulse visible
ax.set_xlim(t[0], t[-1])
ax.set_ylim(0, 1.2/eps0)

# Slider
ax_eps = plt.axes([0.15, 0.08, 0.7, 0.06])
s_eps = Slider(ax_eps, r'$\varepsilon$ (width)', valmin=0.01, valmax=1.0, valinit=eps0, valstep=0.005)
s_eps.label.set_fontsize(14); s_eps.valtext.set_fontsize(14)

def update(val):
    eps = s_eps.val
    y = rect_delta(t, eps)
    line.set_ydata(y)
    ax.set_ylim(0, 1.2/eps)
    fig.canvas.draw_idle()

s_eps.on_changed(update)
plt.show()
