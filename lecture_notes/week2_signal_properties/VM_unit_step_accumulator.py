import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update({
    "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13
})

# Domain
t = np.linspace(-5, 5, 4001)
u = (t >= 0).astype(float)

# Accumulator S(T) = ∫_{-∞}^T u(τ) dτ = max(T, 0)
def S(T):
    return np.maximum(T, 0.0)

T0 = -2.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
plt.subplots_adjust(bottom=0.25, wspace=0.3)

# --- Left subplot: u(t) with shaded area up to T ---
(line_u,) = ax1.plot(t, u, lw=2)
ax1.axhline(0, color="k", lw=2); ax1.axvline(0, color="k", lw=2)
ax1.grid(True, alpha=0.35)
ax1.set_title("Unit step $u(t)$ and accumulated area up to $T$")
ax1.set_xlabel("t"); ax1.set_ylabel("u(t)")
bar = ax1.axvline(T0, color="C1", lw=2, linestyle="--")
# initial fill (area where 0 ≤ t ≤ T)
fill = ax1.fill_between(t, 0, u, where=(t>=0) & (t<=T0), alpha=0.25)

# --- Right subplot: S(T) vs T with moving marker ---
T_plot = t  # reuse same domain for display
S_plot = S(T_plot)
(line_S,) = ax2.plot(T_plot, S_plot, lw=2)
ax2.axhline(0, color="k", lw=2); ax2.axvline(0, color="k", lw=2)
ax2.grid(True, alpha=0.35)
ax2.set_title(r"Accumulated area $S(T)=\int_{-\infty}^{T} u(\tau)\,d\tau$")
ax2.legend(['S(T)'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
ax2.set_xlabel("T"); ax2.set_ylabel("S(T)")
pt, = ax2.plot(T0, S(T0), "o", ms=8)
txt = ax2.text(0.02, 0.92, f"T = {T0:.2f}\nS(T) = {S(T0):.2f}", transform=ax2.transAxes, fontsize=14,
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# Nice limits
ax1.set_xlim(t[0], t[-1]); ax1.set_ylim(-0.2, 1.2)
ax2.set_xlim(t[0], t[-1]); ax2.set_ylim(-0.5, t[-1]*1.05)

# Slider for T
ax_T = plt.axes([0.15, 0.08, 0.7, 0.06])
s_T = Slider(ax_T, "T (integral limit)", valmin=t[0], valmax=t[-1], valinit=T0, valstep=0.01)
s_T.label.set_fontsize(14); s_T.valtext.set_fontsize(14)

def update(val):
    T = s_T.val
    # Update left: vertical bar and shaded area
    bar.set_xdata([T, T])
    global fill
    # Remove old fill and draw new region 0≤t≤T (if T<0, nothing to fill)
    fill.remove()
    fill = ax1.fill_between(t, 0, u, where=(t>=0) & (t<=T), alpha=0.25)
    # Update right: moving point and text
    pt.set_data([T], [S(T)])
    txt.set_text(f"T = {T:.2f}\nS(T) = {S(T):.2f}")
    fig.canvas.draw_idle()

s_T.on_changed(update)
plt.show()
