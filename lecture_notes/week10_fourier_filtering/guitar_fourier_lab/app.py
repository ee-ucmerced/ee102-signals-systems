from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plots import CLEAN, CLIP, CONFIG, GRAY, OUTPUT, base, spectrum_figure, waveform
from signals import (dominant_peaks, example_pluck, filter_signal, read_recording,
                     response, transform, validate_bands, wav_bytes)

st.set_page_config(page_title="Guitar Fourier Lab", layout="wide")
st.markdown("""
<style>
.stMainBlockContainer {max-width: 1580px; padding-top: 4.1rem; padding-bottom: 3rem;}
.st-key-clean, .st-key-clipped, .st-key-reconstruction {
    background: #eff7f1; border: 1px solid #c2d8c9; border-radius: 14px; padding: 1.05rem 1.2rem;
}
.st-key-input_panel, .st-key-filter_panel, .st-key-output_panel {
    background: #f1f3f4; border: 1px solid #d6dddf; border-radius: 14px; padding: 1.05rem 1rem;
}
.st-key-input_panel {border-top: 3px solid #a54728;}
.st-key-filter_panel {border-top: 3px solid #364f7a;}
.st-key-output_panel {border-top: 3px solid #4c4792;}
.eyebrow {font: 700 0.73rem/1.45 system-ui; letter-spacing: .12em; color: #44645a; margin: 0 0 .32rem;}
.panel-title {font: 650 1.22rem/1.3 system-ui; color: #1e3531; margin: 0;}
.freq-label {color: #526075;}
@media (max-width: 900px) {
    .st-key-frequency_stages [data-testid="stHorizontalBlock"] {flex-direction: column;}
    .st-key-frequency_stages [data-testid="stColumn"] {width: 100%; flex: 1 1 auto; min-width: 0;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=2)
def uploaded_audio(payload):
    return read_recording(BytesIO(payload))


@st.cache_data(show_spinner=False, max_entries=2)
def local_audio(path, modified, size):
    return read_recording(path)


@st.cache_data(show_spinner=False)
def demo_audio():
    return example_pluck()


def chart(figure, key):
    st.plotly_chart(figure, theme=None, width="stretch", key=key, config=CONFIG)


def stage_chart(source, column, key):
    """Give a stage its own Plotly figure, retaining the shared transform and scales."""
    suffix = "" if column == 1 else str(column)
    xaxis = source.layout[f"xaxis{suffix}"]
    yaxis = source.layout[f"yaxis{suffix}"]
    fig = go.Figure(data=list(source.select_traces(row=1, col=column)))
    fig.update_traces(xaxis="x", yaxis="y")
    base(fig, 230, GRAY)
    fig.update_layout(margin=dict(l=56, r=12, t=10, b=44))
    fig.update_xaxes(title_text=xaxis.title.text, range=xaxis.range, nticks=5,
                     automargin=False, uirevision=xaxis.uirevision)
    fig.update_yaxes(title_text="Gain" if column == 2 else "Magnitude", range=yaxis.range,
                     tickvals=yaxis.tickvals, automargin=False)
    chart(fig, key)


# Each stage has its own Streamlit card and Plotly instance. Relay user zoom/pan
# events between the three plots in the browser without rerunning the audio pipeline.
link_stage_views = st.components.v2.component("link_stage_views", js="""
export default function () {
    const stages = [".st-key-input_spectrum", ".st-key-filter_response", ".st-key-output_spectrum"];
    const listeners = new Map();
    const updating = new WeakSet();
    let frame = 0;
    function bind() {
        frame = 0;
        const plots = stages.map(s => document.querySelector(s + " .js-plotly-plot"));
        for (const [plot, callback] of listeners) {
            if (!plots.includes(plot)) {
                plot.removeListener?.("plotly_relayout", callback);
                listeners.delete(plot);
            }
        }
        plots.forEach((plot, stage) => {
            if (!plot?.on || listeners.has(plot) || !window.Plotly?.relayout) return;
            const callback = change => {
                if (updating.has(plot)) return;
                stages.forEach((selector, other) => {
                    const target = document.querySelector(selector + " .js-plotly-plot");
                    if (!target || target === plot) return;
                    const update = {};
                    for (const axis of ["xaxis", "yaxis"]) {
                        if (axis === "yaxis" && (stage === 1 || other === 1)) continue;
                        if (!Object.keys(change).some(key => key.startsWith(axis + "."))) continue;
                        const range = plot._fullLayout[axis].range;
                        const current = target._fullLayout[axis].range;
                        if (range[0] !== current[0] || range[1] !== current[1]) {
                            update[axis + ".range"] = range.slice();
                            update[axis + ".autorange"] = false;
                        }
                    }
                    if (Object.keys(update).length) {
                        updating.add(target);
                        window.Plotly.relayout(target, update).finally(() => updating.delete(target));
                    }
                });
            };
            plot.on("plotly_relayout", callback);
            listeners.set(plot, callback);
        });
    }
    const row = document.querySelector(".st-key-frequency_stages");
    const observer = new MutationObserver(() => {
        if (!frame) frame = requestAnimationFrame(bind);
    });
    observer.observe(row || document.body, {childList: true, subtree: true});
    bind();
    return () => {
        observer.disconnect();
        cancelAnimationFrame(frame);
        for (const [plot, callback] of listeners) plot.removeListener?.("plotly_relayout", callback);
    };
}
""")


def heading(kicker, title, freq=False):
    cls = "eyebrow freq-label" if freq else "eyebrow"
    # Only application-owned, static labels are passed to this function.
    st.markdown(f'<p class="{cls}">{kicker}</p><p class="panel-title">{title}</p>',
                unsafe_allow_html=True)


def choose_recording():
    uploaded = st.file_uploader("Load a WAV recording", type=["wav", "wave"],
                                help="An uploaded recording takes precedence over guitar_clean.wav beside app.py.")
    bundled = Path(__file__).resolve().with_name("guitar_clean.wav")
    try:
        if uploaded is not None:
            rate, tracks = uploaded_audio(uploaded.getvalue())
            return rate, tracks, uploaded.name, uploaded.file_id
        if bundled.is_file():
            info = bundled.stat()
            rate, tracks = local_audio(str(bundled), info.st_mtime_ns, info.st_size)
            return rate, tracks, bundled.name, (info.st_mtime_ns, info.st_size)
    except (OSError, ValueError, EOFError) as error:
        st.error(f"Could not read the recording. {error} Choose a PCM or floating-point WAV.")
        st.stop()
    rate, tracks = demo_audio()
    return rate, tracks, "Synthesized plucked strings", "demo"


def select_mono(tracks):
    if tracks.shape[1] == 1:
        return tracks[:, 0]
    options = ["Average of all channels"] + [f"Channel {k + 1}" for k in range(tracks.shape[1])]
    choice = st.selectbox("Signal to analyze and play", options)
    return tracks.mean(axis=1) if choice == options[0] else tracks[:, options.index(choice) - 1]


def filter_editor(nyquist, unit, scale, math_variable):
    state = st.session_state
    # A change of units rebuilds the editor from the same accepted physical frequencies.
    if state.get("filter_rate") != nyquist:
        state.filter_rate = nyquist
        state.bands = ()
        state.elsewhere = 1
        state.editor_context = None
    if state.get("editor_context") != (nyquist, unit):
        state.editor_context = (nyquist, unit)
        state.editor_revision = state.get("editor_revision", 0) + 1
        state.editor_seed = [(low * scale, high * scale, gain) for low, high, gain in state.bands]

    fields = ["Lower", "Upper", "Gain"]
    table = pd.DataFrame(state.editor_seed, columns=fields).astype(
        {"Lower": "float64", "Upper": "float64", "Gain": "Int64"})
    rev = state.editor_revision
    st.caption("Click a blank row and enter the lower and upper frequencies. "
               "Gain 1 keeps that range. Gain 0 removes it. Each interval also applies to its negative frequencies.")
    with st.form(f"filter_form_{rev}", border=False):
        edge = st.column_config.NumberColumn
        edited = st.data_editor(table, num_rows="dynamic", hide_index=True, width="stretch",
                                key=f"band_table_{rev}",
                                column_config={
                                    "Lower": edge(f"From ({unit})", min_value=0.0, format="%.6g"),
                                    "Upper": edge(f"To ({unit})", min_value=0.0, format="%.6g"),
                                    "Gain": st.column_config.SelectboxColumn("Gain", options=[0, 1],
                                                                            default=0, required=True),
                                })
        other = st.selectbox("Gain outside these intervals", [1, 0], index=[1, 0].index(state.elsewhere),
                             key=f"outside_{rev}")
        apply = st.form_submit_button("Apply filter", type="primary")
    if apply:
        try:
            numeric = edited.to_numpy(dtype=float, na_value=np.nan)
            numeric[:, :2] /= scale
            bands = validate_bands(numeric, nyquist)
        except ValueError as error:
            st.error(f"{error} The plots and audio still use the last applied filter.")
        else:
            state.bands, state.elsewhere = bands, other
    symbol = f"|{math_variable}|"
    definition = f"H({math_variable})=" + r"\begin{cases}"
    if state.bands:
        pieces = []
        for lo, hi, gain in state.bands:
            high_op = r"\le" if hi == nyquist else "<"
            pieces.append(rf"{gain} & {lo * scale:.6g}\le {symbol} {high_op} {hi * scale:.6g}")
        pieces.append(rf"{state.elsewhere} & \text{{elsewhere}}")
        st.latex(definition + r"\\".join(pieces) + r"\end{cases}")
    else:
        st.latex(definition + r"g_i & a_i\le " + symbol
                 + r"<b_i\\ g_{\rm out} & \text{elsewhere}\end{cases}")
        st.caption(f"Current filter: gain {state.elsewhere} everywhere. Add a row to define an interval.")
    return state.bands, state.elsewhere


st.markdown('<p class="eyebrow">SIGNAL PROCESSING  /  FOURIER LAB</p>', unsafe_allow_html=True)
st.title("Guitar distortion & filtering")
st.write("Clip a recording, design a frequency response, and listen to the signal you reconstruct.")

with st.expander("Choose recording"):
    rate, tracks, name, source_id = choose_recording()
    clean = select_mono(tracks)
    duration = len(clean) / rate
    if st.session_state.get("source_id") != source_id:
        st.session_state.source_id = source_id
        st.session_state.pop("time_window", None)

if source_id == "demo":
    st.caption("Playing a synthesized pluck. Place guitar_clean.wav beside app.py or upload it under Choose recording.")

left, right = st.columns(2, gap="medium")
with left.container(key="clean"):
    heading("01  /  TIME DOMAIN", "Clean recording")
    st.text(f"{name}  ·  {rate:,} Hz  ·  {duration:.3f} s")
    clean_plot = st.container()
    window = st.slider("Time window for all waveforms (s)", 0.0, duration, (0.0, duration),
                       step=max(1 / rate, duration / 100000), format="%.5f", key="time_window",
                       help="Drag the handles to inspect a shorter interval. All three players use the full recording.")
    start = min(window[0], (len(clean) - 1) / rate)
    window = (start, min(duration, max(start + 1 / rate, window[1])))
    clean_player = st.container()
with right.container(key="clipped"):
    heading("02  /  TIME DOMAIN", "Clipped recording · x(t)")
    clip_label = st.empty()
    clip_plot = st.container()
    percent = st.slider("Clipping threshold (% of clean peak)", 1, 100, 35,
                        help="100% retains every sample. Lower the threshold to flatten more peaks.")
    clip_player = st.container()

peak = float(np.max(np.abs(clean)))
threshold = peak * (percent / 100)
clipped = np.clip(clean, -threshold, threshold)
clip_label.text(f"Clipping limits   ±{threshold:.4f}")
freq, X = transform(clipped, rate)
mx = np.abs(X)
nyquist = rate / 2

with st.expander("Frequency-axis settings for panels 3, 4, and 5"):
    controls = st.columns(3, gap="medium")
    unit = controls[0].selectbox("Frequency units", ["Hz", "rad/s"],
                                 help="Both units select the same sound. An interval entered in Hz is multiplied by 2π when displayed in rad/s.")
    variable = "f" if unit == "Hz" else "ω"
    math_variable = "f" if unit == "Hz" else r"\omega"
    scale = 2 * np.pi if unit == "rad/s" else 1.0
    signed = controls[1].radio("Frequency axis", ["Two-sided", "Positive half"], horizontal=True) == "Two-sided"
    # Keep the same physical span when switching between Hz and rad/s.
    if st.session_state.get("view_rate") != rate:
        st.session_state.view_rate = rate
        st.session_state.span_hz = min(2500.0, nyquist)
        st.session_state.span_context = None
    spacing = rate / len(clean)
    if st.session_state.span_context != (unit, spacing):
        st.session_state.span_hz = float(np.clip(st.session_state.span_hz, spacing, nyquist))
        st.session_state.span_context = (unit, spacing)
        st.session_state["span_input"] = float(st.session_state.span_hz * scale)
    span_axis = controls[2].number_input(f"Display up to ({unit})", min_value=float(spacing * scale),
                                         max_value=float(nyquist * scale), step=float(min(100, nyquist) * scale),
                                         format="%.5f", key="span_input")
    span = min(nyquist, span_axis / scale)
    st.session_state.span_hz = span

with st.container(key="frequency_stages"):
    input_col, filter_col, output_col = st.columns(3, gap="medium")

with input_col.container(key="input_panel"):
    heading("03  /  FREQUENCY DOMAIN", f"Input spectrum · |X({variable})|", freq=True)
    input_plot = st.container()
    top = st.slider("Dominant peaks", 0, 20, 5)
    count = np.searchsorted(freq, span, side="right")
    peaks = dominant_peaks(mx, count, top)
    st.caption("Peaks of the clipped recording, ranked by magnitude. Each positive/negative pair counts once.")
    if peaks.size:
        data = pd.DataFrame({f"Frequency ({unit})": freq[peaks] * scale, f"|X({variable})|": mx[peaks]})
        st.dataframe(data, hide_index=True, width="stretch", column_config={
            data.columns[0]: st.column_config.NumberColumn(format="%.3f"),
            data.columns[1]: st.column_config.NumberColumn(format="%.5g"),
        })
    elif top:
        st.caption("No spectral peak in this range.")
    else:
        st.caption("Peak markers are off.")

with filter_col.container(key="filter_panel"):
    heading("04  /  FREQUENCY DOMAIN", f"Filter design · H({variable})", freq=True)
    filter_plot = st.container()
    bands, outside = filter_editor(nyquist, unit, scale, math_variable)

gains = response(freq, bands, outside, nyquist)
Y, output = filter_signal(X, gains, len(clean), rate)
spectra = spectrum_figure(freq, mx, np.abs(Y), peaks, span, signed, scale, unit, bands, outside, nyquist)
with input_plot:
    stage_chart(spectra, 1, "input_spectrum")
with filter_plot:
    stage_chart(spectra, 2, "filter_response")

with output_col.container(key="output_panel"):
    heading("05  /  FREQUENCY DOMAIN", f"Output spectrum · |Y({variable})|", freq=True)
    stage_chart(spectra, 3, "output_spectrum")
    st.latex(f"Y({math_variable})=H({math_variable}) X({math_variable})")
    with st.container(key="reconstruction"):
        heading("INVERSE TRANSFORM  /  TIME DOMAIN", "Listen to y(t)")
        output_plot = st.container()
        output_player = st.container()

link_stage_views(key="linked_frequency_axes", height=0)

output_peak = float(np.max(np.abs(output)))
display_limit = max(peak, output_peak, 1e-5) * 1.06
playback_gain = 1 / max(1.0, peak, output_peak)
for values, color, plot_slot, player_slot, key, limit in [
    (clean, CLEAN, clean_plot, clean_player, "clean_wave", None),
    (clipped, CLIP, clip_plot, clip_player, "clipped_wave", threshold),
    (output, OUTPUT, output_plot, output_player, "output_wave", None),
]:
    with plot_slot:
        chart(waveform(values, rate, window, display_limit, color, limit), key)
    player_slot.audio(wav_bytes(values, rate, playback_gain), format="audio/wav")


with st.expander("Frequency display computation details"):
    st.markdown("The plots use angular frequency $\\omega=2\\pi f$. The recording has sample interval "
                "$\\Delta t=1/f_s$ and $N$ samples. The FFT evaluates a finite-record approximation to the continuous-time Fourier transform. You will learn about FFTs later in this class but the ideas about frequency domain remain the same as in this app.")
    st.latex(r"X(\omega_k)\approx \Delta t\sum_{n=0}^{N-1}x(n\Delta t)e^{-j\omega_k n\Delta t},"
             r"\qquad \omega_k=\frac{2\pi k f_s}{N}")
    st.caption(f"Frequency spacing {rate / len(clean):.5g} Hz  ·  Nyquist {nyquist:g} Hz  ·  playback gain {playback_gain:.4g}")
    st.caption("AI Use Statement: I (Ayush) have used AI tools to draft this app. I used a specific prescribed design of the app, detailed the steps of computation, and described the pedagogical behavior of the app carefully so that it helps in learning. All technical content was checked for accuracy and correctness.")