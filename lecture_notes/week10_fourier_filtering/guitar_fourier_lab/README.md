# Guitar Fourier Lab

A Streamlit classroom app for exploring how hard clipping changes a guitar recording and how a student-designed frequency response changes the result.

## Run

Use Python 3.11 or newer. Extract the project, open a terminal in the folder containing `app.py`, and run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Place `guitar_clean.wav` beside `app.py` to use it automatically. You can also upload a PCM or floating-point WAV under **Choose recording**. The app supplies a labeled synthesized pluck when the recording is absent. For multichannel audio, choose the channel average or an individual channel.

## Use in class

1. Play the clean and clipped recordings. Set the clipping threshold to 100% for a clean reference, then lower it. The dotted lines show the clipping limits. Use the shared time window or drag inside a waveform to inspect a note.
2. Inspect the clipped-input spectrum. Choose Hz or rad/s, switch between a two-sided and a positive-frequency display, and set the number of dominant spectral peaks. The plotted spectra share a frequency axis. Input and output share a magnitude scale.
3. In **Design the filter**, add table rows with your lower edge, upper edge, and gain (0 or 1). Choose the gain outside those intervals and click **Apply filter**. To change an interval, edit its cells. To remove one, select its row and delete it in the table toolbar. An empty table uses the outside gain at all frequencies. The initial gain is one.
4. Compare the filtered spectrum with the input, then play the reconstructed waveform in the right column. The time window and frequency span control the plots. All players use the complete recording and a common gain.

Enter nonnegative frequency magnitudes. Each interval includes its lower endpoint and excludes its upper endpoint, with Nyquist included when it is the upper edge. The filter mirrors these values across zero to preserve real audio. Overlapping intervals are rejected. Unit changes preserve the last applied physical edges. Apply in-progress edits before changing units.

The FFT approximates the finite-record CTFT on a grid spaced by the reciprocal of the record duration. Connected lines and peak-preserving display reduction make the plots interactive. Filtering multiplies every complex coefficient by its gain and inverse transforms the full record at the original sample rate.
