"""Signal operations for the guitar Fourier lab. Frequencies inside this module are Hz."""
from io import BytesIO

import numpy as np
from scipy import fft, signal
from scipy.io import wavfile


def read_recording(source):
    rate, raw = wavfile.read(source)
    if raw.ndim not in (1, 2) or raw.size == 0 or raw.shape[0] < 2 or rate < 1:
        raise ValueError("Use a WAV containing at least two audio samples.")
    samples = raw.astype(np.float32)
    if raw.dtype.kind == "u":
        middle = float(2 ** (raw.dtype.itemsize * 8 - 1))
        samples = (samples - middle) / middle
    elif raw.dtype.kind == "i":
        # scipy left-justifies 24-bit PCM when decoding it into int32.
        samples /= float(2 ** (raw.dtype.itemsize * 8 - 1))
    elif raw.dtype.kind != "f":
        raise ValueError("Use a PCM or floating-point WAV.")
    if not np.isfinite(samples).all():
        raise ValueError("The WAV contains non-finite samples.")
    return rate, samples.reshape(raw.shape[0], -1)


def example_pluck():
    rate = 22050
    notes = [146.83, 185.0, 220.0, 293.66, 220.0, 185.0]
    audio = np.zeros(int(4.1 * rate), dtype=np.float32)
    t = np.arange(int(1.35 * rate)) / rate
    for i, frequency in enumerate(notes):
        voice = np.zeros_like(t)
        for k in range(1, 15):
            amplitude = np.sin(0.19 * np.pi * k) / k ** 1.45
            voice += amplitude * np.sin(2 * np.pi * k * frequency * t) * np.exp(-(2.9 + 0.5 * k) * t)
        start = int(i * 0.52 * rate)
        audio[start:start + len(t)] += (1 - np.exp(-t * 170)) * voice
    audio *= 0.82 / np.max(np.abs(audio))
    return rate, audio[:, None]


def transform(x, rate):
    # X(ω_k) ≈ Δt Σ x[n] exp(-jω_k nΔt) at ω_k = 2π k rate / N.
    freq = np.arange(len(x) // 2 + 1) * float(rate) / len(x)
    return freq, fft.rfft(x) / rate


def validate_bands(rows, nyquist):
    bands = []
    tol = max(1e-9, nyquist * 1e-12)
    for index, (low, high, gain) in enumerate(rows, 1):
        if not np.all(np.isfinite([low, high, gain])):
            raise ValueError(f"Complete the two bounds and gain in row {index}.")
        if low < 0 or high > nyquist + tol or low >= high:
            raise ValueError(f"Row {index} needs a lower edge below its upper edge, both within the available range.")
        if gain not in (0, 1):
            raise ValueError(f"Choose gain 0 or 1 in row {index}.")
        edge = nyquist if abs(high - nyquist) < tol else float(high)
        bands.append((float(low), edge, int(gain)))
    bands.sort()
    for first, second in zip(bands, bands[1:]):
        if second[0] < first[1]:
            raise ValueError("Two intervals overlap. Give each frequency a single gain. Adjacent intervals may share an edge.")
    return tuple(bands)


def response(frequencies, bands, outside, nyquist):
    """Even, piecewise-constant response. The band reaching Nyquist includes its endpoint."""
    abscissa = np.abs(frequencies)
    gain = np.full(abscissa.shape, outside, dtype=np.float32)
    for low, high, value in bands:
        upper = abscissa <= high if high == nyquist else abscissa < high
        gain[(abscissa >= low) & upper] = value
    return gain


def filter_signal(X, gain, length, rate):
    Y = X * gain
    return Y, fft.irfft(Y, n=length) * rate


def dominant_peaks(magnitude, limit, count):
    """Local maxima inside the view, including true endpoint maxima."""
    if count == 0 or not np.any(magnitude > 0):
        return np.empty(0, dtype=int)
    peaks = signal.find_peaks(magnitude)[0]
    if magnitude[0] > magnitude[1]:
        peaks = np.r_[0, peaks]
    if magnitude[-1] > magnitude[-2]:
        peaks = np.r_[peaks, len(magnitude) - 1]
    peaks = peaks[(peaks < limit) & (magnitude[peaks] > 0)]
    return peaks[np.argsort(-magnitude[peaks], kind="stable")[:count]]


def envelope_indices(values, budget=3200):
    """Keep chronological minima and maxima of display buckets, plus both endpoints."""
    n = len(values)
    if n <= budget:
        return np.arange(n)
    width = int(np.ceil(n / (budget // 2)))
    whole = n // width
    blocks = values[:whole * width].reshape(whole, width)
    bases = np.arange(whole) * width
    indices = np.r_[0, bases + blocks.argmin(axis=1), bases + blocks.argmax(axis=1), n - 1]
    tail = values[whole * width:]
    if len(tail):
        indices = np.r_[indices, whole * width + tail.argmin(), whole * width + tail.argmax()]
    return np.unique(indices)


def wav_bytes(x, rate, gain):
    buffer = BytesIO()
    pcm = np.rint(np.clip(x * gain, -1, 1) * 32767).astype(np.int16)
    wavfile.write(buffer, rate, pcm)
    return buffer.getvalue()
