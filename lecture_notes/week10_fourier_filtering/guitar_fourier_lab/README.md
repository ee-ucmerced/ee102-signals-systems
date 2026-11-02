# Guitar Fourier Lab

A Streamlit classroom app for exploring how hard clipping changes a guitar recording and how a student-designed frequency response changes the result.

## Run

Use Python 3.11 or newer. Extract the project, open a terminal in the folder containing `app.py`, and run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Place `guitar_clean.wav` beside `app.py` to use it automatically. You can also upload a PCM or floating-point WAV under **Choose recording**. The app supplies a labeled synthesized pluck when the recording is absent. For multichannel audio, choose the channel average or an individual channel.

# AI Use Statement
AI tools were used in generating the app. The author verified all content and its technical accuracy. See the general AI use statement at the header of this repository.