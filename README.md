![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Max%20for%20Live](https://img.shields.io/badge/Max%20for%20Live-device-orange)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-alpha-lightgrey)

# IRIDyOM

IRIDyOM enables an IDyOM-style statistical model to be used in a real-time musical workflow. It combines a Max for Live device (`IRIDyOM.amxd`) with a Python TCP server (`main.py`) for MIDI analysis, next-note prediction, continuation, variation, and model training.

At a high level, IRIDyOM learns from symbolic MIDI material and uses sequential context to estimate probabilities for upcoming musical events. These probabilities can be used musically as suggestions, generated continuations, variations, or information-theoretic measures such as surprisal / information content. The project is inspired by the IDyOM model introduced by Marcus T. Pearce and later discussed as part of the information dynamics of music perception and cognition.

The Max for Live plugin requires Ableton Live Suite or Ableton Live with Max for Live. The Python server can also run independently and can be connected to, scripted, or extended from other applications via its local TCP interface (`127.0.0.1:5005`).

## Install / Run

Choose one of the options below.

### Option A: Windows installer
1. Download `IRIDyOM-Setup.exe`.
2. Run the installer.
3. Start the installed IRIDyOM server application.
4. In Ableton Live, drag and drop the Max for Live device from:
   - `C:\VST\IRIDyOM\IRIDyOM.amxd`
5. Keep the server running while using the device in Ableton Live.

### Option B: Repository install
This option runs the server from source on Windows, macOS, or Linux.

1. Download this repository as a ZIP, or clone it:
   ```bash
   git clone <repo-url>
   cd IRIDyOM
   ```
2. Install the Python dependencies.

   Windows:
   ```bash
   py -m pip install -r requirements.txt
   ```

   macOS / Linux:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Start the server.

   Windows:
   ```bash
   py main.py
   ```

   macOS / Linux:
   ```bash
   python3 main.py
   ```
4. To use the plugin, open Ableton Live and load `IRIDyOM.amxd` from this repository. On Linux, run the server directly and connect from another application or custom client.

> Note: Keep `IRIDyOM.amxd` in the same folder as the JavaScript files it depends on: `jsui_ic_plot.js`, `listen.js`, and `midi_to_note.js`.

## Modes (screenshots)

### 1) Sequencer
Autonomous generation mode.
- Streams generated notes to Ableton Live at the current tempo.
- Uses the current history (STM), MIDI range, and sampling settings (e.g., temperature / probabilistic selection).

![Sequencer](screenshots/1-seq.png)

### 2) Assistant
Interactive suggestion mode.
- You play notes; IRIDyOM returns ranked next-note candidates.
- Designed to support real-time exploration (and can compute per-note surprisal / IC during analysis).

![Assistant](screenshots/2-assist.png)

### 3) Continue
Continuation from an existing MIDI phrase.
- Analyze a MIDI clip/file note-by-note (including information content per note).
- Generate and export a continued MIDI file based on the analyzed material.

![Continue](screenshots/3-continue.png)

### 4) Variate
Generate variations from an analyzed phrase.
- Create variations of a selected part of the last analyzed MIDI.
- Control how "surprising" the variation is, then export the resulting MIDI.

![Variate](screenshots/4-variate.png)

### 5) Train
Create your own model from your material.
- Record training sequences and/or add MIDI files.
- Configure training options, then train.

![Train](screenshots/5-train.png)

## Citation

IRIDyOM is an independent Python / Max for Live reimplementation of core ideas from the IDyOM framework developed by Marcus T. Pearce. It does not include or redistribute the original Lisp implementation.

If you use IRIDyOM in academic work, please cite the original IDyOM publications:

```bibtex
@phdthesis{pearce2005construction,
  author = {Pearce, Marcus T.},
  title = {The Construction and Evaluation of Statistical Models of Melodic Structure in Music Perception and Composition},
  school = {School of Informatics, City University, London},
  year = {2005}
}

@article{pearce2012auditory,
  author = {Pearce, Marcus T. and Wiggins, Geraint A.},
  title = {Auditory Expectation: The Information Dynamics of Music Perception and Cognition},
  journal = {Topics in Cognitive Science},
  volume = {4},
  number = {4},
  pages = {625--652},
  year = {2012},
  doi = {10.1111/j.1756-8765.2012.01214.x}
}
```
