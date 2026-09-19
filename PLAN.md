# Sheet-Reading Piano Agent (RL) — Project Plan

## 1. Goal

Train a reinforcement-learning agent that plays a virtual piano by **reading a music score**, not by memorizing songs. After training, I load any piece (for example Rush E) as a score file and the agent performs it on a virtual piano, controlling hands and fingers itself.

**Constraints:** CPU only (integrated graphics), built by me, no heavy physics engine.

## 2. What success looks like

| Level | Result |
|---|---|
| Minimum | Agent plays simple, unseen melodies from a score with correct notes at correct times |
| Target | Agent plays easy-to-medium two-hand pieces it has never seen, with hand-span and movement limits enforced |
| Stretch | Agent attempts Rush E (partial success is fine and expected) |
| Stretch 2 | Score encoder swapped for an image reader (sheet-music pictures) |

The main proof of learning is **held-out evaluation**: pieces never used in training.

## 3. Core idea: reading vs. memorizing

- If trained on one song, the agent memorizes it.
- If trained on hundreds of different scores and tested on unseen ones, it can only succeed by learning to **read the notation and execute it**.
- The agent never sees the whole piece. It only sees a short **sliding window** of upcoming beats, like a person reading a page. It still plays the whole piece from start to finish.

Two kinds of data, kept separate:
1. **Training scores:** many varied pieces, used once to learn the skill.
2. **Performance score:** any piece I load at play time. No retraining per song.

## 4. System overview

```
 Score file (MIDI / MusicXML)
        |
        v
 [Score loader]  -- parses to beat-based events (pitch, start beat, duration in beats, tempo)
        |
        v
 [Score window]  -- returns ONLY the next N beats to the agent (N = 2 to 4)
        |
        v
 [Policy network] <-- observation: window + hand positions + finger states + tempo
        |
        v  actions: move hands, press/release fingers
        v
 [Piano simulation] -- enforces span, movement speed, timing; computes reward
        |
        v
 [Visual player]  -- pygame window or HTML/CSS piano showing keys pressing + sound
```

The same simulation rules are used for training (headless, fast) and performing (with graphics).

## 5. Design decisions so far

| Decision | Choice |
|---|---|
| Input to the agent | Symbolic score (beats, tempo) first; sheet images later as an optional upgrade |
| Learning method | Reinforcement learning (PPO) |
| Score visibility | Sliding lookahead window only |
| Piano for training | Lightweight NumPy simulation, no MuJoCo |
| Piano for performing | Visual version that follows the same rules |
| Song specificity | None. One trained agent plays any loaded piece |

## 6. Components

### 6.1 Score loader (no ML)
- Parses MIDI or MusicXML into a list of events: `pitch`, `start_beat`, `duration_beats`, `hand` (optional), plus a tempo (BPM).
- Libraries: `pretty_midi`, `mido`, or `music21`.
- Time is stored in **beats**, not seconds. Seconds only appear when the tempo is applied inside the simulation.

### 6.2 Score window
- Interface: given the current beat, return the notes starting within the next N beats.
- Format: a small piano-roll-style array, 88 pitches by a fixed number of time slots, with channels for "note starts here" and "note is held here".
- N (lookahead length) is a tunable difficulty knob: longer = easier planning, shorter = harder sight-reading.

### 6.3 Piano and hand simulation
Keep it simple and fast.
- 88 keys.
- Time steps at 16th-note resolution (4 steps per beat) to start. Triplets can come later.
- Two hands. Each hand has an anchor position on the keyboard and 5 fingers.
- **v1:** fixed five-finger position (each finger sits at a fixed offset from the anchor). Moving the anchor shifts the whole hand.
- **v2:** each finger can adjust its offset within a maximum hand span (about an octave), which allows stretches and thumb-under movement.
- **Movement speed limit** is defined in keys per real second. Because the tempo converts steps to seconds, a fast tempo leaves less time to move, which is what makes tempo matter to the agent.

### 6.4 Observation, action, reward

**Observation**
- Score window (upcoming notes).
- Each hand's anchor position.
- Which fingers are currently down.
- Current tempo.
- Position within the beat (so it knows where it is in time).

**Action** (per hand)
- Move: left / right by 0, 1, or 2 keys (clipped by the speed limit).
- Press or release each of the 5 fingers.

**Reward** (per step)
- Positive for pressing a due note within a small timing tolerance (start with plus or minus 1 step).
- Negative for wrong presses.
- Negative for missed notes.
- Small penalty for unnecessary hand movement.
- Optional: reward for holding notes for the right duration (add after onsets work).

### 6.5 Policy and training
- Algorithm: PPO (`stable-baselines3`), running on CPU.
- Network: small MLP first. Try a GRU or 1D conv over the window if the MLP plateaus.
- Environment: custom `gymnasium.Env`.
- Each episode: sample a random training score, play it start to finish (or a random segment while training to save time).
- Use several parallel environments (`SubprocVecEnv`) to use all CPU cores.

### 6.6 Performance frontend
- **Option A:** pygame window drawing the keyboard, keys pressing, and sound via a synth or samples. Stays in Python.
- **Option B:** HTML/CSS piano with Tone.js, driven by a websocket from the Python process.
- Not needed until late. Early on, plot piano-roll comparisons (target vs played) with matplotlib.

## 7. Data

- **Important:** performance MIDI (for example MAESTRO) is timed in seconds and reflects human timing, not clean notation. For a "sheet-like" input, use **beat-quantized** scores, or quantize performance MIDI to a beat grid first.
- Sources to look at: `music21` built-in corpus, Mutopia Project (public domain, MIDI and LilyPond), MusicXML collections, and quantized MAESTRO.
- **Procedurally generated scores** (scales, arpeggios, random melodies in a key, simple chord patterns) are very useful:
  - unlimited data,
  - controllable difficulty,
  - ideal for the early curriculum on a CPU.
- Keep a **held-out set** from the start (about 10 to 15 percent) and never train on it.
- Rush E is only a final test piece, never part of training.

## 8. Curriculum (easy to hard)

1. Single notes, one hand, slow tempo, free keys.
2. Single-line melodies, one hand, five-finger range.
3. One hand with position shifts.
4. Two hands, separate simple lines.
5. Chords and simple accompaniment patterns.
6. Faster tempos and denser passages.
7. Real pieces (easy, then medium).

Advance a stage only when held-out performance on the current stage passes a threshold.

## 9. Evaluation

- **Note F1 score** with a timing tolerance: correct pitch at the correct beat.
- Separate metrics for missed notes, extra notes, and timing error.
- Always report on **held-out pieces**, per difficulty level.
- **Baselines** to beat (they show that learning matters):
  - Random agent.
  - A simple rule-based player (press the note when due, move hand to nearest note).
  - Free-keys agent (upper bound with no physical limits).
- Record videos or piano-roll plots of the agent on unseen pieces for the portfolio.

## 10. Roadmap

| # | Milestone | Done when |
|---|---|---|
| 0 | Setup and data | Environment installed; a few MIDI files load and print as beat events |
| 1 | Score loader and window | Window returns the correct upcoming notes for any beat; unit-tested on a small piece |
| 2 | Free-keys environment plus PPO | Agent reaches near-perfect F1 on simple scores (pipeline sanity check) |
| 3 | Single-hand constraints | Agent plays melodies within one hand's range and speed limit |
| 4 | Two hands | Both hands coordinate on simple pieces |
| 5 | Curriculum and held-out eval | F1 table on unseen pieces at each difficulty level |
| 6 | Visual player | Trained agent performs a loaded MIDI file live with sound |
| 7 | Stretch | Rush E attempt; optional image-based score encoder |

Do not skip milestone 2. If the agent can't learn the easy version, the harder versions will only be harder to debug.

## 11. Tech stack

- Python 3.10+
- `numpy`, `gymnasium`, `stable-baselines3`, `torch` (CPU build)
- `pretty_midi` / `mido` / `music21` for scores
- `matplotlib` for early visualization
- `pygame` or (`fastapi`/`websockets` + Tone.js) for performance
- `tensorboard` for training curves

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reward too sparse, agent never learns | Start with free keys, dense per-step reward, slow tempo |
| Agent presses everything, or nothing | Balance wrong-press and missed-note penalties; check per-note stats |
| Overfitting to training pieces | Held-out set from day one; more procedural variety |
| Training too slow on CPU | Lightweight env, parallel workers, short segments, small network |
| Rush E far too hard | Treat as a stretch demo; report honest partial results |
| Train and perform behave differently | Same rules and same lookahead window in both |
| Data isn't truly "sheet-like" | Quantize to beats; prefer notation-based sources |

## 13. Stretch goals

- Fingering choice as an explicit skill (thumb-under, crossing).
- Sustain pedal.
- Dynamics (velocity) read from the score.
- Sheet-image input with a small CNN replacing the symbolic encoder (keep the encoder swappable from the start).
- Play-by-ear mode (audio spectrogram as input).

## 14. Open decisions

- [ ] Hand model: start with fixed five-finger position (v1) or flexible offsets (v2)?
- [ ] Lookahead length N (start at 2 to 4 beats and experiment).
- [ ] Visual player: pygame or HTML/CSS with Tone.js?
- [ ] Main dataset: procedural only at first, or mix in quantized real pieces from the start?
- [ ] Time resolution: 16th notes only, or support triplets from the start?
