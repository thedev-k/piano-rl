"""Two-voice polyphonic procedural music generator for multi-key piano curriculum (Levels 1M to 8M).

Generates pieces containing chords, dyads, counterpoint, and accompaniment patterns,
where multiple notes can sound and start simultaneously. All notes land strictly
on the 16th-note grid (multiples of 0.25 beats) and within standard 88-key bounds.
"""

import random
from typing import List, Optional, Tuple, Union

from pianorl.score import (
    NoteEvent,
    Score,
    MIN_PIANO_PITCH,
    MAX_PIANO_PITCH,
)

# Common scale and chord interval templates
MAJOR_SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]
HARMONIC_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 11]

MAJOR_TRIAD = [0, 4, 7]
MINOR_TRIAD = [0, 3, 7]


def _clamp_pitch(p: int) -> int:
    """Ensure a MIDI pitch stays strictly inside the 88-key piano limits."""
    return max(MIN_PIANO_PITCH, min(MAX_PIANO_PITCH, p))


def generate_level_1m(rng: random.Random) -> Score:
    """Level 1M: Synchronized 2-note dyads on quarter notes (60-80 BPM, 16 beats).

    At every quarter note, exactly two keys are struck simultaneously: a bass root
    and a harmonizing note (3rd, 5th, 6th, or octave above).
    """
    tempo = float(rng.randint(60, 80))
    # Pick root key in bass/tenor register: C3 (48) to G3 (55)
    root = rng.randint(48, 55)
    is_minor = rng.random() < 0.5
    scale_steps = NATURAL_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    # Harmonizing intervals above root: 3rd (3/4), 4th (5), 5th (7), 6th (8/9), octave (12)
    harmonic_intervals = [3 if is_minor else 4, 5, 7, 8 if is_minor else 9, 12]

    notes: List[NoteEvent] = []
    current_bass_pitch = root

    for b in range(16):
        start_beat = float(b)
        interval = rng.choice(harmonic_intervals)
        treble_pitch = _clamp_pitch(current_bass_pitch + interval)

        # Both notes start at the exact same beat with 1.0 beat duration
        notes.append(
            NoteEvent(
                pitch=current_bass_pitch,
                start_beat=start_beat,
                duration_beats=1.0,
            )
        )
        notes.append(
            NoteEvent(
                pitch=treble_pitch,
                start_beat=start_beat,
                duration_beats=1.0,
            )
        )

        # Step movement for bass in scale: -2, -1, 0, +1, +2
        step = rng.choice([-2, -1, 0, 1, 2])
        current_bass_pitch = _clamp_pitch(current_bass_pitch + step)

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_2m(rng: random.Random) -> Score:
    """Level 2M: Slower bass pedal notes under moving treble melodies (60-85 BPM, 16 beats).

    Lower voice holds long drone/pedal notes (2.0 to 4.0 beats), while upper voice
    strikes new melody notes (1.0 beat) on every beat.
    """
    tempo = float(rng.randint(60, 85))
    bass_root = rng.randint(36, 48)  # C2 to C3
    is_minor = rng.random() < 0.5
    scale_steps = NATURAL_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    notes: List[NoteEvent] = []

    # 1. Generate lower voice (long pedal notes of 2.0 or 4.0 beats)
    beat = 0.0
    current_bass = bass_root
    while beat < 16.0:
        dur = rng.choice([2.0, 4.0])
        dur = min(dur, 16.0 - beat)
        notes.append(
            NoteEvent(
                pitch=current_bass,
                start_beat=round(beat, 2),
                duration_beats=dur,
            )
        )
        beat += dur
        current_bass = _clamp_pitch(current_bass + rng.choice([-2, 0, 2, 5]))

    # 2. Generate upper voice (moving melody notes of 1.0 beat)
    treble_root = bass_root + 24  # 2 octaves higher (C4 to C5)
    current_treble = treble_root
    for b in range(16):
        notes.append(
            NoteEvent(
                pitch=current_treble,
                start_beat=float(b),
                duration_beats=1.0,
            )
        )
        current_treble = _clamp_pitch(current_treble + rng.choice([-2, -1, 0, 1, 2, 3]))

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_3m(rng: random.Random) -> Score:
    """Level 3M: 3-note triad chords in left hand under right-hand melodies (70-95 BPM, 32 beats).

    Left hand plays block triads (root, third, fifth) every 2 beats (4 notes struck at chord onsets).
    Right hand plays melodies with 0.5 or 1.0 beat notes.
    """
    tempo = float(rng.randint(70, 95))
    root = rng.randint(45, 55)  # A2 to G3
    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    notes: List[NoteEvent] = []

    # 1. Left hand: block triads every 2.0 beats
    beat = 0.0
    while beat < 32.0:
        chord_dur = 2.0
        chord_root = _clamp_pitch(root + rng.choice([0, 2, 4, 5, 7]))
        triad_intervals = MINOR_TRIAD if (is_minor and rng.random() < 0.6) else MAJOR_TRIAD
        for offset in triad_intervals:
            p = _clamp_pitch(chord_root + offset)
            notes.append(
                NoteEvent(
                    pitch=p,
                    start_beat=round(beat, 2),
                    duration_beats=chord_dur,
                )
            )
        beat += chord_dur

    # 2. Right hand: melodic line (0.5 or 1.0 beat notes)
    beat = 0.0
    treble_pos = root + 16
    while beat < 32.0:
        dur = rng.choice([0.5, 1.0])
        dur = min(dur, 32.0 - beat)
        notes.append(
            NoteEvent(
                pitch=_clamp_pitch(treble_pos),
                start_beat=round(beat, 2),
                duration_beats=dur,
            )
        )
        beat += dur
        treble_pos += rng.choice([-2, -1, 0, 1, 2, 4])
        treble_pos = max(root + 12, min(root + 36, treble_pos))

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_4m(rng: random.Random) -> Score:
    """Level 4M: Alberti bass (broken arpeggiated accompaniment) under melody (70-100 BPM, 32 beats).

    Left hand plays broken chords (low - high - mid - high) in steady 8th notes (0.5 beats).
    Right hand plays melodies with 0.5, 1.0, or 2.0 beat durations.
    """
    tempo = float(rng.randint(70, 100))
    root = rng.randint(48, 55)  # C3 to G3
    is_minor = rng.random() < 0.5
    triad = MINOR_TRIAD if is_minor else MAJOR_TRIAD

    notes: List[NoteEvent] = []

    # 1. Left hand: Alberti pattern (root, fifth, third, fifth) every 0.5 beats
    beat = 0.0
    while beat < 32.0:
        c_root = root + rng.choice([0, 5, 7])
        # Pattern: low (root), high (fifth + 12), mid (third + 12), high (fifth + 12)
        p_low = _clamp_pitch(c_root)
        p_mid = _clamp_pitch(c_root + triad[1])
        p_high = _clamp_pitch(c_root + triad[2])
        pattern = [p_low, p_high, p_mid, p_high]

        for p in pattern:
            if beat >= 32.0:
                break
            notes.append(
                NoteEvent(
                    pitch=p,
                    start_beat=round(beat, 2),
                    duration_beats=0.5,
                )
            )
            beat += 0.5

    # 2. Right hand: melodic line with mixed durations [0.5, 1.0, 2.0]
    beat = 0.0
    treble_pos = root + 16
    while beat < 32.0:
        dur = rng.choice([0.5, 1.0, 2.0])
        dur = min(dur, 32.0 - beat)
        notes.append(
            NoteEvent(
                pitch=_clamp_pitch(treble_pos),
                start_beat=round(beat, 2),
                duration_beats=dur,
            )
        )
        beat += dur
        treble_pos += rng.choice([-3, -2, -1, 1, 2, 3])
        treble_pos = max(root + 12, min(root + 36, treble_pos))

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_5m(rng: random.Random) -> Score:
    """Level 5M: Full keyboard polyphony, chromatic harmony, minor & major keys (70-100 BPM, 32 beats).

    Uses all 12 chromatic keys across wide registers (24 to 96). Left hand plays dyads or
    triads, while right hand plays chromatic melodies.
    """
    tempo = float(rng.randint(70, 100))
    # Full range root across keyboard: A1 (33) to F4 (65)
    root = rng.randint(33, 65)
    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    notes: List[NoteEvent] = []

    # Left hand accompaniment (dyads or triads of 1.0 or 2.0 beats)
    beat = 0.0
    while beat < 32.0:
        dur = rng.choice([1.0, 2.0])
        dur = min(dur, 32.0 - beat)
        chord_root = _clamp_pitch(root + rng.choice([0, 2, 4, 5, 7, 9, 11]))
        triad = MINOR_TRIAD if (is_minor and rng.random() < 0.6) else MAJOR_TRIAD
        for offset in triad[: rng.choice([2, 3])]:
            notes.append(
                NoteEvent(
                    pitch=_clamp_pitch(chord_root + offset),
                    start_beat=round(beat, 2),
                    duration_beats=dur,
                )
            )
        beat += dur

    # Right hand: chromatic melody across upper registers
    beat = 0.0
    treble_pos = root + 24
    while beat < 32.0:
        dur = rng.choice([0.5, 1.0])
        dur = min(dur, 32.0 - beat)
        pitch = _clamp_pitch(treble_pos)
        # 20% chromatic alteration
        if rng.random() < 0.20:
            pitch = _clamp_pitch(pitch + rng.choice([-1, 1]))
        notes.append(
            NoteEvent(
                pitch=pitch,
                start_beat=round(beat, 2),
                duration_beats=dur,
            )
        )
        beat += dur
        treble_pos += rng.choice([-4, -2, -1, 1, 2, 4])
        treble_pos = max(root + 16, min(root + 44, treble_pos))

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_6m(rng: random.Random) -> Score:
    """Level 6M: Polyphonic scales & parallel harmony (75-105 BPM, 32 beats).

    Two voices moving together in parallel intervals (3rds, 6ths, or octaves)
    or contrary motion scales in major and minor keys.
    """
    tempo = float(rng.randint(75, 105))
    root = rng.randint(40, 60)
    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    # Build 2-octave scale
    scale_pitches = []
    for oct_idx in range(2):
        for s in scale_steps:
            p = _clamp_pitch(root + oct_idx * 12 + s)
            scale_pitches.append(p)

    notes: List[NoteEvent] = []
    beat = 0.0
    idx = 0
    interval = rng.choice([3, 4, 7, 8, 9, 12])  # 3rd, 5th, 6th, or octave

    while beat < 32.0:
        dur = rng.choice([0.5, 1.0])
        dur = min(dur, 32.0 - beat)
        p1 = scale_pitches[idx % len(scale_pitches)]
        p2 = _clamp_pitch(p1 + interval)

        # Both parallel notes struck together
        notes.append(NoteEvent(pitch=p1, start_beat=round(beat, 2), duration_beats=dur))
        notes.append(NoteEvent(pitch=p2, start_beat=round(beat, 2), duration_beats=dur))

        beat += dur
        idx = (idx + rng.choice([1, 2, -1])) % len(scale_pitches)

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_7m(rng: random.Random) -> Score:
    """Level 7M: Contrapuntal melodies with independent rhythms (80-110 BPM, 32 beats).

    Two independent musical voices with distinct syncopated rhythms, leaps, and rests,
    simulating two-part counterpoint (e.g. Bach Inventions).
    """
    tempo = float(rng.randint(80, 110))
    root = rng.randint(48, 60)
    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    notes: List[NoteEvent] = []

    # Voice 1 (Bass / Tenor counterpoint)
    beat = 0.0
    bass_pitch = root - 12
    while beat < 32.0:
        # 15% rest
        if rng.random() < 0.15 and beat + 1.0 <= 32.0:
            beat += rng.choice([0.5, 1.0])
            continue
        dur = rng.choice([0.5, 1.0, 2.0])
        dur = min(dur, 32.0 - beat)
        notes.append(
            NoteEvent(
                pitch=_clamp_pitch(bass_pitch),
                start_beat=round(beat, 2),
                duration_beats=dur,
            )
        )
        beat += dur
        bass_pitch = _clamp_pitch(bass_pitch + rng.choice([-3, -2, -1, 1, 2, 4]))

    # Voice 2 (Treble melody with independent mixed rhythms [0.25, 0.5, 1.0])
    beat = 0.0
    treble_pitch = root + 12
    while beat < 32.0:
        if rng.random() < 0.10 and beat + 0.5 <= 32.0:
            beat += 0.5
            continue
        dur = rng.choice([0.25, 0.5, 1.0])
        dur = min(dur, 32.0 - beat)
        notes.append(
            NoteEvent(
                pitch=_clamp_pitch(treble_pitch),
                start_beat=round(beat, 2),
                duration_beats=dur,
            )
        )
        beat += dur
        # Melodic leaps up to an octave
        leap = rng.choice([-12, -7, -4, -2, -1, 1, 2, 4, 7, 12])
        treble_pitch = _clamp_pitch(treble_pitch + leap)

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_8m(rng: random.Random) -> Score:
    """Level 8M: Fast polyphony: 16th-note runs over staccato chord punches (100-130 BPM, 32-48 beats).

    Right hand plays rapid runs of 16th notes (0.25 beats). Left hand punches sharp staccato
    chords (triads / octaves) on downbeats or syncopations.
    """
    tempo = float(rng.randint(100, 130))
    total_beats = float(rng.choice([32, 48]))
    root = rng.randint(48, 58)
    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS
    triad = MINOR_TRIAD if is_minor else MAJOR_TRIAD

    notes: List[NoteEvent] = []

    # 1. Left hand: staccato chord punches (0.25 or 0.5 duration) every 1.0 or 2.0 beats
    beat = 0.0
    while beat < total_beats:
        punch_dur = rng.choice([0.25, 0.5])
        c_root = _clamp_pitch(root - 12 + rng.choice([0, 5, 7]))
        for offset in triad:
            notes.append(
                NoteEvent(
                    pitch=_clamp_pitch(c_root + offset),
                    start_beat=round(beat, 2),
                    duration_beats=punch_dur,
                )
            )
        # Advance by 1 or 2 beats to next chord punch
        beat += rng.choice([1.0, 2.0])

    # 2. Right hand: rapid continuous 16th-note runs (0.25 beat each)
    beat = 0.0
    treble_pitch = root + 16
    while beat < total_beats:
        run_len = min(rng.choice([4, 8, 12]), int((total_beats - beat) / 0.25))
        direction = rng.choice([-1, 1])
        for _ in range(run_len):
            if beat >= total_beats:
                break
            notes.append(
                NoteEvent(
                    pitch=_clamp_pitch(treble_pitch),
                    start_beat=round(beat, 2),
                    duration_beats=0.25,
                )
            )
            beat += 0.25
            treble_pitch += direction * rng.choice([1, 2])
            treble_pitch = max(root + 8, min(root + 36, treble_pitch))
        # Occasional leap between runs
        treble_pitch = _clamp_pitch(treble_pitch + rng.choice([-7, 7, -12, 12]))

    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch))
    return Score(notes=notes, tempo_bpm=tempo)


def _deduplicate_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Merge duplicate notes with the exact same pitch and start beat (unisons).

    If two procedural voices strike the exact same physical piano key at
    the same moment, the key is pressed once; keep the longer duration.
    """
    notes.sort(key=lambda n: (round(n.start_beat, 4), n.pitch, -n.duration_beats))
    deduped = []
    seen = set()
    for n in notes:
        key = (n.pitch, round(n.start_beat, 4))
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    return deduped


def normalize_level_tag(lvl: Union[int, str]) -> str:
    """Standardize curriculum level input (e.g., 1, '1', '1m', '1M') to '1M'."""
    clean = str(lvl).strip().upper()
    if not clean.endswith("M"):
        clean += "M"
    return clean


def generate_multi_key_score(
    level: Union[int, str], seed: Optional[int] = None
) -> Score:
    """Generate a procedural polyphonic musical Score for a given multi-key level (1M to 8M).

    Args:
        level: Level identifier: integer 1..8 or string "1M".."8M" (case-insensitive).
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        Score: A polyphonic musical Score matching the difficulty specifications.
    """
    if isinstance(level, str):
        clean_lvl = level.upper().rstrip("M")
        try:
            lvl_num = int(clean_lvl)
        except ValueError:
            raise ValueError(
                f"Invalid level '{level}'. Expected 1..8 or '1M'..'8M'."
            )
    elif isinstance(level, int):
        lvl_num = level
    else:
        raise TypeError(f"Level must be int or str, got {type(level)}")

    if lvl_num not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError(
            f"Level must be between 1 and 8 (or '1M'..'8M'); got {level}"
        )

    rng = random.Random(seed)
    generators = {
        1: generate_level_1m,
        2: generate_level_2m,
        3: generate_level_3m,
        4: generate_level_4m,
        5: generate_level_5m,
        6: generate_level_6m,
        7: generate_level_7m,
        8: generate_level_8m,
    }
    raw_score = generators[lvl_num](rng)
    clean_notes = _deduplicate_notes(raw_score.notes)
    return Score(notes=clean_notes, tempo_bpm=raw_score.tempo_bpm)

