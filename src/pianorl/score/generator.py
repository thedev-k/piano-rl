import random
from typing import List, Optional, Tuple

from .score import Score, NoteEvent
from .window import MIN_PIANO_PITCH, MAX_PIANO_PITCH

WHITE_KEY_SEMITONES = {0, 2, 4, 5, 7, 9, 11}
ALL_WHITE_KEYS = [p for p in range(MIN_PIANO_PITCH, MAX_PIANO_PITCH + 1) if p % 12 in WHITE_KEY_SEMITONES]

MAJOR_SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]
HARMONIC_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 11]

MAJOR_ARPEGGIO_STEPS = [0, 4, 7, 12]
MINOR_ARPEGGIO_STEPS = [0, 3, 7, 12]


def _get_five_white_keys(rng: random.Random) -> List[int]:
    """Pick 5 consecutive white keys starting between C3 (48) and C5 (72)."""
    # Candidate starting keys are white keys in [48, 72]
    candidates = [k for k in ALL_WHITE_KEYS if 48 <= k <= 72]
    start_key = rng.choice(candidates)
    start_idx = ALL_WHITE_KEYS.index(start_key)
    # Ensure 5 keys are available
    if start_idx + 5 > len(ALL_WHITE_KEYS):
        start_idx = len(ALL_WHITE_KEYS) - 5
    return ALL_WHITE_KEYS[start_idx : start_idx + 5]


def generate_level_1(rng: random.Random) -> Score:
    """Level 1: Single notes, 1 beat each, within a 5-key range, 16 beats, 60-80 BPM."""
    five_keys = _get_five_white_keys(rng)
    tempo = float(rng.randint(60, 80))
    notes: List[NoteEvent] = []

    for b in range(16):
        pitch = rng.choice(five_keys)
        notes.append(NoteEvent(pitch=pitch, start_beat=float(b), duration_beats=1.0))

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_2(rng: random.Random) -> Score:
    """Level 2: Stepwise melodies (small skips) in 5-key range, 1 or 2 beats, 16 beats, 60-90 BPM."""
    five_keys = _get_five_white_keys(rng)
    tempo = float(rng.randint(60, 90))
    notes: List[NoteEvent] = []

    current_beat = 0.0
    current_idx = rng.randint(0, 4)

    while current_beat < 16.0:
        remaining = 16.0 - current_beat
        if remaining >= 2.0:
            dur = 1.0 if rng.random() < 0.75 else 2.0
        else:
            dur = 1.0

        pitch = five_keys[current_idx]
        notes.append(NoteEvent(pitch=pitch, start_beat=round(current_beat, 2), duration_beats=dur))
        current_beat += dur

        # Movement: 70% step (+-1), 20% skip (+-2), 10% stay
        r = rng.random()
        if r < 0.70:
            step = rng.choice([-1, 1])
        elif r < 0.90:
            step = rng.choice([-2, 2])
        else:
            step = 0
        current_idx = max(0, min(4, current_idx + step))

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_3(rng: random.Random) -> Score:
    """Level 3: Scales and arpeggios in major key, 1 or 2 octaves, 0.5 or 1 beat, 32 beats, 70-100 BPM."""
    tempo = float(rng.randint(70, 100))

    # Root key between C3 (48) and G3 (55) to comfortably span 1-2 octaves within 88 keys
    root = rng.randint(48, 55)
    num_octaves = rng.choice([1, 2])

    # Build scale pitches
    scale_pitches = []
    for oct_idx in range(num_octaves):
        for step in MAJOR_SCALE_STEPS:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                scale_pitches.append(p)
    top_note = root + num_octaves * 12
    if MIN_PIANO_PITCH <= top_note <= MAX_PIANO_PITCH:
        scale_pitches.append(top_note)

    # Build arpeggio pitches
    arpeggio_pitches = []
    for oct_idx in range(num_octaves):
        for step in MAJOR_ARPEGGIO_STEPS:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                arpeggio_pitches.append(p)
    if MIN_PIANO_PITCH <= top_note <= MAX_PIANO_PITCH:
        arpeggio_pitches.append(top_note)
    arpeggio_pitches = sorted(list(set(arpeggio_pitches)))

    patterns = [
        scale_pitches,                      # scale up
        list(reversed(scale_pitches)),      # scale down
        arpeggio_pitches,                   # arpeggio up
        list(reversed(arpeggio_pitches)),   # arpeggio down
    ]

    notes: List[NoteEvent] = []
    current_beat = 0.0

    while current_beat < 32.0:
        pattern = rng.choice(patterns)
        dur_choice = rng.choice([0.5, 1.0])

        for pitch in pattern:
            if current_beat >= 32.0:
                break
            dur = min(dur_choice, 32.0 - current_beat)
            notes.append(NoteEvent(pitch=pitch, start_beat=round(current_beat, 2), duration_beats=dur))
            current_beat += dur

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_4(rng: random.Random) -> Score:
    """Level 4: Melodies in major key across up to 2 octaves, durations [0.5, 1.0, 2.0], occasional rests, 32 beats, 70-110 BPM."""
    tempo = float(rng.randint(70, 110))

    root = rng.randint(48, 55)
    scale_pitches = []
    for oct_idx in range(2):
        for step in MAJOR_SCALE_STEPS:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                scale_pitches.append(p)
    top_note = root + 24
    if MIN_PIANO_PITCH <= top_note <= MAX_PIANO_PITCH:
        scale_pitches.append(top_note)

    notes: List[NoteEvent] = []
    current_beat = 0.0
    current_idx = rng.randint(0, len(scale_pitches) - 1)

    while current_beat < 32.0:
        remaining = 32.0 - current_beat

        # 12% chance of a short rest (0.5 or 1.0 beat) if sufficient time remains
        if rng.random() < 0.12 and remaining > 1.0:
            rest_len = 0.5 if rng.random() < 0.6 else 1.0
            rest_len = min(rest_len, remaining - 0.5)
            current_beat += rest_len
            remaining = 32.0 - current_beat

        # Pick duration from [0.5, 1.0, 2.0]
        valid_durs = [d for d in [0.5, 1.0, 2.0] if d <= remaining]
        if not valid_durs:
            dur = remaining
        else:
            dur = rng.choice(valid_durs)

        pitch = scale_pitches[current_idx]
        notes.append(NoteEvent(pitch=pitch, start_beat=round(current_beat, 2), duration_beats=dur))
        current_beat += dur

        # Melodic movement: mostly step or small skip, occasional jump
        r = rng.random()
        if r < 0.65:
            delta = rng.choice([-1, 1])
        elif r < 0.85:
            delta = rng.choice([-2, 2])
        elif r < 0.95:
            delta = rng.choice([-3, 3])
        else:
            delta = rng.choice([-4, 4])
        current_idx = max(0, min(len(scale_pitches) - 1, current_idx + delta))

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_5(rng: random.Random) -> Score:
    """Level 5: Single notes & small-step melodies using all 12 notes across the 88 keys, slow tempo (60-80 BPM)."""
    tempo = float(rng.randint(60, 80))
    # Pick a starting pitch anywhere across the 88-key keyboard
    current_pitch = rng.randint(MIN_PIANO_PITCH + 4, MAX_PIANO_PITCH - 4)
    notes: List[NoteEvent] = []
    current_beat = 0.0

    while current_beat < 16.0:
        remaining = 16.0 - current_beat
        if remaining >= 2.0:
            dur = 1.0 if rng.random() < 0.75 else 2.0
        else:
            dur = 1.0

        notes.append(NoteEvent(pitch=current_pitch, start_beat=round(current_beat, 2), duration_beats=dur))
        current_beat += dur

        # Movement: chromatic / small steps (including all black keys)
        r = rng.random()
        if r < 0.40:
            step = rng.choice([-1, 1])       # semitone
        elif r < 0.75:
            step = rng.choice([-2, 2])       # whole tone
        elif r < 0.90:
            step = rng.choice([-3, -4, 3, 4]) # minor / major 3rd
        else:
            step = 0                         # repeated note

        current_pitch = max(MIN_PIANO_PITCH, min(MAX_PIANO_PITCH, current_pitch + step))

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_6(rng: random.Random) -> Score:
    """Level 6: Major and minor scales & arpeggios in all 12 keys, 1 to 3 octaves, anywhere on the 88 keys (70-100 BPM)."""
    tempo = float(rng.randint(70, 100))

    # All 12 keys, 1 to 3 octaves
    num_octaves = rng.choice([1, 2, 3])
    max_root = MAX_PIANO_PITCH - num_octaves * 12
    min_root = MIN_PIANO_PITCH
    root = rng.randint(min_root, max_root)

    mode = rng.choice(["major", "natural_minor", "harmonic_minor"])
    if mode == "major":
        scale_steps = MAJOR_SCALE_STEPS
        arp_steps = MAJOR_ARPEGGIO_STEPS
    elif mode == "natural_minor":
        scale_steps = NATURAL_MINOR_STEPS
        arp_steps = MINOR_ARPEGGIO_STEPS
    else:
        scale_steps = HARMONIC_MINOR_STEPS
        arp_steps = MINOR_ARPEGGIO_STEPS

    # Build scale pitches
    scale_pitches = []
    for oct_idx in range(num_octaves):
        for step in scale_steps:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                scale_pitches.append(p)
    top_note = root + num_octaves * 12
    if MIN_PIANO_PITCH <= top_note <= MAX_PIANO_PITCH:
        scale_pitches.append(top_note)

    # Build arpeggio pitches
    arpeggio_pitches = []
    for oct_idx in range(num_octaves):
        for step in arp_steps:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                arpeggio_pitches.append(p)
    if MIN_PIANO_PITCH <= top_note <= MAX_PIANO_PITCH:
        arpeggio_pitches.append(top_note)
    arpeggio_pitches = sorted(list(set(arpeggio_pitches)))

    patterns = [
        scale_pitches,
        list(reversed(scale_pitches)),
        arpeggio_pitches,
        list(reversed(arpeggio_pitches)),
    ]

    notes: List[NoteEvent] = []
    current_beat = 0.0

    while current_beat < 32.0:
        pattern = rng.choice(patterns)
        dur_choice = rng.choice([0.5, 1.0])

        for pitch in pattern:
            if current_beat >= 32.0:
                break
            dur = min(dur_choice, 32.0 - current_beat)
            notes.append(NoteEvent(pitch=pitch, start_beat=round(current_beat, 2), duration_beats=dur))
            current_beat += dur

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_7(rng: random.Random) -> Score:
    """Level 7: Wide-range melodies with leaps up to an octave+, chromatic passing notes, rests, mixed lengths (75-105 BPM)."""
    tempo = float(rng.randint(75, 105))

    # Random root in any of the 12 keys
    root = rng.randint(MIN_PIANO_PITCH + 12, MAX_PIANO_PITCH - 36)
    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    # Build up to 3 octaves of scale pitches
    scale_pitches = []
    for oct_idx in range(3):
        for step in scale_steps:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                scale_pitches.append(p)
    top_note = root + 36
    if MIN_PIANO_PITCH <= top_note <= MAX_PIANO_PITCH:
        scale_pitches.append(top_note)

    notes: List[NoteEvent] = []
    current_beat = 0.0
    current_idx = rng.randint(0, len(scale_pitches) - 1)

    while current_beat < 32.0:
        remaining = 32.0 - current_beat

        # 15% chance of rest (0.5 or 1.0 beat) if remaining > 1.5
        if rng.random() < 0.15 and remaining > 1.5:
            rest_len = 0.5 if rng.random() < 0.6 else 1.0
            rest_len = min(rest_len, remaining - 0.5)
            current_beat += rest_len
            remaining = 32.0 - current_beat

        # Mixed note lengths: 0.25, 0.5, 1.0, 2.0 (all multiples of 0.25)
        valid_durs = [d for d in [0.25, 0.5, 1.0, 2.0] if d <= remaining]
        dur = rng.choice(valid_durs) if valid_durs else remaining

        pitch = scale_pitches[current_idx]

        # 18% chance of chromatic passing note
        if rng.random() < 0.18:
            offset = rng.choice([-1, 1])
            chromatic_pitch = pitch + offset
            if MIN_PIANO_PITCH <= chromatic_pitch <= MAX_PIANO_PITCH:
                pitch = chromatic_pitch

        notes.append(NoteEvent(pitch=pitch, start_beat=round(current_beat, 2), duration_beats=dur))
        current_beat += dur

        # Melodic movement: steps, skips, and wide leaps up to an octave or more
        r = rng.random()
        if r < 0.50:
            delta = rng.choice([-1, 1, -2, 2])
        elif r < 0.75:
            delta = rng.choice([-3, 3, -4, 4])
        elif r < 0.90:
            delta = rng.choice([-7, 7, -8, 8])
        else:
            delta = rng.choice([-12, 12, -14, 14])

        current_idx = max(0, min(len(scale_pitches) - 1, current_idx + delta))

    return Score(notes=notes, tempo_bpm=tempo)


def generate_level_8(rng: random.Random) -> Score:
    """Level 8: Fast pieces with 16th-note runs, repeated notes, higher tempos (100-130 BPM), >=32 beats, full range, major & minor."""
    tempo = float(rng.randint(100, 130))
    total_target_beats = float(rng.choice([32, 48]))

    is_minor = rng.random() < 0.5
    scale_steps = HARMONIC_MINOR_STEPS if is_minor else MAJOR_SCALE_STEPS

    root = rng.randint(MIN_PIANO_PITCH + 5, MAX_PIANO_PITCH - 48)
    scale_pitches = []
    for oct_idx in range(4):
        for step in scale_steps:
            p = root + oct_idx * 12 + step
            if MIN_PIANO_PITCH <= p <= MAX_PIANO_PITCH:
                scale_pitches.append(p)
    scale_pitches = sorted(list(set(scale_pitches)))

    notes: List[NoteEvent] = []
    current_beat = 0.0
    current_idx = rng.randint(0, len(scale_pitches) - 1)

    while current_beat < total_target_beats:
        remaining = total_target_beats - current_beat
        mode_choice = rng.random()

        # Mode A (50%): Fast 16th-note run (4 to 8 notes in rapid sequence)
        if mode_choice < 0.50 and remaining >= 1.0:
            run_len = min(rng.choice([4, 8]), int(remaining / 0.25))
            step_dir = rng.choice([-1, 1])
            for _ in range(run_len):
                if current_beat >= total_target_beats:
                    break
                p = scale_pitches[current_idx]
                notes.append(NoteEvent(pitch=p, start_beat=round(current_beat, 2), duration_beats=0.25))
                current_beat += 0.25
                current_idx = max(0, min(len(scale_pitches) - 1, current_idx + step_dir))

        # Mode B (30%): Rapid repeated notes (2 to 4 strikes of same key)
        elif mode_choice < 0.80 and remaining >= 0.5:
            repeats = min(rng.choice([2, 3, 4]), int(remaining / 0.25))
            rep_dur = 0.25 if rng.random() < 0.7 else 0.5
            p = scale_pitches[current_idx]
            for _ in range(repeats):
                if current_beat + rep_dur > total_target_beats:
                    break
                notes.append(NoteEvent(pitch=p, start_beat=round(current_beat, 2), duration_beats=rep_dur))
                current_beat += rep_dur
            current_idx = max(0, min(len(scale_pitches) - 1, current_idx + rng.choice([-2, -1, 1, 2])))

        # Mode C (20%): Fast leap or arpeggio burst
        else:
            dur = min(rng.choice([0.25, 0.5]), remaining)
            p = scale_pitches[current_idx]
            notes.append(NoteEvent(pitch=p, start_beat=round(current_beat, 2), duration_beats=dur))
            current_beat += dur
            leap = rng.choice([-7, -5, -4, 4, 5, 7, 12])
            current_idx = max(0, min(len(scale_pitches) - 1, current_idx + leap))

    return Score(notes=notes, tempo_bpm=tempo)


def generate_score(level: int, seed: Optional[int] = None) -> Score:
    """Generate a procedural musical Score for a given difficulty level (1 to 8).

    Args:
        level: Integer from 1 to 8.
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        Score: A procedural Score matching the difficulty specifications.
    """
    if level not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"Level must be between 1 and 8; got {level}")

    rng = random.Random(seed)
    generators = {
        1: generate_level_1,
        2: generate_level_2,
        3: generate_level_3,
        4: generate_level_4,
        5: generate_level_5,
        6: generate_level_6,
        7: generate_level_7,
        8: generate_level_8,
    }
    return generators[level](rng)

