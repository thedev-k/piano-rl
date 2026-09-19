import random
from typing import List, Optional, Tuple

from .score import Score, NoteEvent
from .window import MIN_PIANO_PITCH, MAX_PIANO_PITCH

WHITE_KEY_SEMITONES = {0, 2, 4, 5, 7, 9, 11}
ALL_WHITE_KEYS = [p for p in range(MIN_PIANO_PITCH, MAX_PIANO_PITCH + 1) if p % 12 in WHITE_KEY_SEMITONES]

MAJOR_SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11]
MAJOR_ARPEGGIO_STEPS = [0, 4, 7, 12]


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


def generate_score(level: int, seed: Optional[int] = None) -> Score:
    """Generate a procedural musical Score for a given difficulty level (1 to 4).

    Args:
        level: Integer from 1 to 4.
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        Score: A procedural Score matching the difficulty specifications.
    """
    if level not in (1, 2, 3, 4):
        raise ValueError(f"Level must be 1, 2, 3, or 4; got {level}")

    rng = random.Random(seed)
    generators = {
        1: generate_level_1,
        2: generate_level_2,
        3: generate_level_3,
        4: generate_level_4,
    }
    return generators[level](rng)
