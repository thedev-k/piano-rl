import pytest
from pathlib import Path
import mido

from pianorl.score import Score, NoteEvent, load_score, pitch_to_note_name
from scripts.make_test_midi import create_midi_file


@pytest.fixture(scope="session")
def examples_dir(tmp_path_factory) -> Path:
    """Fixture that generates test MIDI files in a temporary directory."""
    temp_dir = tmp_path_factory.mktemp("midi_examples")

    # 1. C major scale
    scale_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    scale_notes = [(p, float(i), 1.0) for i, p in enumerate(scale_pitches)]
    create_midi_file(temp_dir / "c_major_scale.mid", scale_notes, tempo_bpm=120.0)

    # 2. Twinkle Twinkle melody
    melody_notes = [
        (60, 0.0, 1.0),
        (60, 1.0, 1.0),
        (67, 2.0, 1.0),
        (67, 3.0, 1.0),
        (69, 4.0, 1.0),
        (69, 5.0, 1.0),
        (67, 6.0, 2.0),
    ]
    create_midi_file(temp_dir / "melody.mid", melody_notes, tempo_bpm=100.0)

    return temp_dir


def test_pitch_to_note_name():
    assert pitch_to_note_name(60) == "C4"
    assert pitch_to_note_name(61) == "C#4"
    assert pitch_to_note_name(72) == "C5"
    assert pitch_to_note_name(21) == "A0"
    assert pitch_to_note_name(108) == "C8"


def test_c_major_scale_loading(examples_dir: Path):
    score = load_score(examples_dir / "c_major_scale.mid")

    assert isinstance(score, Score)
    assert score.tempo_bpm == 120.0
    assert len(score) == 8

    expected_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    expected_names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

    for i, note in enumerate(score.notes):
        assert note.pitch == expected_pitches[i]
        assert note.note_name == expected_names[i]
        assert note.start_beat == pytest.approx(float(i))
        assert note.duration_beats == pytest.approx(1.0)
        assert note.end_beat == pytest.approx(float(i) + 1.0)

    assert score.total_beats == pytest.approx(8.0)


def test_melody_loading(examples_dir: Path):
    score = load_score(examples_dir / "melody.mid")

    assert len(score) == 7
    assert score.tempo_bpm == 100.0

    # Check start beats and durations
    expected = [
        (60, 0.0, 1.0),  # C4
        (60, 1.0, 1.0),  # C4
        (67, 2.0, 1.0),  # G4
        (67, 3.0, 1.0),  # G4
        (69, 4.0, 1.0),  # A4
        (69, 5.0, 1.0),  # A4
        (67, 6.0, 2.0),  # G4 (half note: 2 beats long)
    ]

    for note, (exp_pitch, exp_start, exp_dur) in zip(score.notes, expected):
        assert note.pitch == exp_pitch
        assert note.start_beat == pytest.approx(exp_start)
        assert note.duration_beats == pytest.approx(exp_dur)

    assert score.total_beats == pytest.approx(8.0)


def test_events_are_sorted(tmp_path: Path):
    # Create MIDI with deliberately unsorted input order
    unsorted_notes = [
        (67, 4.0, 1.0),
        (60, 0.0, 1.0),
        (64, 2.0, 1.0),
    ]
    file_path = tmp_path / "unsorted.mid"
    create_midi_file(file_path, unsorted_notes, tempo_bpm=120.0)

    score = load_score(file_path)
    start_beats = [n.start_beat for n in score.notes]

    assert start_beats == sorted(start_beats)
    assert [n.pitch for n in score.notes] == [60, 64, 67]


def test_missing_file_error():
    missing_path = "non_existent_file_12345.mid"
    with pytest.raises(FileNotFoundError, match="MIDI file not found"):
        load_score(missing_path)


def test_empty_file_error(tmp_path: Path):
    empty_file = tmp_path / "empty.mid"
    empty_file.touch()  # Creates a 0-byte file

    with pytest.raises(ValueError, match="MIDI file is empty"):
        load_score(empty_file)


def test_tempo_change_warning(tmp_path: Path):
    file_path = tmp_path / "tempo_change.mid"
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Initial tempo 120 BPM
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120.0), time=0))
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))

    # Tempo change to 160 BPM
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(160.0), time=0))
    track.append(mido.Message("note_on", note=62, velocity=64, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=480))

    mid.save(str(file_path))

    with pytest.warns(UserWarning, match="Tempo change detected"):
        score = load_score(file_path)

    # Must retain the first tempo
    assert score.tempo_bpm == 120.0
    assert len(score) == 2
