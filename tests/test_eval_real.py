from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pytest
from pianorl.eval import PPOPlayer
from pianorl.score import Score, NoteEvent
from scripts.eval_real import (
    check_chords_or_simultaneous_notes,
    evaluate_real_folder,
    evaluate_single_piece,
    plot_piano_roll_comparison,
)


def test_chord_detection_logic():
    """Verify check_chords_or_simultaneous_notes distinguishes monophonic vs chord scores."""
    # Monophonic score: notes at beat 0, 1, 2
    mono_score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=62, start_beat=1.0, duration_beats=1.0),
            NoteEvent(pitch=64, start_beat=2.0, duration_beats=1.0),
        ],
        tempo_bpm=100.0,
    )
    assert not check_chords_or_simultaneous_notes(mono_score)

    # Chord score: two notes starting at beat 1.0 (e.g. C4 and E4)
    chord_score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0),
            NoteEvent(pitch=64, start_beat=1.0, duration_beats=1.0),
        ],
        tempo_bpm=100.0,
    )
    assert check_chords_or_simultaneous_notes(chord_score)


def test_plot_piano_roll_generation(tmp_path: Path):
    """Verify plot_piano_roll_comparison generates a valid non-empty PNG image."""
    score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=62, start_beat=1.0, duration_beats=1.0),
        ],
        tempo_bpm=120.0,
    )
    model_presses = [
        {"step": 0, "beat": 0.0, "pitch": 60, "result": "exact"},
        {"step": 4, "beat": 1.0, "pitch": 63, "result": "wrong"},
    ]
    img_path = tmp_path / "test_pianoroll.png"

    plot_piano_roll_comparison(
        score=score,
        model_presses=model_presses,
        piece_name="test_piece",
        output_path=img_path,
        precision=0.5,
        recall=0.5,
        f1=0.5,
    )

    assert img_path.exists()
    assert img_path.stat().st_size > 1000  # Valid PNG image


def test_eval_real_folder_runs(tmp_path: Path):
    """Verify evaluate_real_folder runs on data/real with an existing model."""
    data_dir = Path("data/real")
    assert data_dir.exists()
    assert len(list(data_dir.glob("*.mid"))) == 4

    # Use any existing trained checkpoint
    model_path = Path("checkpoints/all_pitch_conv/final.zip")
    if not model_path.exists():
        pytest.skip("Model checkpoint not found")

    output_dir = tmp_path / "outputs" / "real"
    results = evaluate_real_folder(data_dir=data_dir, model_path=model_path, output_dir=output_dir)

    assert "Overall" in results
    assert "twinkle_twinkle.mid" in results
    assert "ode_to_joy.mid" in results
    assert "mary_had_a_little_lamb.mid" in results
    assert "jingle_bells.mid" in results

    # Verify all 4 images were generated
    assert (output_dir / "twinkle_twinkle_pianoroll.png").exists()
    assert (output_dir / "ode_to_joy_pianoroll.png").exists()
    assert (output_dir / "mary_had_a_little_lamb_pianoroll.png").exists()
    assert (output_dir / "jingle_bells_pianoroll.png").exists()
