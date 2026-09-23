"""Tests for web server basic functionality."""

import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path
import numpy as np

# Add scripts to path to import web_server
sys.path.insert(0, os.path.abspath("scripts"))
from web_server import app, is_player_multikey

client = TestClient(app)

def test_api_files():
    """Test that the /api/files endpoint returns a list."""
    response = client.get("/api/files")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_static_files():
    """Test that index.html is served at root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Piano-RL Live Player" in response.text


def test_audio_samples_served():
    """Test that the local Salamander Grand Piano audio samples are served."""
    response = client.get("/audio/salamander/C4.mp3")
    assert response.status_code == 200
    assert len(response.content) > 1000  # Non-empty audio file

def test_websocket_play():
    """Test that the websocket can load a real score and start playback without crashing."""
    # Find a real midi file to test with
    test_file = None
    real_dir = Path("data/real")
    if real_dir.exists():
        mid_files = list(real_dir.glob("*.mid"))
        if mid_files:
            test_file = mid_files[0]
            
    if not test_file:
        pytest.skip("No real midi file found in data/real to test with.")
        
    class DummyPlayer:
        def act(self, obs):
            return 0 # do nothing
            
    import web_server
    web_server.PLAYER = DummyPlayer()
        
    with client.websocket_connect("/ws") as websocket:
        # Send play command
        websocket.send_json({"action": "play", "file": str(test_file)})
        
        # We should receive an init message first
        data = websocket.receive_json()
        assert data["type"] == "init"
        
        # Check notes data has correct fields
        notes = data["notes"]
        assert len(notes) > 0
        for note in notes:
            assert "pitch" in note
            assert "start_beat" in note
            assert "duration_beats" in note
            assert "velocity" in note
            assert 0.0 <= note["velocity"] <= 1.0
            
        # We should then receive at least one step message
        step_data = websocket.receive_json()
        assert step_data["type"] in ("step", "done")
        
        # Stop playback to end test cleanly
        websocket.send_json({"action": "stop"})

def test_piece_info_endpoint():
    """Test that /api/piece-info returns valid info for a real score."""
    test_file = Path("data/real/twinkle_twinkle.mid")
    if not test_file.exists():
        pytest.skip("twinkle_twinkle.mid not found")

    response = client.get(f"/api/piece-info?path={test_file}")
    assert response.status_code == 200
    data = response.json()
    assert "info" in data
    assert "melody_info" in data
    assert "dropped_notes" in data
    assert data["info"]["num_notes"] > 0
    assert "lowest_key" in data["info"]
    assert "highest_key" in data["info"]

def test_upload_endpoint(tmp_path: Path):
    """Test that /api/upload saves and validates an uploaded MIDI file."""
    from pianorl.score import Score, NoteEvent, save_score_to_midi
    test_score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=64, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=67, start_beat=1.0, duration_beats=1.0),
        ],
        tempo_bpm=120.0,
    )
    midi_path = tmp_path / "custom_upload.mid"
    save_score_to_midi(test_score, midi_path)
    midi_bytes = midi_path.read_bytes()

    response = client.post(
        "/api/upload?filename=custom_upload.mid",
        content=midi_bytes,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["file"]["name"] == "custom_upload.mid"
    assert data["info"]["num_notes"] == 3
    assert data["info"]["num_chords"] == 1
    assert data["dropped_notes"] == 1

    uploaded_file = Path(data["file"]["path"])
    assert uploaded_file.exists()

def test_websocket_play_melody_only(tmp_path: Path):
    """Test that websocket play with melody_only=True drops lower chord notes."""
    from pianorl.score import Score, NoteEvent, save_score_to_midi
    chord_score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=67, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=64, start_beat=1.0, duration_beats=1.0),
        ],
        tempo_bpm=120.0,
    )
    test_path = tmp_path / "chord_play.mid"
    save_score_to_midi(chord_score, test_path)

    class DummyPlayer:
        def act(self, obs):
            return 0
            
    import web_server
    web_server.PLAYER = DummyPlayer()

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"action": "play", "file": str(test_path), "melody_only": True})
        data = websocket.receive_json()
        assert data["type"] == "init"
        assert data["melody_only"] is True
        assert data["dropped_notes"] == 1
        assert len(data["notes"]) == 2
        assert data["notes"][0]["pitch"] == 67

        step_data = websocket.receive_json()
        assert step_data["type"] in ("step", "done")
        websocket.send_json({"action": "stop"})


def test_websocket_play_multikey_chords(tmp_path: Path):
    """Test that websocket play with a multi-key model streams multiple pitches simultaneously."""
    from pianorl.score import Score, NoteEvent, save_score_to_midi
    from pianorl.agent import PerfectMultiPlayer

    # C major triad on beat 0
    chord_score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=64, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=67, start_beat=0.0, duration_beats=1.0),
        ],
        tempo_bpm=120.0,
    )
    test_path = tmp_path / "multikey_chord.mid"
    save_score_to_midi(chord_score, test_path)

    import web_server
    web_server.PLAYER = PerfectMultiPlayer()
    assert is_player_multikey(web_server.PLAYER) is True

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"action": "play", "file": str(test_path), "melody_only": False})
        init_data = websocket.receive_json()
        assert init_data["type"] == "init"
        assert init_data["model_type"] == "multikey"
        assert len(init_data["notes"]) == 3

        # First step: perfect multi-player should strike all 3 chord notes [60, 64, 67]
        step_data = websocket.receive_json()
        assert step_data["type"] == "step"
        assert step_data["is_multikey"] is True
        assert "pitches" in step_data
        assert set(step_data["pitches"]) == {60, 64, 67}
        assert step_data["result"] == "exact"
        assert step_data["results"]["60"] == "exact"
        assert step_data["results"]["64"] == "exact"
        assert step_data["results"]["67"] == "exact"

        # Check notes payload has duration and velocity
        assert "notes" in step_data
        assert len(step_data["notes"]) == 3
        for n_info in step_data["notes"]:
            assert "pitch" in n_info
            assert "duration" in n_info
            assert "velocity" in n_info
            # 1 beat at 120 BPM = 0.5s duration
            assert n_info["duration"] == pytest.approx(0.5, rel=0.1)
            assert 0.1 <= n_info["velocity"] <= 1.0

        websocket.send_json({"action": "stop"})


def test_websocket_play_pedal_and_velocity(tmp_path: Path):
    """Test that note velocity and sustain pedal CC 64 are delivered over websocket."""
    from pianorl.score import Score, NoteEvent, save_score_to_midi
    from pianorl.agent import PerfectMultiPlayer

    # Note with velocity 110, held for 0.5 beats, but pedal active until beat 3.0
    pedal_score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=0.5, velocity=110),
        ],
        tempo_bpm=120.0,
        pedal_intervals=[(0.0, 3.0)],
    )
    test_path = tmp_path / "pedal_test.mid"
    save_score_to_midi(pedal_score, test_path)

    import web_server
    web_server.PLAYER = PerfectMultiPlayer()

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"action": "play", "file": str(test_path)})
        init_data = websocket.receive_json()
        assert init_data["type"] == "init"
        assert init_data["has_pedal"] is True
        assert init_data["notes"][0]["velocity"] == pytest.approx(110 / 127.0, abs=0.01)

        step_data = websocket.receive_json()
        assert step_data["type"] == "step"
        assert step_data["pedal"] is True
        assert len(step_data["notes"]) == 1
        note_info = step_data["notes"][0]
        assert note_info["pitch"] == 60
        assert note_info["velocity"] == pytest.approx(110 / 127.0, abs=0.01)
        # Note duration was sustained by pedal until beat 3.0 (3 beats at 120 BPM = 1.5 seconds)
        assert note_info["duration"] == pytest.approx(1.5, rel=0.1)

        websocket.send_json({"action": "stop"})

