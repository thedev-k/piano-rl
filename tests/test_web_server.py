"""Tests for web server basic functionality."""

import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add scripts to path to import web_server
sys.path.insert(0, os.path.abspath("scripts"))
from web_server import app

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
            assert note["velocity"] == 0.8
            
        # We should then receive at least one step message
        step_data = websocket.receive_json()
        assert step_data["type"] in ("step", "done")
        
        # Stop playback to end test cleanly
        websocket.send_json({"action": "stop"})
