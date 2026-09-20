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
