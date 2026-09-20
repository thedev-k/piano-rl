"""FastAPI web server for live piano playing."""

import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pianorl.env import PianoFreeKeysEnv, RewardConfig
from pianorl.eval import PPOPlayer
from pianorl.score import Score, load_score, analyze_score, extract_melody

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model state
PLAYER = None
DATA_DIRS = [Path("data/real"), Path("data/heldout"), Path("data/uploads")]

@app.get("/api/files")
def get_files():
    """List available MIDI files."""
    files = []
    for d in DATA_DIRS:
        if d.exists():
            for f in sorted(d.glob("*.mid*")):
                files.append({"folder": d.name, "name": f.name, "path": str(f).replace('\\', '/')})
    return files

@app.get("/api/piece-info")
def get_piece_info(path: str):
    """Return musical analysis of a piece, both original and melody-only."""
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        score = load_score(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse MIDI: {e}")

    info = analyze_score(score)
    melody_score, dropped_notes = extract_melody(score)
    melody_info = analyze_score(melody_score)
    return {
        "path": str(file_path).replace("\\", "/"),
        "name": file_path.name,
        "info": info,
        "melody_info": melody_info,
        "dropped_notes": dropped_notes,
    }

@app.post("/api/upload")
async def upload_file(request: Request, filename: str):
    """Upload and validate a new MIDI file, saving it into data/uploads/."""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename parameter is required")

    safe_name = Path(filename).name
    if not (safe_name.lower().endswith(".mid") or safe_name.lower().endswith(".midi")):
        raise HTTPException(status_code=400, detail="Only .mid and .midi files are supported")

    body = await request.body()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / safe_name
    target_path.write_bytes(body)

    try:
        score = load_score(target_path)
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid MIDI file: {e}")

    info = analyze_score(score)
    melody_score, dropped_notes = extract_melody(score)
    melody_info = analyze_score(melody_score)

    return {
        "status": "ok",
        "file": {
            "folder": "uploads",
            "name": safe_name,
            "path": str(target_path).replace("\\", "/"),
        },
        "info": info,
        "melody_info": melody_info,
        "dropped_notes": dropped_notes,
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    current_task = None
    playback_speed = 1.0
    
    async def play_piece(file_path: str, melody_only: bool = False):
        nonlocal playback_speed
        try:
            score = load_score(Path(file_path))
            dropped_notes = 0
            if melody_only:
                score, dropped_notes = extract_melody(score)

            # Standard eval rewards
            standard_rewards = RewardConfig(
                hit_exact=1.0,
                hit_off_by_one=0.5,
                wrong_press=-0.5,
                miss=-1.0,
            )
            env = PianoFreeKeysEnv(scores=[score], seed=42, reward_config=standard_rewards)
            obs, info = env.reset(options={"piece_index": 0})
            
            # Send init config
            notes_data = [
                {
                    "start_beat": n.start_beat,
                    "duration_beats": n.duration_beats,
                    "pitch": n.pitch,
                    "velocity": 0.8
                }
                for n in score.notes
            ]
            await websocket.send_json({
                "type": "init",
                "tempo": env.current_score.tempo_bpm,
                "notes": notes_data,
                "total_beats": score.total_beats,
                "melody_only": melody_only,
                "dropped_notes": dropped_notes,
            })
            
            terminated = False
            truncated = False
            
            prev_hits_exact = 0
            prev_hits_off = 0
            prev_wrong = 0
            
            while not (terminated or truncated):
                step_now = env.current_step
                action = PLAYER.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                pressed_pitch = None
                result = None
                
                if action > 0:
                    pressed_pitch = 20 + action
                    if info["hits_exact"] > prev_hits_exact:
                        result = "exact"
                    elif info["hits_off_by_one"] > prev_hits_off:
                        result = "off_by_one"
                    else:
                        result = "wrong"
                
                # Send step update
                await websocket.send_json({
                    "type": "step",
                    "step": step_now,
                    "beat": step_now / env.steps_per_beat,
                    "action": int(action),
                    "pitch": int(pressed_pitch) if pressed_pitch else None,
                    "result": result,
                    "metrics": {
                        "hits": int(info["hits_exact"] + info["hits_off_by_one"]),
                        "wrong_presses": int(info["wrong_presses"]),
                        "missed_notes": int(info["missed_notes"]),
                        "total_notes": int(info["total_notes"])
                    }
                })
                
                prev_hits_exact = info["hits_exact"]
                prev_hits_off = info["hits_off_by_one"]
                prev_wrong = info["wrong_presses"]
                
                # Calculate sleep time
                base_sleep = 60.0 / env.current_score.tempo_bpm / 4.0
                actual_sleep = base_sleep / playback_speed
                await asyncio.sleep(actual_sleep)
                
            await websocket.send_json({"type": "done"})
            
        except asyncio.CancelledError:
            # Task was cancelled (user pressed stop or played a new piece)
            pass
        except Exception as e:
            print(f"Error during playback: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})

    try:
        while True:
            data = await websocket.receive_json()
            if data["action"] == "play":
                if current_task and not current_task.done():
                    current_task.cancel()
                melody_only = bool(data.get("melody_only", False))
                current_task = asyncio.create_task(play_piece(data["file"], melody_only=melody_only))
            elif data["action"] == "stop":
                if current_task and not current_task.done():
                    current_task.cancel()
            elif data["action"] == "speed":
                playback_speed = float(data["value"])
    except WebSocketDisconnect:
        if current_task and not current_task.done():
            current_task.cancel()

# Mount the static files at the end so API routes work
app.mount("/", StaticFiles(directory="web", html=True), name="web")

def main():
    parser = argparse.ArgumentParser(description="Run the Piano-RL web player.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/all_pitch_conv/final.zip",
        help="Path to trained PPO model zip file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    args = parser.parse_args()

    global PLAYER
    print(f"Loading model from {args.model_path}...")
    try:
        PLAYER = PPOPlayer(Path(args.model_path))
    except FileNotFoundError:
        print(f"Warning: Model not found at {args.model_path}. Will crash if play is attempted.")
    
    # Ensure web directory exists
    Path("web").mkdir(exist_ok=True)
    
    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
