"""FastAPI web server for live piano playing (supports both single-key and multi-key models)."""

import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pianorl.env import PianoFreeKeysEnv, MultiKeyPianoEnv, RewardConfig
from pianorl.eval import PPOPlayer, PerfectMultiPlayer
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
DATA_DIRS = [
    Path("data/real"),
    Path("data/heldout"),
    Path("data/multikey_heldout"),
    Path("data/uploads"),
]


def is_player_multikey(player) -> bool:
    """Detect whether a player outputs MultiBinary(88) actions instead of Discrete(89)."""
    if player is None:
        return False
    if hasattr(player, "is_multikey"):
        return bool(player.is_multikey)
    if hasattr(player, "model") and hasattr(player.model, "action_space"):
        from gymnasium.spaces import MultiBinary
        return isinstance(player.model.action_space, MultiBinary) or (
            hasattr(player.model.action_space, "shape")
            and player.model.action_space.shape == (88,)
        )
    if isinstance(player, PerfectMultiPlayer):
        return True
    if "Multi" in player.__class__.__name__:
        return True
    return False


@app.get("/api/files")
def get_files():
    """List available MIDI files across real, heldout, multikey_heldout, and uploads."""
    files = []
    for d in DATA_DIRS:
        if d.exists():
            for f in sorted(d.glob("*.mid*")):
                files.append(
                    {"folder": d.name, "name": f.name, "path": str(f).replace("\\", "/")}
                )
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

            is_multikey = is_player_multikey(PLAYER)

            # Standard eval rewards
            standard_rewards = RewardConfig(
                hit_exact=1.0,
                hit_off_by_one=0.5,
                wrong_press=-0.5,
                miss=-1.0,
            )

            if is_multikey:
                env = MultiKeyPianoEnv(
                    scores=[score], seed=42, reward_config=standard_rewards
                )
            else:
                env = PianoFreeKeysEnv(
                    scores=[score], seed=42, reward_config=standard_rewards
                )

            obs, info = env.reset(options={"piece_index": 0})

            # Send init config
            notes_data = [
                {
                    "start_beat": n.start_beat,
                    "duration_beats": n.duration_beats,
                    "pitch": n.pitch,
                    "velocity": round(getattr(n, "velocity", 80) / 127.0, 3),
                }
                for n in score.notes
            ]
            await websocket.send_json(
                {
                    "type": "init",
                    "tempo": env.current_score.tempo_bpm,
                    "notes": notes_data,
                    "total_beats": score.total_beats,
                    "melody_only": melody_only,
                    "dropped_notes": dropped_notes,
                    "model_type": "multikey" if is_multikey else "single_key",
                    "has_pedal": len(score.pedal_intervals) > 0,
                }
            )

            terminated = False
            truncated = False

            prev_hits_exact = 0
            prev_hits_off = 0
            prev_wrong = 0

            while not (terminated or truncated):
                step_now = env.current_step
                current_beat = step_now / env.steps_per_beat
                raw_action = PLAYER.act(obs)

                # Process action depending on multi-key vs single-key
                if is_multikey:
                    act_arr = np.asarray(raw_action)
                    pressed_pitches = [21 + i for i in range(88) if act_arr[i] > 0]
                    env_action = act_arr
                else:
                    act_int = int(raw_action)
                    pressed_pitches = [20 + act_int] if act_int > 0 else []
                    env_action = act_int

                obs, reward, terminated, truncated, info = env.step(env_action)

                # Determine hit / off_by_one / wrong results
                new_exact = info["hits_exact"] - prev_hits_exact
                new_off = info["hits_off_by_one"] - prev_hits_off
                new_wrong = info["wrong_presses"] - prev_wrong

                overall_result = None
                if new_exact > 0:
                    overall_result = "exact"
                elif new_off > 0:
                    overall_result = "off_by_one"
                elif new_wrong > 0:
                    overall_result = "wrong"

                # Check sustain pedal state at current beat
                is_pedal_down = any(
                    p_start <= current_beat < p_end
                    for p_start, p_end in score.pedal_intervals
                )
                active_pedal_end = max(
                    (p_end for p_start, p_end in score.pedal_intervals if p_start <= current_beat < p_end),
                    default=current_beat,
                )
                pedal_sustain_beats = max(0.0, active_pedal_end - current_beat) if is_pedal_down else 0.0

                # Calculate per-pitch result, duration, and velocity
                pitch_results = {}
                step_notes = []
                tempo_bpm = env.current_score.tempo_bpm

                if pressed_pitches:
                    for p in pressed_pitches:
                        matched_target = None
                        matched_exact = False
                        matched_off = False

                        for t in env.targets:
                            if t["pitch"] == p:
                                if t["start_step"] == step_now:
                                    matched_exact = True
                                    matched_target = t
                                    break
                                elif abs(t["start_step"] - step_now) == 1 and matched_target is None:
                                    matched_off = True
                                    matched_target = t

                        if matched_exact:
                            pitch_results[str(p)] = "exact"
                        elif matched_off:
                            pitch_results[str(p)] = "off_by_one"
                        else:
                            pitch_results[str(p)] = "wrong"

                        # Retrieve authentic note duration and velocity from target if matched
                        if matched_target is not None:
                            dur_beats = matched_target.get("duration_beats", 1.0)
                            vel_raw = matched_target.get("velocity", 80)
                            vel_norm = round(max(0.1, min(1.0, vel_raw / 127.0)), 3)
                        else:
                            dur_beats = 0.5
                            vel_norm = 0.65  # Moderate default for speculative/wrong model strike

                        # If sustain pedal is pressed, let note ring until pedal release
                        eff_dur_beats = max(dur_beats, pedal_sustain_beats)
                        dur_seconds = (eff_dur_beats * 60.0) / tempo_bpm
                        dur_seconds_scaled = round(max(0.08, min(6.0, dur_seconds / playback_speed)), 3)

                        step_notes.append(
                            {
                                "pitch": p,
                                "duration": dur_seconds_scaled,
                                "velocity": vel_norm,
                            }
                        )

                # Send step update
                await websocket.send_json(
                    {
                        "type": "step",
                        "step": step_now,
                        "beat": current_beat,
                        "action": (
                            [int(x) for x in act_arr] if is_multikey else int(env_action)
                        ),
                        "pitches": pressed_pitches,
                        "pitch": pressed_pitches[0] if pressed_pitches else None,
                        "notes": step_notes,
                        "pedal": is_pedal_down,
                        "results": pitch_results,
                        "result": overall_result,
                        "is_multikey": is_multikey,
                        "metrics": {
                            "hits": int(info["hits_exact"] + info["hits_off_by_one"]),
                            "wrong_presses": int(info["wrong_presses"]),
                            "missed_notes": int(info["missed_notes"]),
                            "total_notes": int(info["total_notes"]),
                        },
                    }
                )

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
                current_task = asyncio.create_task(
                    play_piece(data["file"], melody_only=melody_only)
                )
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
        help="Path to trained PPO model zip file",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host IP to bind to"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    global PLAYER
    print(f"Loading model from {args.model_path}...")
    try:
        PLAYER = PPOPlayer(Path(args.model_path))
        arch_name = (
            "Multi-Key MultiBinary(88)"
            if PLAYER.is_multikey
            else "Single-Key Discrete(89)"
        )
        print(f"Model loaded successfully! Detected architecture: {arch_name}")
    except FileNotFoundError:
        print(
            f"Warning: Model not found at {args.model_path}. Will crash if play is attempted."
        )

    # Ensure web directory exists
    Path("web").mkdir(exist_ok=True)

    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
