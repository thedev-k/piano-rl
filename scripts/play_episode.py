import argparse
from pathlib import Path

from pianorl.env import PianoFreeKeysEnv
from pianorl.eval import PerfectPlayer, SilentPlayer, SpamPlayer
from pianorl.score import pitch_to_note_name


def run_episode(player_name: str, midi_path: str) -> None:
    path = Path(midi_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    # Select player
    player_name = player_name.lower()
    if player_name == "perfect":
        player = PerfectPlayer()
    elif player_name == "silent":
        player = SilentPlayer()
    elif player_name == "spam":
        player = SpamPlayer(seed=42)
    else:
        raise ValueError(f"Unknown player '{player_name}'. Choose from: perfect, silent, spam")

    env = PianoFreeKeysEnv(scores=path, seed=42)
    obs, info = env.reset(seed=42)

    print("=" * 70)
    print(f"Running Episode: Player = {player_name.upper()} | Piece = {path.name}")
    print(f"Total Notes = {info['total_notes']} | Tempo = {env.current_score.tempo_bpm:.0f} BPM")
    print("=" * 70)
    print(f"{'Step':<6} | {'Key Struck':<12} | {'Note (Pitch)':<14} | {'Reward':<8} | {'Running Total'}")
    print("-" * 70)

    running_reward = 0.0
    step_idx = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_num = env.current_step
        action = player.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        running_reward += reward

        # Print first 24 steps
        if step_num < 24:
            if action == 0:
                key_str = "-"
                note_str = "-"
            else:
                pitch = 20 + action
                key_str = f"Key {action}"
                note_str = f"{pitch_to_note_name(pitch)} ({pitch})"

            reward_str = f"{reward:+.1f}" if reward != 0.0 else " 0.0"
            print(f"{step_num:<6} | {key_str:<12} | {note_str:<14} | {reward_str:<8} | {running_reward:+.1f}")

        step_idx += 1

    print("-" * 70)
    print(f"... completed in {step_idx} steps.")
    print("=" * 70)
    print("FINAL EPISODE SUMMARY:")
    print(f"  Hits (Exact):      {info['hits_exact']}")
    print(f"  Hits (Off-by-1):   {info['hits_off_by_one']}")
    print(f"  Wrong Presses:     {info['wrong_presses']}")
    print(f"  Missed Notes:      {info['missed_notes']}")
    print(f"  Total Notes:       {info['total_notes']}")
    print(f"  Total Reward:      {running_reward:+.1f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run a dummy player on an episode of PianoFreeKeysEnv.")
    parser.add_argument(
        "--player",
        type=str,
        default="perfect",
        choices=["perfect", "silent", "spam"],
        help="Dummy player to test (perfect, silent, or spam)",
    )
    parser.add_argument(
        "--midi",
        type=str,
        default="data/train/level2_0001.mid",
        help="Path to the MIDI file to play",
    )
    args = parser.parse_args()
    run_episode(args.player, args.midi)


if __name__ == "__main__":
    main()
