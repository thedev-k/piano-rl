# Piano-RL: Sheet-Reading Piano Agent

Piano-RL is an artificial intelligence project designed to train a virtual piano player using reinforcement learning. Instead of memorizing specific songs, the agent learns the core skill of reading musical scores beat-by-beat using a short lookahead window. It controls virtual hands and fingers to press the correct keys with accurate timing on an 88-key piano. The ultimate goal is for the trained agent to play new, unseen pieces of music from scratch on a standard computer processor (CPU).

## Dataset Generation

To procedurally generate the training and held-out dataset (800 pieces across 4 difficulty levels with an 85/15 split), run:
```bash
python scripts/generate_dataset.py
```

## How to Train

1. **Activate your virtual environment (Windows PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start a training run (300,000 timesteps on CPU):**
   - **Shared Pitch-Convolution brain (`pitch_conv` - Recommended):**
     Learns one shared note-reading rule across all 88 keys using 1D convolution (~49k parameters):
     ```powershell
     python scripts/train.py --policy pitch_conv --levels 1 --timesteps 300000 --run-name level1_pitch_conv
     ```
   - **Standard Multilayer Perceptron brain (`mlp` - Baseline):**
     Treats all piano keys independently with separate weights (~750k parameters):
     ```powershell
     python scripts/train.py --policy mlp --levels 1 --timesteps 300000 --run-name level1_mlp
     ```

3. **Open TensorBoard to view live learning curves:**
   ```powershell
   tensorboard --logdir runs
   ```
   *(Then open http://localhost:6006 in your web browser)*

4. **Evaluate the finished model on the held-out pieces:**
   ```powershell
   python scripts/evaluate.py --player ppo --model-path checkpoints/level1_run/final.zip --split heldout
   ```

## Diagnosing Mistakes (`scripts/diagnose.py`)

To understand *why* an agent made mistakes and where its wrong key presses came from, run the diagnostic tool:

```powershell
python scripts/diagnose.py --model-path checkpoints/all_levels_run/final.zip --split heldout
```

Every wrong key press is categorized into exactly one of three simple groups:
- **Repeat**: The model struck the right pitch, but did it again right before or right after the note started (a double-strike or echo).
- **Wrong Key Near Note**: A note was starting nearby in time, but the model struck the wrong piano key.
- **No Note Nearby**: The model pressed a key when there was no note playing or coming up at all (unprovoked ghost presses during rests).

The tool also reports **Presses/Note** (how many keys the model pressed per note, where 1.0 is ideal), **Recall**, and **Precision**.

## Customizing Training Rewards

You can adjust the reward and penalty numbers from the command line in `scripts/train.py`:

```powershell
python scripts/train.py --levels 1 --timesteps 300000 --run-name custom_rewards_run `
    --exact-reward 1.0 `
    --off-by-one-reward 0.5 `
    --wrong-press-penalty -1.0 `
    --miss-penalty -1.0
```

Available reward flags (and their default values):
- `--exact-reward` (default `1.0`): Points awarded for striking a note at the exact start step.
- `--off-by-one-reward` (default `0.5`): Points awarded for striking a note 1 step early or late.
- `--wrong-press-penalty` (default `-0.5`): Penalty for pressing a key when no note matches.
- `--miss-penalty` (default `-1.0`): Penalty for allowing a note to pass completely unplayed.

The chosen configuration is printed at the start of training and saved to `checkpoints/<run-name>/reward_config.json`.

> **Note on Evaluation:** `scripts/evaluate.py` always evaluates models using the standard default reward numbers (`+1.0`, `+0.5`, `-0.5`, `-1.0`). This guarantees that Mean Reward scores remain directly and fairly comparable across runs trained with different penalty settings.


