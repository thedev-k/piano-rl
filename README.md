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

2. **Start the Level 1 training run (300,000 timesteps on CPU):**
   ```powershell
   python scripts/train.py --levels 1 --timesteps 300000 --run-name level1_run
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


