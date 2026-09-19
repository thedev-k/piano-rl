# Project Rules & Ground Guidelines

Follow these rules strictly across all sessions:

1. **Read PLAN.md before every task.** It is the source of truth. If the user asks for something that conflicts with it, explain the conflict instead of silently changing the plan.
2. **Beginner-friendly explanations.** The user is a complete beginner. Explain what was done in plain, simple English after every task, avoiding technical jargon or explaining any terms used.
3. **One step at a time.** Work on ONE step at a time. Never start the next step until the user explicitly confirms and approves it.
4. **Hardware and engine limits.** CPU only. Do NOT write or use GPU code. Do NOT use heavy physics engines like MuJoCo.
5. **Music representation.** Time in music scores must always be stored in beats, not in seconds. Seconds only apply when tempo is factored into the simulation.
6. **Sliding lookahead window.** The learning agent must only ever see the upcoming few beats of the score (a sliding window), never the entire piece.
7. **Test-driven quality.** Every piece of code must have tests. Run the tests and display the results before declaring any task complete.
8. **File protection.** Never delete any files without asking the user first and receiving explicit confirmation.
9. **Git and GitHub policy.** Never connect to GitHub, add a git remote, or push anything. The user handles all remote and GitHub tasks. Only make local git commits.
