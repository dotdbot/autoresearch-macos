# Tutorial: Running autoresearch on your Mac

This guide takes you from a fresh clone to an AI agent running experiments on its own overnight, and then shows how to read the results in the morning.

**What autoresearch is:** you give a coding agent (Claude Code, Codex, etc.) a small but real LLM training setup. The agent edits `train.py`, trains for 5 minutes, and checks whether validation loss improved. It keeps the change if it did, reverts it if not, and repeats. You don't write the Python yourself. You write the *instructions* the agent follows, which live in `program.md`.

---

## 0. Prerequisites

| Requirement | Notes |
|---|---|
| Apple Silicon Mac (M1–M4) | `prepare.py` and `train.py` both call `verify_macos_env()` and stop with an error if they aren't on macOS with MPS available. |
| Python 3.10+ | Pinned in `.python-version`. |
| [uv](https://docs.astral.sh/uv/) | Package manager used for every command below. |
| A coding agent | Anything that can run shell commands and edit files, such as Claude Code. |
| ~A few GB of disk | Data shards and tokenizer go in `~/.cache/autoresearch/`. |

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1. Install and prepare data (one time)

```bash
git clone <this-repo> autoresearch-macos
cd autoresearch-macos

uv sync              # install dependencies into .venv
uv run prepare.py    # download data shards + train BPE tokenizer (~2 min)
```

`prepare.py` downloads 10 training shards by default, plus the pinned validation shard. You can change how many it downloads:

```bash
uv run prepare.py --num-shards 20          # more training data
uv run prepare.py --download-workers 4     # gentler on your network
```

Check that it worked:

```bash
ls ~/.cache/autoresearch/data ~/.cache/autoresearch/tokenizer
```

---

## 2. Run one baseline by hand

Before handing anything to an agent, make sure a single training run finishes on your machine:

```bash
uv run train.py > run.log 2>&1
grep "^val_bpb:" run.log
```

A run takes about 5 minutes of training, plus startup and the final evaluation. When it finishes, it prints a summary:

```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    331.4
peak_vram_mb:     0.0
mfu_percent:      ...
total_tokens_M:   ...
num_steps:        ...
num_params_M:     ...
depth:            4
```

How to read it:

- **`val_bpb`** (validation bits per byte) is the only metric that matters. **Lower is better.** Because it is measured per byte rather than per token, changes to the vocabulary or architecture can still be compared fairly.
- **`peak_vram_mb` is always `0.0` on MPS.** The script only measures memory on CUDA. On a Mac, use Activity Monitor (or `memory_pressure`) to watch memory use instead.
- **`mfu_percent`** is measured against an H100's peak FLOPs, so on a Mac it will be very small. You can ignore it.
- The 5-minute budget counts training wall-clock time only. The first 10 steps (warmup/compile) are not counted.

If this run crashes with an out-of-memory error, lower `DEVICE_BATCH_SIZE` near the top of the hyperparameter block in `train.py` and try again.

---

## 3. Understand the three files that matter

| File | Who edits it | What it contains |
|---|---|---|
| `prepare.py` | **Nobody.** It is read-only. | Fixed constants (`TIME_BUDGET = 300`, `MAX_SEQ_LEN = 2048`, `VOCAB_SIZE = 8192`), the data loader, the tokenizer, and `evaluate_bpb`, which is the ground-truth metric. |
| `train.py` | **The agent** | The GPT model, the optimizer (Muon + AdamW), the training loop, and every hyperparameter. Everything in it is fair game. |
| `program.md` | **You** | The agent's instructions: how to set up a run, the experiment loop, how to log results, and when to keep or discard a change. |

These rules make experiments comparable. The time budget is fixed, the evaluation is fixed, and the agent changes only one file. A lower `val_bpb` therefore means a better model *for your Mac in 5 minutes*.

The knobs the agent usually touches first are in `train.py`:

```python
DEPTH = 4               # number of transformer layers
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
DEVICE_BATCH_SIZE = 16  # per-device batch size (reduce if OOM)
TOTAL_BATCH_SIZE = 2**16
MATRIX_LR = 0.04        # Muon LR
EMBEDDING_LR = 0.6
WARMDOWN_RATIO = 0.5
WINDOW_PATTERN = "L"    # L=full attention, S=half-context sliding window
...
```

---

## 4. Start the agent

Open your coding agent in the repo root. Give it permission to run commands without asking each time; the whole point is that it runs unattended. Then prompt:

```
Have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent goes through the setup steps in `program.md`:

1. Proposes a **run tag** based on today's date (e.g. `oct7`).
2. Creates a branch `autoresearch/<tag>` from `master`.
3. Reads `README.md`, `prepare.py`, and `train.py`.
4. Checks that `~/.cache/autoresearch/` contains data and a tokenizer.
5. Creates `results.tsv` with just the header row.
6. Asks you to confirm.

Say yes. From that point on, the agent is told **never to stop and ask** whether it should continue. It keeps looping until you interrupt it.

> Tip: plug your Mac in and stop it from sleeping during the run, e.g. `caffeinate -dims` in a separate terminal.

---

## 5. What the agent does in its loop

Each iteration (~5–6 minutes) looks like this:

```
┌─► edit train.py with one idea
│   git commit
│   uv run train.py > run.log 2>&1
│   grep "^val_bpb:\|^peak_vram_mb:" run.log
│     ├─ empty?  → crashed: tail run.log, fix if trivial, else log "crash"
│     ├─ better? → log "keep", branch advances
│     └─ worse?  → log "discard", git reset to previous commit
└── repeat
```

That works out to about **12 experiments per hour, ~100 overnight**.

Every result goes into `results.tsv`. It is tab-separated and stays untracked, so it survives the `git reset`s:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	0.0	keep	baseline
b2c3d4e	1.229800	0.0	keep	increase MATRIX_LR to 0.05
c3d4e5f	1.240100	0.0	discard	switch to GeLU
d4e5f6a	0.000000	0.0	crash	depth 12 (OOM)
```

(`memory_gb` will be `0.0` on a Mac, for the reason given in step 2.)

The agent also follows a **simplicity rule** from `program.md`. A tiny gain that adds ugly code isn't worth keeping. Deleting code and getting equal or better results is a win.

---

## 6. Check on progress

You can check in at any time without disturbing the agent:

```bash
git log --oneline autoresearch/<tag>         # kept improvements only
column -t -s $'\t' results.tsv | tail -20    # latest experiments
sort -t $'\t' -k2 -n results.tsv | grep keep | head -5   # best so far
```

To stop the run, interrupt the agent (Esc / Ctrl+C). The branch holds the best `train.py` found so far.

---

## 7. Analyse the results

Open `analysis.ipynb` (e.g. `uv run --with jupyter jupyter lab`) and run all cells. It loads `results.tsv` and shows:

- the number of experiments that were kept, discarded, or crashed, and the keep rate
- a plot of `val_bpb` over time with the running best, saved as `progress.png`
- the total improvement over the baseline
- a ranking of the kept changes by how much each one helped

> **Known issue:** the plotting cell uses a variable `best` that is never defined. Add `best = valid["val_bpb"].min()` before the line `margin = ...` or the cell will fail.

---

## 8. Iterate on `program.md`, your actual job

The runs produce a better `train.py`. The bigger lever is improving the *research process*, and that lives in `program.md`. After a night of results, ask yourself:

- **Did the agent waste runs?** For example, many tiny learning-rate tweaks, or repeated crashes from the same OOM. Add guidance such as "batch similar hyperparameter sweeps" or "keep params under N million".
- **Did it get stuck?** Give it a list of ideas to try, papers to draw from, or tell it to combine near-misses.
- **Are the results noisy?** If two identical runs differ by ~0.002, add a rule that a change must beat the best result by more than that to count as a "keep", or must be re-run to confirm.
- **Is the dataset limiting you?** At this small scale, Karpathy suggests [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) as a drop-in replacement for cleaner results. Swapping it in means editing `prepare.py`, so do it on purpose, outside of an agent run.

Commit your `program.md` changes to `master`, and start the next run on a new branch tag.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: This script requires macOS with Metal` | You're not on an Apple Silicon Mac, or your PyTorch build lacks MPS. Run `uv sync` again and check `python -c "import torch; print(torch.backends.mps.is_available())"`. |
| Agent says data is missing | Run `uv run prepare.py`. |
| OOM / Metal allocation errors | Lower `DEVICE_BATCH_SIZE` or `DEPTH` in `train.py`. |
| Run prints `FAIL` | Loss exceeded 100 (training diverged). The learning rate is usually too high. |
| Run takes > 10 min | `program.md` tells the agent to kill it and treat it as a failure. If it happens to your baseline, close other GPU-heavy apps. |
| Agent keeps asking "should I continue?" | Point it at the **NEVER STOP** section of `program.md`, or make that section more forceful. |
| Context window fills up | Make sure the agent redirects output to `run.log` and only `grep`s it. It should never stream the full training log. |
