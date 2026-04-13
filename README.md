# KerbalRoverOperator

An LLM-powered autonomous rover controller for **Kerbal Space Program (KSP)**. The agent connects to KSP via [kRPC](https://krpc.github.io/krpc/), reads live telemetry, and issues drive/brake commands to navigate a rover to a GPS target using a [LangGraph](https://github.com/langchain-ai/langgraph) agent loop backed by OpenAI GPT-4.1-mini.

<video src="[Demo.mp4](https://github.com/user-attachments/assets/87c1d095-3635-4fca-840a-c900d0338e39)" controls width="100%"></video>

---

## Overview

The agent operates in a closed-loop cycle:

```
observe → LLM → safety check → execute tool → update plot → observe …
```

At each step the LLM receives a structured observation (distance, forward/right offsets, speed, bearing error) and responds with exactly one tool call — either `drive(throttle, steer, duration_s)` or `brake(brake_strength, duration_s)`. A safety layer intercepts the command before execution and overrides it when the rover's speed exceeds the configured limit. Each episode ends when the rover arrives within **3 m** of the target or exceeds **100 steps**.

---

## Project Structure

```
KerbalRoverOperator/
├── agent/
│   ├── agent.py                  # LangGraph agent: graph definition and main loop
│   ├── config.py                 # Hyperparameters, model name, system prompt
│   ├── executor.py               # Wraps kRPC wheel controls (drive / brake)
│   ├── krpc_client.py            # kRPC connection context manager
│   ├── state.py                  # StateTracker: computes telemetry from kRPC
│   ├── logger.py                 # Per-step CSV logger
│   ├── plotter.py                # Real-time matplotlib path visualizer
│   └── experiment_locations.csv  # 40 predefined target coordinates (lat, lon)
├── analyze.py                    # Post-hoc metrics computation and LaTeX export
├── requirements.txt
├── Experimento 1/ … Experimento 5/   # Raw episode CSVs (40 per experiment)
│   └── results_output/
│       ├── episode_metrics_summary.csv
│       ├── final_summary_table.csv
│       └── final_summary_table.tex
└── README.md
```

---

## Requirements

- **Kerbal Space Program** with the [kRPC mod](https://krpc.github.io/krpc/) installed and running.
- Python 3.10+
- An OpenAI API key.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `krpc`, `langgraph`, `langchain-openai`, `openai`, `matplotlib`, `pandas`, `scikit-learn`.

---

## Configuration

All tuneable parameters live in `agent/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL` | `gpt-4.1-mini` | OpenAI model used by the LLM node |
| `MIN_STEP_S` | `1` | Minimum action duration (seconds) |
| `MAX_STEP_S` | `5` | Maximum action duration (seconds) |
| `MAX_SAFE_SPEED_MPS` | `3.0` | Speed threshold for safety override |
| `ARRIVE_M` | `3` | Arrival radius (metres) |
| `MAX_DISTANCE_M` | `300` | Maximum random target offset when generating targets |
| `MAX_HISTORY` | `12` | Number of recent messages kept in context |

The `OPENAI_API_KEY` environment variable must be set before running:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Linux / macOS
export OPENAI_API_KEY="sk-..."
```

---

## Running the Agent

1. Start KSP and enable the kRPC server (default address `127.0.0.1:50000`).
2. Load a save with a rover on the surface.
3. From the `agent/` directory, run:

```bash
cd agent
python agent.py
```

The agent will run **40 sequential episodes**. Each episode:
- Loads the `quicksave` in KSP to reset the rover position.
- Reads the target coordinates from `experiment_locations.csv` (line = episode number).
- Streams the LangGraph agent loop until arrival or 100-step timeout.
- Saves a trajectory plot (`<episode>.png`) and a telemetry log (`<episode>.csv`).

To use **random targets** instead of the predefined CSV, edit `agent.py` and swap the commented lines in `main()`:

```python
# Predefined target (default)
target_latlon = read_target_from_csv(line_number=step, csv_path="experiment_locations.csv")

# Random target within MAX_DISTANCE_M
# target_latlon = tracker.generate_random_target_latlon(vessel, space_center, MAX_DISTANCE_M)
```

---

## Agent Architecture

### State Graph

```
[observe] ──► [llm] ──► [safety] ──► [tools] ──► [update_plot]
   ▲                       │                            │
   └───────────────────────┴────────────────────────────┘
                     (loop until arrival or step limit)
```

| Node | Responsibility |
|---|---|
| `observe` | Calls `StateTracker.get_state()` to pull live kRPC telemetry |
| `llm` | Builds the observation string, invokes the LLM, appends the response |
| `safety` | Overrides `drive` commands with `brake` when `speed_mps > MAX_SAFE_SPEED_MPS` |
| `tools` | Executes the first tool call via `Executor`; logs the step |
| `update_plot` | Updates the real-time matplotlib trajectory chart |

### Observation Format

Each LLM turn receives:

```
OBSERVATION:
distance_meters=<float>        # straight-line distance to target
forward_distance_meters=<float> # signed component along rover's heading
right_distance_meters=<float>  # signed component perpendicular to heading
speed_mps=<float>              # horizontal ground speed
bearing_error_deg=<float>      # signed angle to target (+ = right, − = left)
```

### Available Tools

| Tool | Signature | Description |
|---|---|---|
| `drive` | `drive(throttle, steer, duration_s)` | Apply throttle and steering for `duration_s` seconds |
| `brake` | `brake(brake_strength, duration_s)` | Apply brakes at `brake_strength` (0–100) for `duration_s` seconds |

---

## Log Format

Each episode produces a CSV with the following columns:

| Column | Description |
|---|---|
| `step` | Decision step index |
| `timestamp` | Unix timestamp |
| `distance_m` | Distance to target (m) |
| `forward_m` | Forward component (m) |
| `right_m` | Lateral component (m) |
| `speed_mps` | Ground speed (m/s) |
| `bearing_deg` | Bearing error (°) |
| `tool` | Tool called (`drive` or `brake`) |
| `throttle` | Throttle value applied |
| `steer` | Steering value applied |
| `brake_strength` | Brake strength applied |
| `duration_s` | Action duration (s) |
| `overridden_by_safety` | Whether the safety node overrode the LLM's command |

---

## Analysis

`analyze.py` processes all episode CSVs in a chosen experiment folder and computes per-episode and aggregate metrics.

Configure the target folder at the top of the file:

```python
data_dir = os.path.join(current_dir, "Experimento 5")
```

Then run:

```bash
python analyze.py
```

Outputs written to `<experiment>/results_output/`:

- `episode_metrics_summary.csv` — one row per episode with metrics such as success, steps, duration, speed statistics, safety overrides, steering oscillation, and overshoot count.
- `final_summary_table.csv` — aggregated summary statistics.
- `final_summary_table.tex` — LaTeX-formatted table for direct inclusion in papers.

**Success criterion:** an episode is considered successful if it completes in fewer than 100 steps (i.e., the rover reached the 3 m arrival radius before the step limit).
