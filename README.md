# SnortTracker

A playful snort counter that behaves like a step counter: continuously listens, detects **full snorts**, and increments a local count when a snort is confidently confirmed.

## Architecture

```
capture → cheap prefilter → classifier → event state machine → append-only log
```

The runtime is a single pipeline. Everything else — dataset tooling, training, export — supports it.

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `runtime/` | Live detection pipeline: capture, gating, inference, state machine, logging, config |
| `dataset/` | Dataset collection, slicing, labeling, preprocessing, augmentation |
| `train/` | Model definition, training, evaluation, export |
| `cli/` | User-facing command-line interface |
| `models/` | Versioned exported model artifacts |
| `logs/` | Local append-only event logs |
| `tests/` | Unit, integration, and smoke tests |
| `Data/Pages/` | Design documents, implementation plans, and task tracking |

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd SnortTracker

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies (once requirements.txt is added)
# pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/
```

## CLI Usage

```bash
python -m cli.main view       # Show current count
python -m cli.main reset      # Clear count and logs
python -m cli.main tail       # Show recent events
python -m cli.main status     # Show runtime health
python -m cli.main calibrate  # Run calibration mode
```

## Design Principles

- **Keep the runtime small.** One pipeline, not a sprawling graph.
- **Treat detection as a state machine.** Confirmation is deterministic and testable.
- **Treat labels as the real bottleneck.** Label quality matters more than model size.
- **Use one shared preprocessing contract.** Training and inference must match.
- **Prefer portable inference.** Quantized model, stable runtime.
- **Measure what matters.** Event-level performance, not window-level accuracy.
- **Add calibration and health checks.** Thresholds must adapt to the local environment.

## Target Hardware

Raspberry Pi Zero-class device with a USB microphone.

## License

See [LICENSE](LICENSE).
