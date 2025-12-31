# CoMLRL Training Progress Bar and Logging Enhancement

## Summary

This document describes the progress bar and logging enhancements made to the CoMLRL training framework.

## Features Implemented

### 1. Logging System (`comlrl/utils/logging_utils.py`)
- **TqdmLoggingHandler**: Custom logging handler that writes to `tqdm.write()` to prevent log messages from disrupting progress bars
- **setup_logger()**: Function to create a logger with configurable log level
- **get_logger()**: Function to retrieve or create logger instances

### 2. Command-Line Log Level Control
Modified `basic-train.py` to accept a `--log-level` argument:
```bash
python basic-train.py --log-level DEBUG
python basic-train.py --log-level INFO   # default
python basic-train.py --log-level WARNING
python basic-train.py --log-level ERROR
python basic-train.py --log-level CRITICAL
```

### 3. Progress Bars
All trainers now display a progress bar that:
- Shows current epoch and batch progress
- Remains at the bottom of the console
- Displays key metrics (reward_mean, policy_loss, etc.) in real-time
- Updates smoothly without interfering with log messages

### 4. Enhanced Logging

#### Log Levels and Usage

**DEBUG Level** - Detailed diagnostic information:
- Model loading progress (agent by agent)
- Buffer processing details
- Rollout collection details
- Training step parameters

**INFO Level** - General progress information:
- Training initialization
- Epoch start/completion
- Evaluation runs
- Final training completion
- Configuration summary

**WARNING Level** - Important warnings:
- CUDA availability warnings
- Configuration issues

**ERROR/CRITICAL Level** - Error conditions only

#### Logging Added to All Trainers

##### IAC Trainer (`comlrl/trainers/iac.py`)
- Initialization logging with device and agent count
- Model loading progress
- Epoch-level progress information
- Rollout collection and buffer processing
- Training completion

##### MAAC Trainer (`comlrl/trainers/maac.py`)
- Similar logging structure as IAC
- Multi-turn specific logging
- Evaluation run logging
- Critic model loading

##### MAGRPO Trainer (`comlrl/trainers/magrpo.py`)
- Comprehensive initialization logging
- Training configuration display
- Homogeneous agent loading
- Batch processing details
- Training step details with turn/generation info

## File Changes

### New Files
1. `comlrl/utils/logging_utils.py` - Logging utility module
2. Updated `train.sh` - Example training script with log level options

### Modified Files
1. `basic-train.py` - Added argument parsing and logging
2. `comlrl/utils/__init__.py` - Export logging functions
3. `comlrl/trainers/iac.py` - Added logging and progress bar
4. `comlrl/trainers/maac.py` - Added logging and progress bar
5. `comlrl/trainers/magrpo.py` - Added logging and progress bar

## Usage Examples

### Basic Usage (INFO level)
```bash
python basic-train.py
```

Output:
```
[2025-12-31 10:00:00] [INFO] [comlrl] Starting training with log level: INFO
[2025-12-31 10:00:01] [INFO] [comlrl] Loading model and tokenizer from: ...
[2025-12-31 10:00:05] [INFO] [comlrl] Loading dataset: trl-lib/tldr
[2025-12-31 10:00:06] [INFO] [comlrl] Dataset loaded with 128 samples
[2025-12-31 10:00:06] [INFO] [comlrl.magrpo] Initializing MAGRPO Trainer
[2025-12-31 10:00:10] [INFO] [comlrl.magrpo] Starting MAGRPO training
Epoch 1/8: 100%|████████████████| 128/128 [10:23<00:00, reward_mean=0.8534]
[2025-12-31 10:10:33] [INFO] [comlrl.magrpo] Epoch 1/8 completed. Metrics: {...}
```

### Verbose Debug Mode
```bash
python basic-train.py --log-level DEBUG
```

Additional output includes:
- Per-agent model loading
- Detailed buffer operations
- Rollout collection details
- Training step parameters

### Minimal Output
```bash
python basic-train.py --log-level WARNING
```

Only warnings, errors, and the progress bar are shown.

## Progress Bar Behavior

The progress bar:
1. **Always displays at the bottom** of the console
2. **Updates in real-time** with current metrics
3. **Shows epoch progress** (e.g., "Epoch 1/8")
4. **Displays batch count** and processing rate
5. **Shows key metrics** in postfix (reward_mean, expected_return, policy_loss)
6. **Never gets disrupted** by log messages (they appear above it)

## Log Format

Standard format: `[timestamp] [LEVEL] [logger_name] message`

Example:
```
[2025-12-31 10:15:23] [INFO] [comlrl.magrpo] Starting epoch 2/8
[2025-12-31 10:15:24] [DEBUG] [comlrl.magrpo] Processing batch 1
[2025-12-31 10:15:25] [DEBUG] [comlrl.magrpo] Starting training step with returns-based updates
```

## Benefits

1. **Clear Progress Indication**: Users always know what's happening and how long it will take
2. **Configurable Verbosity**: Choose the right log level for your needs
3. **Clean Console Output**: Progress bar stays at bottom, logs don't interfere
4. **Better Debugging**: DEBUG level provides detailed diagnostic information
5. **Production Ready**: WARNING/ERROR levels for production runs
6. **No Code Changes Needed**: Existing code works without modification

## Technical Implementation

### TqdmLoggingHandler
The custom logging handler uses `tqdm.write()` instead of `print()` or standard output, ensuring log messages are written above the progress bar without disrupting it.

### Logger Hierarchy
- Root logger: `comlrl`
- Trainer loggers: `comlrl.iac`, `comlrl.maac`, `comlrl.magrpo`
- All loggers use the same handler and format

### Progress Bar Integration
- Uses `tqdm` context manager
- Position fixed at bottom (`position=0`)
- Leaves progress bar visible after completion (`leave=True`)
- Automatically calculates total batches when available

## Compatibility

- Works with all existing trainer classes
- Inherited trainers (MAREINFORCE, MAREMAX, MARLOO) automatically get logging
- No breaking changes to existing code
- Compatible with wandb logging

## Future Enhancements

Potential improvements:
1. File logging support (log to file in addition to console)
2. Separate log levels for different modules
3. Colored log output for better visibility
4. Progress bar for multi-GPU training
5. Training time estimation

