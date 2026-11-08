# Fixed issues

A code review of this repository found 10 problems. All 10 are fixed on the local branch `fix/review-issues`.

**How far this was tested:** the fixes were run on CPU with PyTorch 2.15 and a small synthetic dataset, using a very small autoencoder. They have not been run on a GPU, on the real ShapeNet data, with `torchrun`, or at full model size. See [What was not tested](#what-was-not-tested).

## Before you use this

- **Only load checkpoints you trust.** Issue 2 sets `weights_only=False`, which lets a checkpoint file run code when it is loaded. This includes pretrained models you download. Older PyTorch versions already behaved this way, but PyTorch 2.6 and later would otherwise block it.
- **PyTorch 1.13 or newer is now required**, because older versions don't accept the `weights_only` argument.
- **Eval numbers will change.** Per-category results from `eval.py` now cover every test shape, so they won't match numbers printed by the old code (issue 3).
- **Single-GPU training will differ from before**, because batches are now reshuffled every epoch (issue 4). `torchrun` training already reshuffled and is unchanged.
- **The default weight decay is now listed as 0.01, not 0.05.** See the decision under issue 5.

## Stopped the code running on current PyTorch

### 1. Import crash on startup
- **Problem:** `util/misc.py` imported `inf` from `torch._six`, which modern PyTorch has removed. Every entry point imports this file, so nothing could start.
- **Fix:** it now uses `from math import inf` (`util/misc.py:13`).

### 2. Checkpoints could not be loaded
- **Problem:** checkpoints contain an `argparse.Namespace`, which `torch.load` rejects by default from PyTorch 2.6. This broke resume, eval, second-stage training and sampling.
- **Fix:** added `weights_only=False` at each call: `util/misc.py:317`, `eval.py:47`, `main_class_cond.py:163`, and `sample_class_cond.py:33` and `:39`.

## Silently wrong results

### 3. Eval metrics only covered the last 20 shapes
- **Problem:** `eval.py` printed `SmoothedValue.avg`, which only averages the most recent 20 values. Large categories reported only their final 20 shapes.
- **Fix:** it now prints `global_avg` for IoU, Chamfer distance and F-score (`eval.py:151` and `:155`).

### 4. No reshuffling without `torchrun`
- **Problem:** the training sampler is always a `DistributedSampler`, but `set_epoch` was only called in distributed runs. Other runs repeated the same batch order every epoch.
- **Fix:** `set_epoch` is now always called (`main_ae.py:206`, `main_class_cond.py:214`).

### 5. `--weight_decay` had no effect
- **Problem:** the value was never passed to `AdamW`, so training always used PyTorch's default of 0.01, even though the flag claimed 0.05.
- **Fix:** the flag is now passed to the optimizer (`main_ae.py:187`, `main_class_cond.py:196`).
- **Decision to check:** the flag's default changed from 0.05 to 0.01 (`main_ae.py:42`, `main_class_cond.py:45`). Keeping 0.05 would have silently changed the regularization of every run that doesn't set the flag, including the README's training commands. If you want 0.05, pass `--weight_decay 0.05`.
- **Resuming:** a resumed run takes its weight decay from the checkpoint, not from the flag, because the saved optimizer state overrides it.

### 6. NaN or inflated IoU
- **Problem:** `engine_ae.py` added its epsilon after the division, so 0/0 gave NaN and every other IoU was 1e-5 too high. One such shape turned the whole evaluation's average into NaN. `eval.py` had the same 0/0 problem with no epsilon.
- **Fix:** the epsilon now goes in the denominator, matching the training code (`engine_ae.py:166-167`, `eval.py:100`).
- **Side effect:** a shape with empty labels and an empty prediction now scores IoU 0, not 1.

## Crashes in specific situations

### 7. `main_class_cond.py --eval` always crashed
- **Problem:** `evaluate` was called without its `ae` and `criterion` arguments, which raised a `TypeError`.
- **Fix:** it now passes all the arguments (`main_class_cond.py:206`).

### 8. Second stage failed on non-default point cloud sizes
- **Problem:** the autoencoder was always built for 2048 points, so any other `--point_cloud_size` hit an assertion on the first batch.
- **Fix:** it is now built with `N=args.point_cloud_size` (`main_class_cond.py:160`). Existing autoencoder checkpoints still load, because `N` doesn't change the model's weights.

### 9. Misleading data-loading errors
- **Problem:** a corrupt `.npz` was caught and printed, then the code carried on and crashed with `UnboundLocalError`, which hid the cause. Separately, a `.lst` file ending in a newline produced an empty model name, which crashed with `FileNotFoundError` for a file called `.npy`.
- **Fix:** a failed load now raises `RuntimeError` naming the file, with the original error attached (`util/shapenet.py:129-130`). Blank lines in `.lst` files are skipped (`util/shapenet.py:107`).
- **Not checked:** the real `.lst` files aren't in this repository, so it's unknown whether they end in a newline.

### 10. README commands did not work as written
- **`eval.py`:** the command was missing the required `--data_path`. The default `--pth` also pointed at a folder training never creates (`eval.py:23`). Both are fixed.
- **First training stage:** added `--data_path`. Without it, the command uses the original author's cluster path.
- **Sampling:** it pointed to `output/class_cond_dm/.../checkpoint-499.pth`, which training never writes. It now points to `output/dm/.../checkpoint-999.pth`.

The README examples use `~/data/` as the data folder, so change that to wherever you unpacked the data.

## Test results

The original code was tested from an untouched copy of `master`. To reach the later bugs at all, that copy was given temporary workarounds for issues 1 and 2.

| Issue | Test | Original code | Fixed code |
|---|---|---|---|
| 1 | Start `main_ae.py` | `ModuleNotFoundError: torch._six` | Starts |
| 2 | Load a checkpoint saved by this repo | `UnpicklingError` | Loads, and training resumes |
| 3 | Category whose true average IoU is about 0.30 | 0.123 | 0.298 |
| 4 | Two epochs in a single process | `set_epoch` never called | Called for epochs 0 and 1 |
| 5 | Train with `--weight_decay 0.03` | Optimizer used 0.01 | Optimizer used 0.03 |
| 6 | Shape with empty prediction and labels | NaN | 0.0 |
| 7 | `main_class_cond.py --eval` | `TypeError` | Runs, loss 0.993 |
| 8 | `--point_cloud_size 1024` | `AssertionError` | Trains and evaluates one epoch |
| 9 | Corrupt `.npz` | `UnboundLocalError` | `RuntimeError` naming the file |
| 9 | `.lst` ending in a newline | `FileNotFoundError` for `.npy` | Blank line skipped |
| 10 | Checkpoint names after a training run | N/A | Last epoch is saved, which matches the README's naming |

Issue 10's command lines were not run exactly as written, because they need a GPU and the real data. The arguments and the checkpoint naming they rely on were checked.

## What was not tested

- **`sample_class_cond.py`:** never run, because it is hard-coded to `cuda:0` and generates 1000 shapes. Its only change is the `weights_only=False` argument.
- **GPU and `torchrun`:** everything ran on CPU in a single process.
- **Real data and quality:** the data was synthetic, and training ran for only a few epochs, so these results say nothing about model quality.
- **Full-size autoencoder training:** the 106M-parameter model ran out of memory on the 7 GB test machine, so a narrower autoencoder was used. The diffusion model was the real `kl_d512_m512_l8_edm`.
- **`torch_cluster` and `PyMCubes`:** neither was installed, so simple stand-ins replaced them during testing. The test environment used Python 3.14, where `PyMCubes` would not build, and `torch_cluster` wasn't attempted because it needs compiling.

The test scripts were temporary and are not part of this repository.

## Noticed but not changed

These were outside the 10 issues:

- There is no `requirements.txt`, so dependency versions aren't pinned.
- Compiled `.pyc` files in `util/__pycache__/` are tracked in git, and there is no `.gitignore`.
- `torch.cuda.amp.autocast` and `GradScaler` print deprecation warnings on current PyTorch. They still work for now.
- `eval.py` calls `.cuda()` directly and `sample_class_cond.py` hard-codes `cuda:0`, so neither runs on a machine without a GPU. In `eval.py`, this ignores the `--device` flag.
