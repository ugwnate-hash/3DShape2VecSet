# Running on Kaggle GPUs

This trains the first-stage autoencoder (`main_ae.py`) on Kaggle's free GPUs, using [train_ae_kaggle.ipynb](train_ae_kaggle.ipynb). The notebook clones this repository, reads a config from `configs/`, and launches training with `torchrun` on two T4 GPUs.

There are two configs, both with C₀ = 2 latent channels:

| Config | What it is | Expected time on 2× T4 |
|---|---|---|
| `configs/kl_d512_m256_l2_d8_2cat.yaml` (default) | Reduced run: 256 latents, an 8-block decoder, the airplane and chair categories, fp16 mixed precision, 40 epochs | about 10 GPU-hours, so one 11-hour session if data loading keeps up. The notebook prints a measured estimate. |
| `configs/kl_d512_m512_l2.yaml` | The README's full setup with C₀ = 2 | about 2,000+ GPU-hours. Not practical on Kaggle. |

Kaggle gives about 30 GPU-hours a week, and one session lasts at most 12 hours. Training saves checkpoints during each epoch, stops cleanly before the 12-hour limit, and continues in the next session.

## 1. One-time setup

1. **Kaggle account:** verify your phone number in your Kaggle account settings. Kaggle requires it for GPUs and for internet access in notebooks.
2. **Push the code.** The notebook clones `https://github.com/ugwnate-hash/3DShape2VecSet.git`, branch `fix/review-issues`. Any change you want on Kaggle, configs included, must be committed and pushed to that branch first. If you use another repository or branch, change `REPO_URL` and `BRANCH` in the notebook's first code cell.
3. **Prepare the data on Kaggle.** Don't download the zips yourself: `occupancies.zip` is 589 GB and `surfaces.zip` is 56 GB. Instead:
   - Import [prepare_data_kaggle.ipynb](prepare_data_kaggle.ipynb) as a new notebook. Set the accelerator to **None** (it runs on CPU and uses no GPU quota) and Internet to **on**.
   - Run it with **Save Version → Save & Run All (Commit)**. It streams only the airplane and chair files from the zips on Google Drive (about 133 GB) and keeps a random 50k volume, 50k near-surface and 30k surface points per shape. It writes about 16 GB to its output, split files included. This takes roughly 1–3 hours, depending on Google Drive's speed.
   - Its last cell checks that every shape in the split files is complete. The output stays private to your account. Keep it that way: ShapeNet's terms of use don't allow redistribution.
   - To train other categories, change `CATEGORIES` in that notebook and `categories` in the training config. The output must stay under Kaggle's 20 GB limit, at about 1.5 MB per shape.
   - **If Google Drive refuses part of the download ("Quota exceeded")**, the notebook retries every 30 minutes for up to 9.5 hours. Drive limits how much a public file can be downloaded per day, counting all users. If it's still refused by then, the notebook finishes, keeps what it got, and reports the data as INCOMPLETE. In that case, add the notebook's **own previous output** as an input (**Add Input → Your Work**), and commit it again. Finished files are copied from that input, and only the missing ones are downloaded. Repeat until it reports the data as COMPLETE.
   - In the first attempt, all of airplane's occupancy data (45 GB) downloaded in 24 minutes, at about 32 MB/s, before Drive refused the next block.
4. **Create the notebook:** **Create → New Notebook**, then **File → Import Notebook**, and upload `kaggle/train_ae_kaggle.ipynb`.
5. **Notebook settings**, in the right-hand panel:
   - **Accelerator:** GPU T4 x2
   - **Internet:** on
   - **Input:** click **Add Input** and add the **output of the data notebook** from step 3 (under Notebooks / Your Work).

## 2. Run the smoke test first (about 20–30 minutes, including setup)

The smoke test checks, on the real GPUs, everything that couldn't be tested without one. Run it once before spending quota on training.

1. In the first code cell, set `MODE = "smoke"`. This is the default.
2. Click **Run All** in the interactive editor.
3. Section 5 prints one line per check: `PASS`, `FAIL` or `SKIP`. Section 6 prints the time estimate for the training run.
4. The report is saved to `/kaggle/working/smoke_test_report.json`. Download it from the **Output** panel. If anything fails, send this file. It contains every measured number and the error messages.

What it checks (code in [smoke_test.py](smoke_test.py)):

| Check | Passes when |
|---|---|
| environment | 2 GPUs are visible and PyTorch is 2.1 or newer (needed for fused attention). It also reports whether `torch_cluster` is the compiled wheel or the pure-PyTorch fallback. |
| fps | Farthest point sampling returns valid, well-spread indices. With the compiled wheel, it must also match a reference implementation. |
| fused attention | The fused attention kernel gives the same output as the original code (relative difference under 1e-4) |
| fp16 forward | fp16 logits stay within 5% of fp32 |
| memory and speed | The configured batch (64 per GPU) fits, fp16 losses stay finite, the loss scale doesn't collapse, and fp16 is faster than fp32 |
| learning | The model, trained for 400 steps on synthetic spheres and boxes, halves its loss and reaches validation IoU ≥ 0.5. This floor only shows that it learns; the report gives the actual IoU. |
| multi-GPU stop and resume | Real `main_ae.py` on 2 GPUs: a tiny time limit stops it at the first optimizer step with `checkpoint-0-it2.pth`. Resuming continues at that exact iteration, finishes 2 epochs, and leaves `checkpoint-0.pth` and `checkpoint-1.pth`. The category filter excludes a decoy category. |
| eval.py | `eval.py` runs on the synthetic model and reports finite IoU, Chamfer and F-score per category, with IoU ≥ 0.5 |
| real data | If the dataset is attached: the two categories load, tensors have the right shapes, labels are 0/1, and a forward pass gives a finite loss |

The smoke test uses its own synthetic shapes, so it also runs without the dataset attached. In that case the real-data check and the time estimate are skipped.

## 3. Train

1. Set `MODE = "train"` in the first code cell.
2. Click **Save Version**, choose **Save & Run All (Commit)**, and save. The run continues in the background, so you can close the tab.
3. When it finishes, open the version and its **Output** tab. Under `output/ae/kl_d512_m256_l2_d8/` you'll find:
   - `log.txt`: one JSON line per epoch, with training loss and, every 5 epochs, validation IoU (`test_iou`)
   - `checkpoint-*.pth`: the newest checkpoint
   - `stdout.log`: the full training output
   - `run_info.json`: the commit, config and command used
   - `events.out.tfevents.*`: TensorBoard logs

Training stops on its own after `time_budget_hours` (11 h by default). It saves a checkpoint before stopping, so at most a few minutes of work are lost.

## 4. Continue in the next session

1. Open the notebook in the editor.
2. Click **Add Input** and add **this notebook's own output** from the previous version (under **Notebooks**, or *Your Work*).
3. **Save & Run All (Commit)** again.

With `resume: auto` in the config, the notebook finds the newest checkpoint among its inputs, including a mid-epoch one. It continues from the exact epoch and iteration. Keep the GPU count and batch size the same between sessions, so the interrupted epoch continues in the same data order.

## 5. Evaluate

After training, add a cell (or use a new notebook with the same inputs) and run:

```
!cd /tmp/run/3DShape2VecSet && python eval.py --model kl_d512_m256_l2_d8 \
    --pth /kaggle/working/output/ae/kl_d512_m256_l2_d8/checkpoint-49.pth \
    --data_path /tmp/run/data --categories 02691156,03001627
```

Adjust the checkpoint path to where your checkpoint is, for example under `/kaggle/input/...` if it comes from a previous version's output. `eval.py` uses the `test` split, so the data needs `test.lst` files.

For reference, the paper reports IoU 0.816 for C₀ = 2 with the full model and full training on 8 A100s. This reduced run will probably score lower.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `git clone` fails | Internet is off in the notebook settings, or the branch isn't pushed |
| "The config asks for 2 GPUs but 1 are available" | Set the accelerator to GPU T4 x2, not P100 |
| "batch_size … does not fit in GPU memory" | Lower `hardware.batch_size` and raise `accum_iter` so that 2 × batch_size × accum_iter stays 512 |
| The estimate says "limited by data loading" | The 4 CPU cores can't keep up. Later epochs may speed up once the files are cached in RAM. Otherwise, preprocessing the data into smaller files would help. |
| "No torch_cluster wheel … using the pure-PyTorch fps fallback" | Not an error. Results are the same, a few percent slower. |
| `test_iou` stops improving at a low value | Raise `epochs` in the config before changing anything else |
| A run hit the 12-hour limit without output | Lower `time_budget_hours`. Setup and the benchmark take about 10–15 minutes before training starts. |
