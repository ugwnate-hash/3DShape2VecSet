# Experiment: a 2-channel latent space (C₀ = 2)

## Question

What does 3DShape2VecSet's latent space look like when every latent vector has only two channels?

The KL autoencoder encodes a shape as a set of M latent vectors with C₀ channels each. With C₀ = 2, each shape is literally a set of points in the plane, so its latents can be plotted directly, with no PCA or t-SNE in between. We want to see:

- whether nearby surface regions of a shape map to nearby latents
- how latents of different categories are distributed relative to each other and to the N(0, 1) prior
- how much information each channel carries

## Why C₀ = 2 is interesting

The paper's ablation (Table 5) shows that C₀ = 2 is where the bottleneck starts to bite. Values are reconstruction quality on ShapeNet with the full model and training:

| C₀ | IoU | Chamfer | F-score |
|---|---|---|---|
| 1 | 0.727 | 0.133 | 0.703 |
| **2** | **0.816** | **0.087** | **0.815** |
| 4 | 0.957 | 0.038 | 0.967 |
| 8 | 0.960 | 0.038 | 0.967 |

So C₀ = 2 still reconstructs recognizable shapes but is clearly compressed, which makes it the most informative setting to visualize.

No pretrained C₀ = 2 model is published. The authors release only `kl_d512_m512_l8` and `ae_d512_m512`. So we train our own.

## Setup

The README setup with C₀ = 2 would need about 2,000+ GPU-hours on Kaggle's 2× T4, so the run is reduced. Config: [`configs/kl_d512_m256_l2_d8_2cat.yaml`](../configs/kl_d512_m256_l2_d8_2cat.yaml).

| | README setup | This experiment | Why |
|---|---|---|---|
| Latent channels C₀ | 8 | **2** | the subject of the experiment |
| Latents per shape M | 512 | **256** | about half the cost; 256 points per plot is still plenty |
| Decoder self-attention blocks | 24 | **8** | about 0.4× the cost; all of these blocks are in the decoder, so the encoder that produces the latents is unchanged |
| Categories | 55 | **airplane, chair** | fewer shapes per epoch; two very different shape types should be easy to tell apart in plots |
| Points stored per shape (volume / near-surface / surface) | 500k / 500k / 100k | **50k / 50k / 30k**, a fixed random subset made by [`kaggle/prepare_subset.py`](../kaggle/prepare_subset.py) | The full data is 645 GB; this subset is about 16 GB and fits on Kaggle. Training draws 1024 + 1024 query points and 2048 surface points per sample, so each kept point is reused about 13 times over 40 epochs. Validation IoU is computed on the 50k kept volume points instead of 500k. |
| Precision | fp32 | **fp16 mixed precision** | faster on T4 tensor cores. The point embedding's Fourier features stay in fp32, since fp16 would distort them. |
| Epochs | 200 | **40** (2 warm-up) | sized to fit one 11-hour Kaggle session; each shape is still seen 640 times (16 per epoch) |
| Effective batch, learning rate | 512, 2e-4 | same | |
| KL weight, input points, query points | 1e-3, 2048, 1024 + 1024 | same | |

The model has 39M parameters, against 106M for the full model. Training costs 81 GFLOP per sample, about 0.21× of the full model (measured). At the speed measured on 2× T4 (about 15 min per epoch), 40 epochs take about 10 hours, which is one Kaggle session. That figure assumes about 10k training shapes and data loading that keeps up. The notebook measures the real figure before training. If training runs over, it saves a checkpoint and continues in the next session.

How to run it: [`kaggle/KAGGLE.md`](../kaggle/KAGGLE.md).

## Expectations

These are hypotheses, not results:

- Reconstruction IoU will be **below the paper's 0.816**, because the model is smaller and trained for less time. By how much is unknown. As a rough reading, above 0.7 is usable for visualization, and if IoU levels off well below that, train more epochs first.
- The 2-channel bottleneck dominates the error, so cutting the decoder from 24 to 8 blocks should cost relatively little. This is untested.
- Because the KL weight is small (1e-3), the latents will probably not look like N(0, 1). How far they spread is one of the things to look at.

## Latent pairs: two chairs, two airplanes

Script: [`experiments/plot_latent_pairs.py`](plot_latent_pairs.py). It encodes two chairs and two airplanes from the `test` split (not seen in training) and draws their 4 × 256 latents in one scatter. Each class has one hue (chairs blue, airplanes orange), and each shape has one tone of it. The figure also shows the input point clouds and a 4 × 4 matrix of distances between the latent sets.

**Hypothesis:** the two chairs' latents are close to each other, the two airplanes' latents are close to each other, and chairs are far from airplanes.

**Distance between two shapes.** A shape's latents are an unordered set: their order comes from farthest point sampling with a random start. So the distance is the mean Euclidean distance under the optimal one-to-one matching of the two sets of 256 points (earth mover's distance), not a comparison of latent *i* with latent *i*.

**Two shapes per class can't prove the hypothesis on their own.** The script therefore also encodes `--n_stats` shapes per class (50 by default) and reports:
- the mean within-chair, within-airplane and between-class distances,
- the share of shapes whose nearest neighbour in latent space has the same class,
- the AUC that a within-class pair is closer than a between-class pair,
- the percentile of each plotted pair in those distributions, which shows whether the four plotted shapes are typical.

Report these numbers together with the figure.

**Baseline: an untrained encoder.** The encoder is a function of the input geometry, so even random weights can map chairs and airplanes to different places. On synthetic boxes and crosses, a random-weight encoder already separated the two classes, with an AUC of 0.95. Run the same script with `--pth random`. The trained model's separation means something only to the extent that it beats this baseline.

Also check each shape's reconstruction IoU, which the script prints: if the autoencoder doesn't reconstruct the shapes, its latents say little.

Run it after training (GPU or CPU; it takes a few minutes, mostly the `--n_stats` encodings):

```
python experiments/plot_latent_pairs.py --pth <checkpoint.pth> --data_path <data> --out output/analysis/latent_pairs
python experiments/plot_latent_pairs.py --pth random          --data_path <data> --out output/analysis/latent_pairs_random
```

To choose the shapes yourself, pass `--chairs <id>,<id> --airplanes <id>,<id>`; by default they are drawn at random from the split with `--seed`. Outputs: `latent_pairs.png`/`.pdf`, `latents.csv` (every plotted latent, with its mean, std and FPS surface point), `latent_pairs.npz` and `summary.json`. On Kaggle, add the data notebook's output and the training output as inputs, clone the repository, and set `--data_path` to the folder that holds `ShapeNetV2_point` and `ShapeNetV2_watertight`. The script uses a pure-PyTorch fps if `torch_cluster` isn't installed.

## Planned analysis

To do after training (item 2 is partly covered by `plot_latent_pairs.py` above):

1. **One shape:** scatter its 256 latents, colored by the x, y or z position of the surface point each latent comes from (the encoder's farthest-point samples). This tests whether the latent layout follows the geometry.
2. **Many shapes:** overlay latents of many airplanes and chairs, colored by category, together with the N(0, 1) prior.
3. **Posterior noise per channel:** the encoder's predicted standard deviation for each channel. A channel whose value stays near 1 carries almost no information.
4. **Interpolation:** decode linear interpolations between two shapes' latent sets.

Use the posterior **mean** for plots. `encode()` returns a random sample, and it doesn't return the sampled surface points, so the plotting script must compute both itself.

## Status

- Code: checkpointing that resumes mid-epoch, the reduced model, category filter, mixed precision, and the Kaggle notebook are done. They were tested on CPU only.
- GPU smoke test (`kaggle/smoke_test.py`), first run on 2× T4 (PyTorch 2.11, CUDA 13.0):
  - Passed: fps, fused attention, fp16 accuracy, memory and speed, learning.
    - fp16 is 2.41× faster than fp32: 0.64 s per step of 64 on one GPU, about 100 samples/s, with 9.5 of 15.6 GB peak memory.
    - Without the fp32 guard, the Fourier features would be off by up to 0.31.
  - Failed, now fixed: two failures caused by the test, not the training code (stale files in the work folder, and synthetic shapes that weren't scaled like the training inputs). The two-GPU resume and `eval.py` still need a rerun.
  - Projected from the measured speed, assuming about 10k training shapes: about 15 min per epoch, so roughly 10 h for the 40 configured epochs if the GPUs set the pace. Data loading hasn't been measured on real data yet.
- GPU smoke test, second run with real data: all 9 checks passed. Two-GPU stop and resume work, `eval.py` runs, and the synthetic learning check reached validation IoU 0.73. The real data has 10,013 training shapes (3,742 airplanes, 6,271 chairs) and 270 validation shapes. At 96 samples/s per GPU, one epoch takes about 15.5 min, so 40 epochs take about 10.3 h if the GPUs set the pace.
- Data: complete (airplane + chair, about 16 GB), built on Kaggle by `kaggle/prepare_data_kaggle.ipynb` over several sessions. Google Drive's daily download quota (shared by all users of the files) refused `occupancies.zip` several times; the retry loop got it through.
- Training: not started.
- `experiments/plot_latent_pairs.py`: tested end to end on CPU with synthetic shapes and random weights only; not yet run on a trained model.
- Results: none yet.
