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

## Planned analysis

To do after training (no plotting script exists yet):

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
- Training: not started.
- Results: none yet.
