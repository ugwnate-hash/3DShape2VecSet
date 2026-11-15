"""GPU smoke test for the Kaggle setup.

Checks, with assertions, the parts of the training setup that can't be tested on a CPU:
fp16 mixed precision on the GPU, two-GPU training (including the time-limit stop and the
mid-epoch resume), torch_cluster, fused attention, GPU memory at the configured batch size,
eval.py, and whether the model learns at all on simple synthetic shapes. If real data is
given, it also checks that it loads.

Run from the repository root:
    python kaggle/smoke_test.py --config configs/kl_d512_m256_l2_d8_2cat.yaml \
        --report /kaggle/working/smoke_test_report.json [--data /path/to/data]

Each check prints PASS, FAIL or SKIP. The report file collects every measured number,
so it can be pasted back as-is. The exit code is 1 if any check failed.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import models_ae  # noqa: E402

RESULTS = []
SPHERES, BOXES, DECOY = '02691156', '03001627', '04379243'   # folder names must be real ShapeNet ids


def check(name):
    """Run a check function, record PASS / FAIL / SKIP and its details."""
    def wrap(fn):
        def run(*args, **kwargs):
            t = time.time()
            details = {}
            try:
                out = fn(details, *args, **kwargs)
                status = 'SKIP' if out == 'SKIP' else 'PASS'
            except AssertionError as e:
                status = 'FAIL'
                details['error'] = 'assertion failed: %s' % e
            except Exception as e:
                status = 'FAIL'
                details['error'] = '%s: %s' % (type(e).__name__, e)
                details['traceback'] = traceback.format_exc()[-2000:]
            details['seconds'] = round(time.time() - t, 1)
            RESULTS.append({'check': name, 'status': status, 'details': details})
            shown = {k: v for k, v in details.items() if k != 'traceback'}
            print('[%s] %s  %s' % (status, name, json.dumps(shown, default=str)), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return run
    return wrap


# ----------------------------------------------------------------------------
# Synthetic data in the ShapeNet layout: spheres and boxes, plus a decoy category
# that must be filtered out by --categories.
# ----------------------------------------------------------------------------

def make_shape(kind, rng, n=20000):
    c = rng.uniform(-0.2, 0.2, 3)
    if kind == 'sphere':
        r = rng.uniform(0.35, 0.6)
        inside = lambda x: np.linalg.norm(x - c, axis=1) < r
        d = rng.normal(size=(n, 3))
        surf = c + r * d / np.linalg.norm(d, axis=1, keepdims=True)
    else:
        h = rng.uniform(0.2, 0.6, 3)
        inside = lambda x: np.all(np.abs(x - c) < h, axis=1)
        areas = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]] * 2)
        face = rng.choice(6, n, p=areas / areas.sum())
        surf = rng.uniform(-1, 1, (n, 3)) * h
        axis, sign = face % 3, np.where(face < 3, 1.0, -1.0)
        surf[np.arange(n), axis] = sign * h[axis]
        surf = surf + c
    vol = rng.uniform(-1, 1, (n, 3))
    near = surf + rng.normal(scale=0.02, size=surf.shape)
    return (vol.astype(np.float32), inside(vol).astype(np.float32),
            near.astype(np.float32), inside(near).astype(np.float32), surf.astype(np.float32))


def make_dataset(root, per_split=(8, 4, 4), seed=0):
    rng = np.random.default_rng(seed)
    for cat, kind in [(SPHERES, 'sphere'), (BOXES, 'box'), (DECOY, 'sphere')]:
        pdir = os.path.join(root, 'ShapeNetV2_point', cat)
        mdir = os.path.join(root, 'ShapeNetV2_watertight', cat, '4_pointcloud')
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(mdir, exist_ok=True)
        i = 0
        for split, count in zip(['train', 'val', 'test'], per_split):
            names = []
            for _ in range(count):
                name = '%s_%03d' % (kind, i)
                i += 1
                vol, vl, near, nl, surf = make_shape(kind, rng)
                np.savez(os.path.join(pdir, name + '.npz'), vol_points=vol, vol_label=vl,
                         near_points=near, near_label=nl)
                np.save(os.path.join(pdir, name + '.npy'), np.array(1.0))
                np.savez(os.path.join(mdir, name + '.npz'), points=surf)
                names.append(name)
            with open(os.path.join(pdir, split + '.lst'), 'w') as f:
                f.write('\n'.join(names) + '\n')
    return root


def batch_iou(logits, labels):
    pred = (logits >= 0).float()
    inter = (pred * labels).sum(1)
    union = (pred + labels).gt(0).sum(1).float() + 1e-5
    return (inter / union).mean().item()


def autocast(device, enabled):
    dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def reference_fps(pos, n_batch, m, start=None):
    """Plain farthest point sampling, starting at index 0 of each batch element."""
    p = pos.view(n_batch, -1, 3)
    n = p.shape[1]
    ar = torch.arange(n_batch, device=pos.device)
    idx = torch.empty(n_batch, m, dtype=torch.long, device=pos.device)
    dist = torch.full((n_batch, n), float('inf'), device=pos.device)
    cur = torch.zeros(n_batch, dtype=torch.long, device=pos.device)
    for i in range(m):
        idx[:, i] = cur
        dist = torch.minimum(dist, ((p - p[ar, cur][:, None]) ** 2).sum(-1))
        cur = dist.argmax(-1)
    return (idx + ar[:, None] * n).view(-1)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

@check('environment')
def check_environment(d, want_gpus):
    d['torch'] = torch.__version__
    d['cuda'] = torch.version.cuda
    d['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    d['fused_attention'] = models_ae._FUSED_ATTENTION
    import torch_cluster
    d['torch_cluster'] = 'fallback (pure PyTorch)' if 'shims' in (torch_cluster.__file__ or '') else 'compiled wheel'
    assert models_ae._FUSED_ATTENTION, 'PyTorch < 2.1: fused attention is off, fp16 training is less stable'
    assert len(d['gpus']) >= want_gpus, 'config wants %d GPUs, found %d' % (want_gpus, len(d['gpus']))


@check('fps (farthest point sampling)')
def check_fps(d, device):
    from torch_cluster import fps
    torch.manual_seed(0)
    B, N, M = 4, 2048, 512
    pos = (torch.rand(B * N, 3, device=device) * 2 - 1)
    batch = torch.repeat_interleave(torch.arange(B, device=device), N)
    idx = fps(pos, batch, ratio=M / N)
    assert idx.numel() == B * M, 'expected %d indices, got %d' % (B * M, idx.numel())
    per = idx.view(B, M)
    assert ((per // N) == torch.arange(B, device=device)[:, None]).all(), 'indices not grouped by shape'
    assert all(len(set(r.tolist())) == M for r in per), 'duplicate indices'
    # Farthest point sampling spreads points: minimum spacing far above random sampling's.
    s = pos[idx].view(B, M, 3)
    eye = torch.eye(M, device=device) * 9
    fps_gap = (torch.cdist(s, s) + eye).min().item()
    r = pos.view(B, N, 3)[:, torch.randperm(N, device=device)[:M]]
    rnd_gap = (torch.cdist(r, r) + eye).min().item()
    d['min_spacing_fps'], d['min_spacing_random'] = round(fps_gap, 4), round(rnd_gap, 4)
    assert fps_gap > 5 * rnd_gap, 'points are not spread out like farthest point sampling'
    try:
        same = (fps(pos, batch, ratio=M / N, random_start=False) == reference_fps(pos, B, M)).float().mean().item()
        d['matches_reference_fps'] = round(same, 4)
        assert same > 0.99, 'differs from reference farthest point sampling'
    except TypeError:
        d['matches_reference_fps'] = 'not compared (no random_start argument)'


@check('fused attention == original attention')
def check_attention(d, device, model_name):
    torch.manual_seed(0)
    m = models_ae.__dict__[model_name]().to(device).eval()
    pc = torch.rand(2, 2048, 3, device=device) * 2 - 1
    q = torch.rand(2, 2048, 3, device=device) * 2 - 1
    with torch.no_grad():
        torch.manual_seed(1)
        _, z = m.encode(pc)
        a = m.decode(z, q)
        models_ae._FUSED_ATTENTION = False
        try:
            torch.manual_seed(1)
            _, z2 = m.encode(pc)
            b = m.decode(z2, q)
        finally:
            models_ae._FUSED_ATTENTION = True
    rel = ((a - b).norm() / b.norm()).item()
    d['relative_difference_fp32'] = '%.2e' % rel
    assert rel < 1e-4, 'fused attention output differs from the original'


@check('fp16 forward pass accuracy')
def check_fp16_forward(d, device, model_name):
    if device.type != 'cuda':
        return 'SKIP'
    torch.manual_seed(0)
    m = models_ae.__dict__[model_name]().to(device).eval()
    pc = torch.rand(4, 2048, 3, device=device) * 2 - 1
    q = torch.rand(4, 2048, 3, device=device) * 2 - 1
    with torch.no_grad():
        torch.manual_seed(1)
        ref = m(pc, q)['logits'].float()
        torch.manual_seed(1)
        with autocast(device, True):
            half = m(pc, q)['logits'].float()
        # The point embedding guard: fp32 Fourier features vs what fp16 would give.
        x = torch.rand(1, 4096, 3, device=device) * 2 - 1
        f32 = m.point_embed.embed(x, m.point_embed.basis)
        f16 = m.point_embed.embed(x.half(), m.point_embed.basis.half()).float()
        with autocast(device, True):
            guarded = m.point_embed(x).float()
        unguarded_err = (f16 - f32).abs().max().item()
    rel = ((half - ref).norm() / ref.norm()).item()
    d['logits_relative_error'] = '%.2e' % rel
    d['fourier_feature_error_if_fp16'] = round(unguarded_err, 4)
    d['point_embed_output_dtype_under_autocast'] = str(guarded.dtype)
    assert torch.isfinite(half).all(), 'non-finite logits in fp16'
    assert rel < 5e-2, 'fp16 logits too far from fp32'


@check('memory and speed at the configured batch size')
def check_memory_speed(d, device, model_name, batch_size, amp, steps=10):
    if device.type != 'cuda':
        return 'SKIP'

    def run(use_amp):
        torch.manual_seed(0)
        m = models_ae.__dict__[model_name]().to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        crit = torch.nn.BCEWithLogitsLoss()
        torch.cuda.reset_peak_memory_stats()
        losses = []

        def step():
            pc = torch.rand(batch_size, 2048, 3, device=device) * 2 - 1
            q = torch.rand(batch_size, 2048, 3, device=device) * 2 - 1
            y = (torch.rand(batch_size, 2048, device=device) > 0.5).float()
            with autocast(device, use_amp):
                out = m(pc, q)
                loss = crit(out['logits'].float(), y) + 1e-3 * out['kl'].float().sum() / batch_size
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        sec = (time.time() - t) / steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        scale = scaler.get_scale() if use_amp else None
        del m, opt
        torch.cuda.empty_cache()
        return sec, peak, losses, scale

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    d['gpu_memory_gb'] = round(total_gb, 1)
    sec, peak, losses, scale = run(amp)
    d['amp'] = amp
    d['sec_per_step'] = round(sec, 3)
    d['samples_per_sec_one_gpu'] = round(batch_size / sec, 1)
    d['peak_memory_gb'] = round(peak, 2)
    d['loss_scale'] = scale
    assert all(math.isfinite(l) for l in losses), 'non-finite loss'
    assert peak < 0.95 * total_gb, 'peak memory too close to the GPU limit'
    if amp:
        assert scale is not None and scale >= 1.0, 'loss scale collapsed: fp16 gradients keep overflowing'
        try:
            sec32, peak32, _, _ = run(False)
            d['fp32_sec_per_step'] = round(sec32, 3)
            d['fp32_peak_memory_gb'] = round(peak32, 2)
            d['amp_speedup'] = round(sec32 / sec, 2)
            assert sec32 / sec > 1.1, 'mixed precision is not faster than fp32'
        except torch.cuda.OutOfMemoryError:
            d['fp32_comparison'] = 'fp32 does not fit at this batch size'
            torch.cuda.empty_cache()


@check('the model learns synthetic spheres and boxes')
def check_learning(d, device, model_name, data_root, out_dir, amp, steps, batch_size, min_iou):
    import argparse as ap
    from util.datasets import build_shape_surface_occupancy_dataset
    args = ap.Namespace(data_path=data_root, point_cloud_size=2048, categories='%s,%s' % (SPHERES, BOXES))
    train = build_shape_surface_occupancy_dataset('train', args)
    val = build_shape_surface_occupancy_dataset('val', args)
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    m = models_ae.__dict__[model_name]().to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == 'cuda')
    crit = torch.nn.BCEWithLogitsLoss()
    losses, step = [], 0
    while step < steps:
        for points, labels, surface, _ in loader:
            points, labels, surface = points.to(device), labels.to(device), surface.to(device)
            for g in opt.param_groups:        # short linear warm-up
                g['lr'] = 2e-4 * min(1.0, (step + 1) / 20)
            with autocast(device, amp):
                out = m(surface, points)
                logits = out['logits'].float()
                loss = (crit(logits[:, :1024], labels[:, :1024]) + 0.1 * crit(logits[:, 1024:], labels[:, 1024:])
                        + 1e-3 * out['kl'].float().sum() / out['kl'].shape[0])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break
    m.eval()
    ious = []
    with torch.no_grad():
        for i in range(len(val)):
            points, labels, surface, _ = val[i]
            with autocast(device, amp):
                logits = m(surface[None].to(device), points[None].to(device))['logits'].float()
            ious.append(batch_iou(logits, labels[None].to(device)))
    first, last = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
    d['steps'], d['batch_size'] = steps, batch_size
    d['loss_first10'], d['loss_last10'] = round(first, 4), round(last, 4)
    d['val_iou'] = round(float(np.mean(ious)), 4)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model': m.state_dict()}, os.path.join(out_dir, 'checkpoint.pth'))
    assert all(math.isfinite(l) for l in losses), 'non-finite loss during training'
    assert last < 0.5 * first, 'loss did not halve'
    assert d['val_iou'] >= min_iou, 'val IoU %.3f below %.2f' % (d['val_iou'], min_iou)


def run_cmd(cmd, timeout, cwd=REPO):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


@check('multi-GPU training: time-limit stop and mid-epoch resume')
def check_ddp(d, device, model_name, data_root, out_dir, n_gpus):
    # 16 training shapes x 16 replicas = 256 samples per epoch. With batch 8 per process,
    # each process does 256 / (8 * n) iterations; accum_iter 2 -> an optimizer step every 2.
    nproc = max(1, n_gpus)
    iters = 256 // (8 * nproc)
    if device.type == 'cuda':
        launch = ['torchrun', '--standalone', '--nproc_per_node=%d' % nproc]
    else:
        launch = [sys.executable]   # local CPU run of the same code path, single process
    base = launch + ['main_ae.py', '--model', model_name, '--data_path', data_root,
                     '--categories', '%s,%s' % (SPHERES, BOXES), '--amp',
                     '--epochs', '2', '--warmup_epochs', '1', '--batch_size', '8', '--accum_iter', '2',
                     '--num_workers', '1', '--save_every_epochs', '1', '--output_dir', out_dir, '--log_dir', out_dir,
                     '--device', device.type]
    d['processes'] = nproc

    # Leg 1: a tiny time limit stops training at the first optimizer step.
    code, log1 = run_cmd(base + ['--max_hours', '1e-9'], timeout=900)
    d['leg1_exit'] = code
    assert code == 0, 'leg 1 exited with %d: %s' % (code, log1[-1500:])
    assert "['%s', '%s']" % (SPHERES, BOXES) in log1, 'category filter: decoy category not excluded'
    assert 'Time limit' in log1, 'no time-limit stop message'
    mid = sorted(glob.glob(os.path.join(out_dir, 'checkpoint-*-it*.pth')))
    d['leg1_checkpoints'] = [os.path.basename(p) for p in glob.glob(os.path.join(out_dir, 'checkpoint-*.pth'))]
    assert [os.path.basename(p) for p in mid] == ['checkpoint-0-it2.pth'], 'expected checkpoint-0-it2.pth'
    assert not os.path.exists(os.path.join(out_dir, 'checkpoint-0.pth')), 'epoch 0 should not be finished'

    # Leg 2: resume mid-epoch and finish both epochs.
    code, log2 = run_cmd(base + ['--resume', mid[0]], timeout=1800)
    d['leg2_exit'] = code
    assert code == 0, 'leg 2 exited with %d: %s' % (code, log2[-1500:])
    assert 'Continuing at epoch 0, iteration 2' in log2, 'resume did not continue at epoch 0, iteration 2'
    assert re.search(r'Epoch: \[0\]\s+\[\s*0/%d\]' % (iters - 2), log2), \
        'resumed epoch should run the remaining %d of %d iterations' % (iters - 2, iters)
    assert re.search(r'Epoch: \[1\]\s+\[\s*0/%d\]' % iters, log2), 'next epoch should be a full %d iterations' % iters
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, 'checkpoint-*.pth')))
    d['leg2_checkpoints'] = names
    assert names == ['checkpoint-0.pth', 'checkpoint-1.pth'], 'unexpected checkpoints %s' % names
    with open(os.path.join(out_dir, 'log.txt')) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    d['log_rows'] = [{k: v for k, v in r.items() if k in ('epoch', 'train_loss', 'test_iou')} for r in rows]
    assert [r['epoch'] for r in rows] == [0, 1], 'log.txt epochs %s' % [r['epoch'] for r in rows]
    assert all(math.isfinite(r['train_loss']) for r in rows), 'non-finite training loss'
    assert 'test_iou' in rows[0], 'epoch 0 should have been validated'


@check('eval.py on the learned synthetic model')
def check_eval(d, device, model_name, data_root, ckpt, min_iou):
    if device.type != 'cuda':
        return 'SKIP'
    assert os.path.exists(ckpt), 'no checkpoint from the learning check'
    code, log = run_cmd([sys.executable, 'eval.py', '--model', model_name, '--pth', ckpt, '--data_path', data_root,
                         '--categories', '%s,%s' % (SPHERES, BOXES)], timeout=1800)
    d['exit'] = code
    assert code == 0, 'eval.py exited with %d: %s' % (code, log[-1500:])
    rows = re.findall(r'^(%s|%s) ([-\d.eE]+) ([-\d.eE]+) ([-\d.eE]+)$' % (SPHERES, BOXES), log, re.M)
    results = {}
    for cat, iou, cd, fs in rows:   # the final summary lines come last
        results[cat] = {'iou': float(iou), 'chamfer': float(cd), 'fscore': float(fs)}
    d['results'] = results
    assert set(results) == {SPHERES, BOXES}, 'per-category results missing'
    for cat, r in results.items():
        assert all(math.isfinite(v) for v in r.values()), 'non-finite metric for %s' % cat
        assert r['iou'] >= min_iou, 'eval IoU %.3f for %s below %.2f' % (r['iou'], cat, min_iou)


@check('real data loads')
def check_real_data(d, device, model_name, data_root, categories, amp):
    if not data_root:
        return 'SKIP'
    import argparse as ap
    from util.datasets import build_shape_surface_occupancy_dataset
    args = ap.Namespace(data_path=data_root, point_cloud_size=2048, categories=categories)
    train = build_shape_surface_occupancy_dataset('train', args)
    val = build_shape_surface_occupancy_dataset('val', args)
    d['train_shapes'] = len(train.models)
    d['val_shapes'] = len(val.models)
    d['per_category'] = {c: sum(1 for x in train.models if x['category'] == c) for c in sorted({x['category'] for x in train.models})}
    assert len(train.models) > 0 and len(val.models) > 0, 'empty train or val split'
    rng = np.random.default_rng(0)
    batch = [train[int(i)] for i in rng.choice(len(train), 4, replace=False)]
    points = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    surface = torch.stack([b[2] for b in batch])
    d['points_shape'], d['surface_shape'] = list(points.shape), list(surface.shape)
    d['occupied_fraction_volume'] = round(labels[:, :1024].mean().item(), 4)
    d['occupied_fraction_near_surface'] = round(labels[:, 1024:].mean().item(), 4)
    assert points.shape == (4, 2048, 3) and labels.shape == (4, 2048) and surface.shape == (4, 2048, 3)
    assert torch.isfinite(points).all() and torch.isfinite(surface).all()
    assert set(labels.unique().tolist()) <= {0.0, 1.0}, 'labels are not 0/1'
    assert 0.001 < d['occupied_fraction_volume'] < 0.9, 'implausible occupancy fraction'
    assert surface.abs().max() <= 1.0 + 1e-4, 'surface points outside [-1, 1]'
    m = models_ae.__dict__[model_name]().to(device).train()
    with autocast(device, amp):
        out = m(surface.to(device), points.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out['logits'].float(), labels.to(device))
    d['untrained_loss'] = round(loss.item(), 4)
    assert math.isfinite(loss.item()), 'non-finite loss on real data'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='training config (model, batch size, amp, categories)')
    parser.add_argument('--report', default='smoke_test_report.json')
    parser.add_argument('--data', default=None, help='real data folder (optional)')
    parser.add_argument('--workdir', default=None, help='scratch folder (default: a new temporary folder)')
    parser.add_argument('--learn_steps', type=int, default=400)
    parser.add_argument('--learn_batch', type=int, default=16)
    parser.add_argument('--min_iou', type=float, default=0.5,
                        help='IoU the learning and eval checks must reach: a floor showing the model learns, '
                             'not a quality target (the measured IoU is in the report)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg['model']
    hw, targs = cfg['hardware'], cfg['args']
    amp = bool(targs.get('amp'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    work = args.workdir or tempfile.mkdtemp(prefix='smoke_')
    print('model %s | amp %s | device %s | %d GPU(s) | workdir %s' % (model_name, amp, device, n_gpus, work), flush=True)

    data_root = make_dataset(os.path.join(work, 'synthetic'))

    check_environment(int(hw['nproc_per_node']) if device.type == 'cuda' else 0)
    check_fps(device)
    check_attention(device, model_name)
    check_fp16_forward(device, model_name)
    check_memory_speed(device, model_name, int(hw['batch_size']), amp)
    check_learning(device, model_name, data_root, os.path.join(work, 'learn'), amp,
                   args.learn_steps, args.learn_batch, args.min_iou)
    check_ddp(device, model_name, data_root, os.path.join(work, 'ddp'), min(n_gpus, int(hw['nproc_per_node'])))
    check_eval(device, model_name, data_root, os.path.join(work, 'learn', 'checkpoint.pth'), args.min_iou)
    check_real_data(device, model_name, args.data, targs.get('categories'), amp)

    summary = {s: sum(r['status'] == s for r in RESULTS) for s in ('PASS', 'FAIL', 'SKIP')}
    report = {'summary': summary, 'config': args.config, 'model': model_name, 'results': RESULTS}
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('\nSUMMARY: %(PASS)d passed, %(FAIL)d failed, %(SKIP)d skipped' % summary)
    print('report written to', args.report)
    sys.exit(1 if summary['FAIL'] else 0)


if __name__ == '__main__':
    main()
