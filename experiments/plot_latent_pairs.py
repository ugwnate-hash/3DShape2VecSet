"""C0 = 2 experiment: plot the latents of two chairs and two airplanes in the plane.

With C0 = 2 every latent vector has two channels, so a shape's latent set (M = 256 vectors
for kl_d512_m256_l2_d8) is a set of points in 2D and can be plotted directly, with no
projection in between. This script encodes two chairs and two airplanes and draws:

  * top row: the 2048 input surface points of each shape (what the encoder sees),
  * bottom left: the latents of the four shapes in one scatter. One class is one hue
    (chair blue, airplane orange) and each shape is one tone of that hue,
  * bottom right: the 4 x 4 matrix of distances between the latent sets.

It plots the posterior mean of each latent. model.encode() returns a random sample from the
posterior and doesn't return the farthest-point samples, so the encoder is re-run here step by step.

The latents are a set: their order comes from farthest point sampling with a random start,
so latent i of one shape has nothing to do with latent i of another. The distance between
two shapes is therefore a set distance, the mean Euclidean distance under the optimal
one-to-one matching of their latents (earth mover's distance between equal-size point sets).

Two shapes per class can't show that classes separate on their own: four other shapes
could tell a different story. So the script also encodes --n_stats shapes per class from
the same split and reports within-class and between-class distances, the share of shapes
whose nearest neighbour in latent space has the same class, and where each plotted
pair's distance falls in those distributions. Report those numbers next to the figure.

Outputs in --out:
  latent_pairs.png / .pdf   the figure
  latents.csv               every plotted latent (mean and std per channel) with its FPS surface point
  latent_pairs.npz          the same as arrays, plus the input point clouds
  summary.json              shape ids, distances, reconstruction IoU, and the --n_stats statistics

    python experiments/plot_latent_pairs.py --pth output/ae/kl_d512_m256_l2_d8/checkpoint-39.pth \\
        --data_path ~/data --out output/analysis/latent_pairs
"""
import argparse
import csv
import json
import math
import os
import sys
import types
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fps(src, batch=None, ratio=0.5, random_start=True):
    """Pure-PyTorch stand-in for torch_cluster.fps (same as the Kaggle notebook's fallback).
    Valid when every batch element has the same number of points, as in models_ae.py."""
    if batch is None:
        batch = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
    B = int(batch.max()) + 1
    N = src.shape[0] // B
    M = int(-(-ratio * N // 1))
    p = src.view(B, N, -1)
    ar = torch.arange(B, device=src.device)
    idx = torch.empty(B, M, dtype=torch.long, device=src.device)
    dist = torch.full((B, N), float('inf'), device=src.device, dtype=src.dtype)
    cur = (torch.randint(0, N, (B,), device=src.device) if random_start
           else torch.zeros(B, dtype=torch.long, device=src.device))
    for i in range(M):
        idx[:, i] = cur
        dist = torch.minimum(dist, ((p - p[ar, cur][:, None]) ** 2).sum(-1))
        cur = dist.argmax(-1)
    return (idx + ar[:, None] * N).view(-1)


try:
    import torch_cluster  # noqa: F401
except ImportError:
    _shim = types.ModuleType('torch_cluster')
    _shim.fps = _fps
    sys.modules['torch_cluster'] = _shim
    print('torch_cluster not installed: using the pure-PyTorch fps fallback')

import models_ae  # noqa: E402  (after the torch_cluster fallback)

# One class = one hue, one shape = one tone of it. Light mode, chart surface #fcfcfb.
# Checked with the dataviz skill's validate_palette.js (--pairs all): all four inside the
# lightness band, worst CVD dE 20.2, worst normal-vision dE 20.7. The two light tones sit
# below 3:1 contrast on the surface, so every shape is named in the legend and the
# values are written to latents.csv.
CLASSES = [
    # name,      synset,     (dark tone, light tone)
    ('chair',    '03001627', ('#184f95', '#5598e7')),
    ('airplane', '02691156', ('#a33d12', '#ee7a4c')),
]
SURFACE = '#fcfcfb'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'


def shape_seed(seed, cat, sid):
    return (seed * 1000003 + zlib.crc32(('%s/%s' % (cat, sid)).encode())) % (2 ** 31)


def load_surface(data, cat, sid, n, seed):
    """2048 surface points, scaled like util/shapenet.py does for training."""
    with open(os.path.join(data, 'ShapeNetV2_point', cat, sid + '.npy'), 'rb') as f:
        scale = np.load(f).item()
    with np.load(os.path.join(data, 'ShapeNetV2_watertight', cat, '4_pointcloud', sid + '.npz')) as d:
        pts = d['points'].astype(np.float32) * scale
    ind = np.random.default_rng(shape_seed(seed, cat, sid)).choice(len(pts), n, replace=False)
    return pts[ind]


def load_volume(data, cat, sid):
    with np.load(os.path.join(data, 'ShapeNetV2_point', cat, sid + '.npz')) as d:
        return d['vol_points'].astype(np.float32), d['vol_label'].astype(np.float32)


def read_split(data, cat, split):
    with open(os.path.join(data, 'ShapeNetV2_point', cat, split + '.lst')) as f:
        return [l.strip().replace('.npz', '') for l in f if l.strip()]


@torch.no_grad()
def encode_posterior(model, pc):
    """KLAutoEncoder.encode up to the posterior: returns (mean, std, fps points), each B x M x ..."""
    B, N, D = pc.shape
    assert N == model.num_inputs, 'the model expects %d input points' % model.num_inputs
    flat = pc.reshape(B * N, D)
    batch = torch.repeat_interleave(torch.arange(B, device=pc.device), N)
    idx = models_ae.fps(flat, batch, ratio=1.0 * model.num_latents / model.num_inputs)
    sampled = flat[idx].view(B, -1, 3)
    cross_attn, cross_ff = model.cross_attend_blocks
    s_emb, pc_emb = model.point_embed(sampled), model.point_embed(pc)
    x = cross_attn(s_emb, context=pc_emb, mask=None) + s_emb
    x = cross_ff(x) + x
    post = models_ae.DiagonalGaussianDistribution(model.mean_fc(x), model.logvar_fc(x))
    return post.mean, post.std, sampled


@torch.no_grad()
def recon_iou(model, latents, points, labels, device, chunk=25000):
    """IoU of decode(posterior mean) on the stored volume points, as in eval.py."""
    q = torch.from_numpy(points)[None].to(device)
    logits = torch.cat([model.decode(latents, q[:, i:i + chunk]).squeeze(-1)
                        for i in range(0, q.shape[1], chunk)], dim=1)[0].cpu()
    pred = (logits >= 0).float()
    lab = torch.from_numpy(labels)
    return float((pred * lab).sum() / ((pred + lab).gt(0).sum() + 1e-5))


def set_distance(a, b):
    """Mean Euclidean distance under the optimal one-to-one matching of two equal-size point sets."""
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def encode_shape(model, data, cat, sid, args, device):
    pc = load_surface(data, cat, sid, model.num_inputs, args.seed)
    torch.manual_seed(shape_seed(args.seed, cat, sid))      # fps random start
    mean, std, sampled = encode_posterior(model, torch.from_numpy(pc)[None].to(device))
    return pc, mean, std, sampled


def pick(ids_arg, pool, k, rng, cat):
    """The given ids, or k random ones from the split."""
    if ids_arg:
        ids = [s.strip() for s in ids_arg.split(',') if s.strip()]
        missing = [s for s in ids if s not in pool]
        if missing:
            print('WARNING: %s: %s not in the split list' % (cat, missing))
        return ids
    return list(rng.choice(pool, k, replace=False))


def statistics(model, args, device, plotted):
    """Within- vs between-class latent-set distances over --n_stats shapes per class."""
    rng = np.random.default_rng(args.seed + 1)
    names, cats, sets = [], [], []
    for name, cat, _ in CLASSES:
        pool = read_split(args.data_path, cat, args.split)
        ids = list(rng.choice(pool, min(args.n_stats, len(pool)), replace=False))
        print('statistics: encoding %d %ss' % (len(ids), name), flush=True)
        for sid in ids:
            _, mean, _, _ = encode_shape(model, args.data_path, cat, sid, args, device)
            names.append(name)
            cats.append(cat)
            sets.append(mean[0].float().cpu().numpy())
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = set_distance(sets[i], sets[j])
    names = np.array(names)
    iu = np.triu_indices(n, 1)
    same = names[iu[0]] == names[iu[1]]
    groups = {}
    for cname, _, _ in CLASSES:
        m = (names[iu[0]] == cname) & (names[iu[1]] == cname)
        groups['within_' + cname] = D[iu][m]
    groups['between'] = D[iu][~same]
    Dn = D + np.diag(np.full(n, np.inf))
    nn_same = float((names[Dn.argmin(1)] == names).mean())

    def pct(vals, x):
        return float((vals < x).mean() * 100)

    out = {'n_per_class': args.n_stats, 'split': args.split,
           'nearest_neighbour_same_class': nn_same}
    for k, v in groups.items():
        out[k] = {'mean': float(v.mean()), 'median': float(np.median(v)),
                  'p5': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)), 'n_pairs': int(len(v))}
    within_all = np.concatenate([groups['within_' + c[0]] for c in CLASSES])
    # AUC: probability that a random within-class pair is closer than a random between-class pair
    w, b = within_all[:, None], groups['between'][None, :]
    out['auc_within_closer_than_between'] = float((w < b).mean() + 0.5 * (w == b).mean())
    # Where the plotted pairs fall (percentile within each distribution)
    out['plotted_pairs_percentile'] = {}
    for (la, lb), d in plotted.items():
        key = 'within_' + la.split()[0] if la.split()[0] == lb.split()[0] else 'between'
        out['plotted_pairs_percentile']['%s | %s' % (la, lb)] = {
            'distance': d, 'compared_with': key,
            'percentile': pct(groups[key], d)}
    return out


def make_figure(shapes, D, labels, stats, args, model_name, trained):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': AXIS, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
    })
    fig = plt.figure(figsize=(12, 8.4))
    gs = GridSpec(2, 4, height_ratios=[0.85, 2.3], hspace=0.12, wspace=0.35, figure=fig,
                  left=0.06, right=0.97, top=0.84, bottom=0.08)

    # Top row: the input point clouds, in each shape's tone. ShapeNet is y-up.
    for k, s in enumerate(shapes):
        ax = fig.add_subplot(gs[0, k], projection='3d')
        p = s['pc']
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=0.6, c=s['color'], linewidths=0, depthshade=False)
        c, lim = (p.max(0) + p.min(0)) / 2, (p.max(0) - p.min(0)).max() / 2
        ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[2] - lim, c[2] + lim); ax.set_zlim(c[1] - lim, c[1] + lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()
        ax.text2D(0.5, 1.0, '%s\n%s' % (s['label'], s['id']), transform=ax.transAxes, ha='center', va='bottom',
                  fontsize=8.5, color=INK2)

    # Bottom left: all latents in one scatter. Points are drawn in a shuffled order so no
    # shape is systematically hidden under another; the legend names every shape.
    ax = fig.add_subplot(gs[1, :3])
    xy = np.concatenate([s['mean'] for s in shapes])
    col = np.concatenate([[s['color']] * len(s['mean']) for s in shapes])
    order = np.random.default_rng(0).permutation(len(xy))
    ax.scatter(xy[order, 0], xy[order, 1], s=16, c=col[order], edgecolors=SURFACE, linewidths=0.5, zorder=2)
    for s in shapes:   # centroid of each latent set
        c = s['mean'].mean(0)
        ax.scatter([c[0]], [c[1]], s=110, c=s['color'], edgecolors=INK, linewidths=1.2, zorder=3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('latent channel 1 (posterior mean)')
    ax.set_ylabel('latent channel 2 (posterior mean)')
    handles = [Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor=s['color'],
                      markeredgecolor=SURFACE, label='%s  (%s)' % (s['label'], s['id'][:10]))
               for s in shapes]
    handles.append(Line2D([], [], marker='o', linestyle='', markersize=9, markerfacecolor='none',
                          markeredgecolor=INK, markeredgewidth=1.2, label='centroid of a shape\'s latents'))
    ax.legend(handles=handles, loc='best', frameon=True, facecolor=SURFACE, edgecolor=GRID,
              fontsize=8.5, labelcolor=INK)
    M = len(shapes[0]['mean'])
    ax.set_title('%d latents per shape, each a point in the 2D latent space' % M,
                 loc='left', fontsize=9.5, color=INK2)

    # Bottom right: distance matrix, achromatic ramp (blue and orange already mean classes).
    ax2 = fig.add_subplot(gs[1, 3])
    n = len(shapes)
    off = D[~np.eye(n, dtype=bool)]
    im = ax2.imshow(D, cmap='Greys', vmin=0, vmax=off.max() * 1.1)
    for i in range(n):
        for j in range(n):
            v = D[i, j]
            dark = im.norm(v) > 0.55
            ax2.text(j, i, '%.3f' % v, ha='center', va='center', fontsize=8.5,
                     color='#ffffff' if dark else INK)
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, rotation=45, ha='right', color=INK2)
    ax2.set_yticklabels(labels, color=INK2)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title('latent-set distance\n(optimal 1-to-1 matching)', loc='left', fontsize=9, color=INK2)

    title = 'C₀ = 2 latent space (%s, %s): two chairs and two airplanes' % (model_name, trained)
    fig.text(0.06, 0.975, title, fontsize=12.5, color=INK, weight='bold', va='bottom')
    if stats:
        w = ', '.join('within %ss %.3f' % (c, stats['within_' + c]['mean']) for c, _, _ in CLASSES)
        sub = ('Over %d %s shapes per class, mean latent-set distance: %s, between classes %.3f.\n'
               'The nearest neighbour in latent space has the same class for %.0f%% of shapes.'
               % (stats['n_per_class'], stats['split'], w, stats['between']['mean'],
                  100 * stats['nearest_neighbour_same_class']))
    else:
        sub = 'Two shapes per class only; run with --n_stats > 0 for within/between-class statistics.'
    fig.text(0.06, 0.955, sub, fontsize=9, color=INK2, va='top', linespacing=1.5)

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(args.out, 'latent_pairs.' + ext), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='kl_d512_m256_l2_d8', help='a KL model with latent_dim 2')
    p.add_argument('--pth', required=True,
                   help='checkpoint of the trained autoencoder, or "random" for an untrained encoder '
                        '(baseline: how much of the class separation comes from the input geometry alone)')
    p.add_argument('--data_path', required=True, help='folder with ShapeNetV2_point and ShapeNetV2_watertight')
    p.add_argument('--out', default='output/analysis/latent_pairs')
    p.add_argument('--split', default='test', help='split to draw shapes from (test: not seen in training)')
    p.add_argument('--chairs', default='', help='two chair ids (comma-separated); default: random from --split')
    p.add_argument('--airplanes', default='', help='two airplane ids (comma-separated); default: random from --split')
    p.add_argument('--n_stats', type=int, default=50,
                   help='shapes per class for the within/between statistics (0: skip)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    args.data_path = os.path.expanduser(args.data_path)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    model = models_ae.__dict__[args.model]()
    if not isinstance(model, models_ae.KLAutoEncoder) or model.mean_fc.out_features != 2:
        raise ValueError('%s is not a KL autoencoder with C0 = 2' % args.model)
    if args.pth == 'random':
        torch.manual_seed(args.seed)
        model = models_ae.__dict__[args.model]()   # fresh weights from the seed
        state, trained = {'epoch': None}, 'untrained, random weights'
    else:
        state = torch.load(args.pth, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'] if 'model' in state else state, strict=True)
        trained = 'epoch %s' % state.get('epoch', '?')
    model.to(device).eval()
    print('%s: %s (%s)' % (args.model, args.pth, trained), flush=True)

    rng = np.random.default_rng(args.seed)
    shapes = []
    for (name, cat, tones), ids_arg in zip(CLASSES, (args.chairs, args.airplanes)):
        pool = read_split(args.data_path, cat, args.split)
        ids = pick(ids_arg, pool, 2, rng, cat)
        if len(ids) != 2:
            raise ValueError('need exactly two %s ids, got %s' % (name, ids))
        for k, sid in enumerate(ids):
            pc, mean, std, sampled = encode_shape(model, args.data_path, cat, sid, args, device)
            vp, vl = load_volume(args.data_path, cat, sid)
            iou = recon_iou(model, mean, vp, vl, device)
            shapes.append({'label': '%s %s' % (name, 'AB'[k]), 'class': name, 'category': cat, 'id': sid,
                           'color': tones[k], 'pc': pc, 'iou': iou,
                           'mean': mean[0].float().cpu().numpy(), 'std': std[0].float().cpu().numpy(),
                           'fps': sampled[0].float().cpu().numpy()})
            print('%-10s %s  reconstruction IoU %.3f  mean posterior std per channel %s'
                  % (shapes[-1]['label'], sid, iou, np.round(shapes[-1]['std'].mean(0), 3)), flush=True)

    labels = [s['label'] for s in shapes]
    n = len(shapes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = set_distance(shapes[i]['mean'], shapes[j]['mean'])
    print('latent-set distances:')
    for i in range(n):
        print('  %-10s ' % labels[i] + '  '.join('%.3f' % v for v in D[i]))
    plotted = {(labels[i], labels[j]): float(D[i, j]) for i in range(n) for j in range(i + 1, n)}

    stats = statistics(model, args, device, plotted) if args.n_stats > 0 else None
    if stats:
        print(json.dumps({k: v for k, v in stats.items() if k != 'plotted_pairs_percentile'}, indent=1))
        for k, v in stats['plotted_pairs_percentile'].items():
            print('  %-24s %.3f  = percentile %.0f of %s distances' % (k, v['distance'], v['percentile'], v['compared_with']))
        low = [s for s in shapes if s['iou'] < 0.5]
        if low and args.pth != 'random':
            print('WARNING: reconstruction IoU below 0.5 for %s: the autoencoder may not be trained enough '
                  'for its latents to mean much.' % [s['label'] for s in low])

    make_figure(shapes, D, labels, stats, args, args.model, trained)

    with open(os.path.join(args.out, 'latents.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['shape', 'class', 'category', 'id', 'latent', 'mean_c1', 'mean_c2', 'std_c1', 'std_c2',
                    'fps_x', 'fps_y', 'fps_z'])
        for s in shapes:
            for i in range(len(s['mean'])):
                w.writerow([s['label'], s['class'], s['category'], s['id'], i,
                            *('%.6f' % v for v in (*s['mean'][i], *s['std'][i], *s['fps'][i]))])
    np.savez(os.path.join(args.out, 'latent_pairs.npz'), labels=np.array(labels),
             ids=np.array([s['id'] for s in shapes]), distance=D,
             **{'%s_%s' % (s['label'].replace(' ', '_'), k): s[k] for s in shapes for k in ('mean', 'std', 'fps', 'pc')})
    summary = {'model': args.model, 'checkpoint': args.pth, 'epoch': state.get('epoch'), 'split': args.split,
               'seed': args.seed,
               'shapes': [{k: s[k] for k in ('label', 'class', 'category', 'id', 'color', 'iou')} for s in shapes],
               'distance_matrix': {'labels': labels, 'values': D.round(6).tolist()},
               'statistics': stats}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('wrote', ', '.join(os.path.join(args.out, x) for x in
                             ('latent_pairs.png', 'latent_pairs.pdf', 'latents.csv', 'latent_pairs.npz', 'summary.json')))


if __name__ == '__main__':
    main()
