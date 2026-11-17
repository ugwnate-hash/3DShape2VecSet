"""Build a small training dataset for chosen ShapeNet categories, straight from the authors' zips.

The preprocessed data is two zips on Google Drive (see the main README): occupancies.zip
(589 GB) and surfaces.zip (56 GB). Each shape stores 500k volume points, 500k near-surface
points and 100k surface points, while training draws only 1024 + 1024 query points and 2048
surface points per sample. This script:

  1. reads only the index of each zip, using HTTP range requests,
  2. streams just the chosen categories (each is one contiguous block of the zip),
  3. keeps a random subset of points per shape (50k volume, 50k near-surface, 30k surface by default),
  4. writes the layout util/shapenet.py expects:
       OUT/ShapeNetV2_point/<cat>/{<shape>.npz, <shape>.npy, train.lst, val.lst, test.lst}
       OUT/ShapeNetV2_watertight/<cat>/4_pointcloud/<shape>.npz

For airplane + chair it downloads about 133 GB and writes about 16 GB. It's meant for a Kaggle
CPU notebook with Internet on (see kaggle/KAGGLE.md), but runs anywhere with Python and NumPy.

Google Drive limits how much a public file can be downloaded per day ("Quota exceeded"),
counting every user. When that happens the script moves on to the next block, keeps
everything it has written, and exits with code 3 ("incomplete"). With --retry_for_hours it
keeps trying every --retry_every_min minutes instead. Run it again later with --previous
pointing at the earlier output: finished files are not downloaded again. They are copied over
at the end, so the downloads start right away while Drive allows them.

    python kaggle/prepare_subset.py --out /kaggle/working/shapenet_subset --categories 02691156,03001627 \
        [--previous /kaggle/input/<earlier-output>/shapenet_subset]
"""
import argparse
import concurrent.futures
import io
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib

import numpy as np

ZIPS = {   # Google Drive file ids, from the README's data link
    'occupancies': '1dbFfd0YSNItQi6eVzCOLo1m321aM52id',
    'surfaces': '1Rg_Uyl7DNdX6uKPeQ8g4gm9od6bCWUa5',
}


def drive_url(file_id):
    return 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t' % file_id


class DriveRefused(RuntimeError):
    """Google Drive answered with a page instead of the file (usually its daily download quota)."""


def _refusal(status, body):
    title = body.split('<title>')[1].split('</title>')[0] if '<title>' in body else body[:200]
    return DriveRefused('Google Drive refused the download (HTTP %d: %s). Its daily download quota for this '
                        'file is probably used up; try again in about 24 hours.' % (status, title.strip()))


def open_range(url, start, end=None, retries=8):
    """GET bytes [start, end] (inclusive), retrying network errors with backoff."""
    rng = 'bytes=%d-%s' % (start, '' if end is None else end)
    for attempt in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': rng}), timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):   # quota or rate limit: retrying now won't help
                raise _refusal(e.code, e.read(5000).decode('utf-8', 'replace'))
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        else:
            if r.status != 206:
                raise _refusal(r.status, r.read(5000).decode('utf-8', 'replace'))
            return r
        wait = min(300, 5 * 2 ** attempt)
        print('  request failed (%s), retrying in %d s' % (err, wait), flush=True)
        time.sleep(wait)
    raise RuntimeError('giving up after %d failed requests' % retries)


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (used to read the zip index)."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        with open_range(url, 0, 0) as r:
            self.size = int(r.headers['Content-Range'].split('/')[1])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b''
        with open_range(self.url, self.pos, min(self.pos + n, self.size) - 1) as r:
            data = r.read()
        self.pos += len(data)
        return data


class Stream:
    """Sequential reader over one long range request, reconnecting where it left off."""
    def __init__(self, url, start, end):
        self.url, self.pos, self.end = url, start, end
        self.resp = None
        self.bytes = 0

    def read(self, n):
        out = bytearray()
        while len(out) < n:
            if self.resp is None:
                self.resp = open_range(self.url, self.pos, self.end)
            try:
                chunk = self.resp.read(min(n - len(out), 1 << 22))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                chunk, err = b'', e
            else:
                err = None
            if not chunk:
                print('  connection dropped at byte %d (%s), reconnecting' % (self.pos, err or 'closed'), flush=True)
                self.resp.close()
                self.resp = None
                time.sleep(5)
                continue
            out += chunk
            self.pos += len(chunk)
            self.bytes += len(chunk)
        return bytes(out)

    def skip(self, n):
        if n > 0:
            self.read(n)


def member_data(stream, info):
    """Read one zip member at the stream's position (must be its local header) and decompress it."""
    stream.skip(info.header_offset - stream.pos)
    header = stream.read(30)
    if header[:4] != b'PK\x03\x04':
        raise RuntimeError('bad local header for %s' % info.filename)
    name_len, extra_len = struct.unpack('<HH', header[26:30])
    stream.skip(name_len + extra_len)
    raw = stream.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise RuntimeError('unsupported compression %d for %s' % (info.compress_type, info.filename))


def save_npz(path, **arrays):
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)            # uncompressed: fast to load during training
    os.replace(tmp, path)


def shape_rng(name):
    return np.random.default_rng(zlib.crc32(name.encode()))   # same subset on every run


def process_occupancy(data, name, out_path, n_vol, n_near):
    with np.load(io.BytesIO(data)) as d:
        vol, vl, near, nl = d['vol_points'], d['vol_label'], d['near_points'], d['near_label']
    iv = np.sort(shape_rng(name + ':vol').choice(len(vol), min(n_vol, len(vol)), replace=False))
    inr = np.sort(shape_rng(name + ':near').choice(len(near), min(n_near, len(near)), replace=False))
    save_npz(out_path, vol_points=vol[iv], vol_label=vl[iv], near_points=near[inr], near_label=nl[inr])


def process_surface(data, name, out_path, n_surf):
    with np.load(io.BytesIO(data)) as d:
        pts = d['points']
    i = np.sort(shape_rng(name + ':surf').choice(len(pts), min(n_surf, len(pts)), replace=False))
    save_npz(out_path, points=pts[i])


def output_path(out, zip_name, filename):
    """Map a zip member to its output path, or None to ignore it."""
    parts = filename.split('/')
    if zip_name == 'occupancies':      # ShapeNetV2_point/<cat>/<file>
        if len(parts) != 3 or not parts[2]:
            return None
        return os.path.join(out, *parts)
    if len(parts) != 3 or parts[1] != '4_pointcloud' or not parts[2]:   # <cat>/4_pointcloud/<file>
        return None
    return os.path.join(out, 'ShapeNetV2_watertight', *parts)


def category_of(zip_name, filename):
    parts = filename.split('/')
    return parts[1] if zip_name == 'occupancies' and len(parts) > 1 else parts[0]


def run_zip(zip_name, categories, args):
    """Process every category of one zip. Returns the categories left incomplete."""
    url = drive_url(ZIPS[zip_name])
    print('== %s: reading the index' % zip_name, flush=True)
    try:
        infos = zipfile.ZipFile(RemoteFile(url)).infolist()
    except DriveRefused as e:
        print('!! %s: %s' % (zip_name, e), flush=True)
        return list(categories)
    incomplete = []
    for cat in categories:
        members = sorted((i for i in infos if category_of(zip_name, i.filename) == cat
                          and output_path(args.out, zip_name, i.filename)), key=lambda i: i.header_offset)
        if not members:
            raise RuntimeError('category %s not found in %s' % (cat, zip_name))
        if args.limit:
            members = members[:args.limit]
        todo = [m for m in members if not is_done(output_path(args.out, zip_name, m.filename), args)]
        total = sum(m.compress_size for m in todo)
        print('-- %s / %s: %d files, %d left to do (%.1f GB to download)'
              % (zip_name, cat, len(members), len(todo), total / 1e9), flush=True)
        if not todo:
            continue
        try:
            fetch(url, zip_name, todo, total, args)
        except DriveRefused as e:
            print('!! %s / %s stopped: %s Files written so far are kept.' % (zip_name, cat, e), flush=True)
            incomplete.append(cat)
    return incomplete


def fetch(url, zip_name, todo, total, args):
    """Stream the zip members in `todo` (sorted by offset) and write their outputs."""
    last = todo[-1]
    stream = Stream(url, todo[0].header_offset, last.header_offset + 30 + 65535 * 2 + last.compress_size)
    try:
        _fetch(stream, zip_name, todo, total, args)
    finally:
        if stream.resp is not None:
            stream.resp.close()


def _fetch(stream, zip_name, todo, total, args):
    t0, done = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = []
        for m in todo:
            data = member_data(stream, m)
            path = output_path(args.out, zip_name, m.filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if m.filename.endswith('.npz'):
                if zip_name == 'occupancies':
                    pending.append(pool.submit(process_occupancy, data, m.filename, path, args.vol_points, args.near_points))
                else:
                    pending.append(pool.submit(process_surface, data, m.filename, path, args.surface_points))
            else:                      # .npy scale files and .lst split files: copy as-is
                with open(path + '.tmp', 'wb') as f:
                    f.write(data)
                os.replace(path + '.tmp', path)
            # bounded queue: keep at most a few shapes in memory
            while len(pending) > 2 * args.workers:
                pending.pop(0).result()
            done += 1
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                rate = stream.bytes / max(el, 1e-9)
                print('   %d/%d files | %.1f GB | %.0f MB/s | about %.0f min left'
                      % (done, len(todo), stream.bytes / 1e9, rate / 1e6,
                         (total - stream.bytes) / max(rate, 1) / 60), flush=True)
        for p in pending:
            p.result()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--categories', required=True, help='comma-separated ShapeNet synset ids, e.g. 02691156,03001627')
    p.add_argument('--vol_points', type=int, default=50000, help='volume points kept per shape (of 500k)')
    p.add_argument('--near_points', type=int, default=50000, help='near-surface points kept per shape (of 500k)')
    p.add_argument('--surface_points', type=int, default=30000, help='surface points kept per shape (of 100k)')
    p.add_argument('--workers', type=int, default=3, help='threads that subsample and write while downloading')
    p.add_argument('--limit', type=int, default=0, help='only the first N files per category and zip (for testing)')
    p.add_argument('--previous', action='append', default=[],
                   help='output folder of an earlier run (can be given several times): its finished files '
                        'are not downloaded again, and are copied into --out at the end')
    p.add_argument('--retry_for_hours', type=float, default=0,
                   help='if Google Drive refuses, keep retrying for up to this many hours (0: try once)')
    p.add_argument('--retry_every_min', type=float, default=30, help='minutes between retries')
    args = p.parse_args()
    categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    for prev in args.previous:
        if not os.path.isdir(prev):
            print('WARNING: --previous %s does not exist; ignored' % prev, flush=True)
    args.previous = [prev for prev in args.previous if os.path.isdir(prev)]
    t = time.time()
    deadline = t + args.retry_for_hours * 3600
    # surfaces.zip first: it is smaller, and it is a separate file with its own download quota
    remaining = {'surfaces': list(categories), 'occupancies': list(categories)}
    attempt = 0
    while True:
        attempt += 1
        for zip_name in list(remaining):
            remaining[zip_name] = run_zip(zip_name, remaining[zip_name], args)
            if not remaining[zip_name]:
                del remaining[zip_name]
        if not remaining or time.time() + args.retry_every_min * 60 > deadline:
            break
        print('[attempt %d] still refused: %s. Retrying in %.0f min (until %.1f h after start).'
              % (attempt, ', '.join('%s/%s' % (z, c) for z, cs in remaining.items() for c in cs),
                 args.retry_every_min, args.retry_for_hours), flush=True)
        time.sleep(args.retry_every_min * 60)
    incomplete = [(z, c) for z, cs in remaining.items() for c in cs]
    for prev in args.previous:
        copy_previous(prev, args.out, categories)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs)
    print('finished in %.0f min: %.1f GB in %s' % ((time.time() - t) / 60, size / 1e9, args.out))
    if incomplete:
        print('INCOMPLETE: %s. Google Drive refused part of the download. Run again in about 24 hours '
              'with --previous %s (on Kaggle: add this run\'s output as an input).'
              % (', '.join('%s/%s' % z for z in incomplete), args.out))
        sys.exit(3)
    print('COMPLETE')


def is_done(path, args):
    """True if this output file exists in --out or in one of the --previous folders."""
    if os.path.exists(path):
        return True
    rel = os.path.relpath(path, args.out)
    return any(os.path.exists(os.path.join(prev, rel)) for prev in args.previous)


def copy_previous(prev, out, categories):
    """Copy finished files of the chosen categories from an earlier run's output."""
    if not os.path.isdir(prev):
        print('WARNING: --previous %s does not exist; nothing copied' % prev, flush=True)
        return
    n = 0
    for sub in ('ShapeNetV2_point', 'ShapeNetV2_watertight'):
        for cat in categories:
            src_root = os.path.join(prev, sub, cat)
            for root, _, files in os.walk(src_root):
                for f in files:
                    if '.tmp' in f:         # half-written file from an interrupted run
                        continue
                    src = os.path.join(root, f)
                    dst = os.path.join(out, os.path.relpath(src, prev))
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copyfile(src, dst)
                        n += 1
    print('copied %d finished files from %s' % (n, prev), flush=True)


if __name__ == '__main__':
    main()
