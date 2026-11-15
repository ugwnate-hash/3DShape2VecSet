import argparse
import datetime
import itertools
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(8)

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_shape_surface_occupancy_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_ae

from engine_ae import train_one_epoch, evaluate


class SkippingSampler(torch.utils.data.Sampler):
    """Wraps a sampler and skips its first `skip` indices.

    Used to resume in the middle of an epoch: the wrapped DistributedSampler gives the same
    order for the same epoch, so skipping the indices already trained on continues where
    the saved run stopped.
    """
    def __init__(self, base):
        self.base = base
        self.skip = 0

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        return itertools.islice(iter(self.base), self.skip, None)

    def __len__(self):
        return max(0, len(self.base) - self.skip)

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='ae_blob64', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--point_cloud_size', default=2048, type=int,
                        help='input size')
    parser.add_argument('--amp', action='store_true',
                        help='mixed precision training (fp16 autocast with loss scaling)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')


    # Dataset parameters
    parser.add_argument('--data_path', default='/ibex/scratch/projects/c2168/diffusion-shapes/datasets', type=str,
                        help='dataset path')
    parser.add_argument('--categories', default=None, type=str,
                        help='comma-separated ShapeNet synset ids to train and validate on, '
                             'e.g. 02691156,03001627 (airplane, chair). Default: all categories found')

    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='data-loader iteration to start the first epoch at (set by --resume from a mid-epoch checkpoint)')

    # Checkpointing and time limit
    parser.add_argument('--save_every_epochs', default=10, type=int,
                        help='save an end-of-epoch checkpoint every N epochs (the last epoch is always saved)')
    parser.add_argument('--save_every_steps', default=0, type=int,
                        help='also save a mid-epoch checkpoint every N optimizer steps; 0 disables. '
                             'Only the newest mid-epoch checkpoint is kept.')
    parser.add_argument('--max_hours', default=0, type=float,
                        help='stop training cleanly after this many hours, saving a checkpoint first; 0 disables')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_shape_surface_occupancy_dataset('train', args=args)
    dataset_val = build_shape_surface_occupancy_dataset('val', args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = SkippingSampler(torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        ))
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=args.batch_size,
        batch_size=1,
        # num_workers=args.num_workers,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    model = models_ae.__dict__[args.model](N=args.point_cloud_size)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp=args.amp)
        print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    first_epoch = args.start_epoch
    # Iterations in a full epoch for this process (the loader drops the last partial batch).
    iters_per_epoch = len(sampler_train.base) // args.batch_size

    def time_is_up():
        up = args.max_hours > 0 and time.time() - start_time > args.max_hours * 3600
        if args.distributed:
            # Every process must stop at the same step, so use the decision of the slowest clock.
            flag = torch.tensor([int(up)], device=device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            up = bool(flag.item())
        return up

    stop = False          # time limit reached
    stopped_mid = False   # ...and training stopped before the end of an epoch
    for epoch in range(args.start_epoch, args.epochs):
        # The train sampler is always a DistributedSampler, even on a single process.
        data_loader_train.sampler.set_epoch(epoch)
        start_iter = args.start_iter if epoch == first_epoch else 0
        data_loader_train.sampler.skip = start_iter * args.batch_size

        def on_step(next_iter, epoch=epoch):
            # Called after every optimizer step. Returns True to end training now.
            nonlocal stop, stopped_mid
            stop = time_is_up()
            steps_done = next_iter // args.accum_iter
            periodic = args.save_every_steps > 0 and steps_done % args.save_every_steps == 0
            if args.output_dir and (stop or periodic) and next_iter < iters_per_epoch:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, next_iter=next_iter)
            stopped_mid = stop and next_iter < iters_per_epoch
            return stopped_mid

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
            start_iter=start_iter,
            step_callback=on_step if (args.save_every_steps > 0 or args.max_hours > 0) else None,
        )
        if stopped_mid:
            print(f"Time limit of {args.max_hours} h reached in epoch {epoch}: saved a mid-epoch checkpoint, stopping.")
            break

        if args.output_dir and (epoch % args.save_every_epochs == 0 or epoch + 1 == args.epochs or stop):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            test_stats = evaluate(data_loader_val, model, device, amp=args.amp)
            print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
            max_iou = max(max_iou, test_stats["iou"])
            print(f'Max iou: {max_iou:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if stop:
            print(f"Time limit of {args.max_hours} h reached at the end of epoch {epoch}: stopping.")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)