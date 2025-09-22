import torch
import time
from mmengine.config import Config as MMConfig
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.latr import LATR
from data.apollo_dataset import ApolloLaneDataset
from torch.utils.data import DataLoader
import argparse
import math

parser = argparse.ArgumentParser(description='FPS calculator for LATR')
parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint .pth.tar')
parser.add_argument('--config', type=str, required=True, help='Path to config .py')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Do not use CUDA (override config, default: False)')
parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to measure')
args = parser.parse_args()

cfg = MMConfig.fromfile(args.config)
cfg.top_view_region = np.array(cfg.top_view_region)
if hasattr(cfg, 'K'):
    cfg.K = np.array(cfg.K)
cfg.anchor_y_steps = np.linspace(3, 103, 20)
if not hasattr(cfg, 'anchor_y_steps_dense'):
    cfg.anchor_y_steps_dense = np.linspace(3, 103, 200)
cfg.merge_from_dict(vars(args))

class ArgsForDataset(dict):
    def __getattr__(self, k):
        return self[k] if k in self else None
args_ds = ArgsForDataset(cfg)
for k, v in [
    ('mean', None), ('std', None), ('photo_aug', None), ('dataset_name', 'apollo'),
    ('num_category', getattr(cfg, 'num_category', 1)), ('org_h', getattr(cfg, 'resize_h', 720)),
    ('org_w', getattr(cfg, 'resize_w', 960)), ('crop_y', 0), ('ipm_h', getattr(cfg, 'resize_h', 720)),
    ('ipm_w', getattr(cfg, 'resize_w', 960)), ('max_lanes', getattr(cfg, 'max_lanes', 6)),
    ('K', getattr(cfg, 'K', np.eye(3))), ('proc_id', 0), ('batch_size', 1), ('nworkers', 0), ('dist', False)
]:
    if k not in args_ds:
        args_ds[k] = v

dataset_base_dir = cfg.dataset_dir
json_file_path = os.path.join(cfg.data_dir, 'test.json')
val_dataset = ApolloLaneDataset(dataset_base_dir, json_file_path, args_ds, data_aug=False)

batch_size = 2

dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = LATR(cfg)
if not cfg.no_cuda and args.cuda:
    model = model.cuda()
model.eval()
for m in model.modules():
    m.eval()

max_batches = math.ceil(len(val_dataset) / dataloader.batch_size)
num_batches = max_batches
print(f'Max batches: {num_batches}')

warmup = 5

total_time = 0
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if i >= num_batches + warmup:
            break
        input_tensor = batch['image'] if 'image' in batch else batch[list(batch.keys())[0]]
        extra_dict = batch
        if args.cuda:
            input_tensor = input_tensor.cuda()
            for k in extra_dict:
                if isinstance(extra_dict[k], torch.Tensor):
                    extra_dict[k] = extra_dict[k].cuda()
        start = time.time()
        _ = model(input_tensor, None, False, extra_dict)
        if args.cuda:
            torch.cuda.synchronize()
        end = time.time()
        if i >= warmup:
            total_time += (end - start)

fps = num_batches / total_time if total_time > 0 else 0
print(f'FPS: {fps:.2f} (batch_size={batch_size}, {num_batches} batches, exclude {warmup} warmup)')
