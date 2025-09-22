import os
import argparse
# from mmcv.utils import Config, DictAction
from mmengine.config import Config, DictAction

from utils.utils import *
from experiments.ddp import *
from experiments.runner import *

import torch
torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser()
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local-rank", type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--use_slurm', default=False, action='store_true')

    # exp setting
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='overwrite config param.')
    return parser.parse_args()

    # args.local_rank = int(os.environ.get('LOCAL_RANK', 0))


def main():
    args = get_args()

    # define runner to begin training or evaluation
    cfg = Config.fromfile(args.config)
    cfg.top_view_region = np.array(cfg.top_view_region)
    cfg.K = np.array(cfg.K)
    if cfg.dataset_name == 'apollo':
        cfg.anchor_y_steps = np.linspace(3, 103, 20)
    elif cfg.dataset_name == 'once':
        cfg.anchor_y_steps = np.linspace(0.5, 65, 20)
    else:
        raise ValueError('dataset_name not recognized; add the new dataset in your config')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # os.environ['LOCAL_RANK'] = str(0)
    # initialize distributed data parallel set
    ddp_init(args)
    cfg.merge_from_dict(vars(args))
    runner = Runner(cfg)
    if not cfg.evaluate:
        runner.train()
    else:
        runner.eval()


if __name__ == '__main__':
    main()