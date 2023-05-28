import copy

from torch.nn.init import kaiming_normal_
from datasets import cub
from types import SimpleNamespace
import hydra
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import glob
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from models.model_factory import model_generator

from datasets.lit_dataset import LitDataset
from utils.utils import seed_worker
import resource
print("Inference")


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    print(os.getcwd())
    cfg = dict(cfg)
    cfg['dataset'] = 'CUB'

    wandb.init(project='', entity='', mode='disabled')
    wandb.config.update(cfg)
    args = wandb.config
    cudnn.enabled = True

    print("---------------------------------------")
    print(f"Arguments received: ")
    print("---------------------------------------")
    for k, v in sorted(args.items()):
        print(f"{k:25}: {v}")
    print("---------------------------------------")



    # This is theirs dataset
    # We need to be able to pass ours
    # if you don't have their dataset the code will break there
    args.split = 'train'
    train_dataset = cub.CUBDataset(args)
    args_test = SimpleNamespace(**copy.deepcopy(dict(args)))
    args_test.split = 'test'
    test_dataset = cub.CUBDataset(args_test)

    trainloader = DataLoader(
        LitDataset(train_dataset, args.use_lab),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        drop_last=True)

    trainloader_iter = enumerate(trainloader)

    testloader = DataLoader(
        LitDataset(test_dataset),
        batch_size=args_test.batch_size,
        shuffle=True,
        num_workers=args_test.num_workers,
        worker_init_fn=seed_worker,
        drop_last=False)

    testloader_iter = enumerate(testloader)


        
        
    model, optimizer_state_dict = model_generator(args, add_bg_mask=False)
    model = model.cuda()
 
    print(f"Model generated: {model.__class__.__name__}")

    # Try to do inference on the model
    # Try to use dataset that protopnet uses
    # Understand which output we need to use to combine with protopnet

if __name__ == '__main__':
    main()
