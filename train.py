#!/usr/bin/env python3
"""Training and Evaluate the Neural Netowrk
Usage:
    train.py [options] [<yaml-config>]
    train.py (-h | --help )

Options:
   -h --help                         Show this screen.
   -d --devices <devices>            Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>      Folder name
   --from <checkpoint>               Load a checkpoint before evaluation
   --eval                            Evaluate instead of training
"""

import os
import sys
import shlex
import random
import os.path as osp
import datetime
import platform
import threading
import subprocess
import pprint

import yaml
import numpy as np
import torch
from torch import nn
from docopt import docopt

import wireframe
from wireframe.datasets import WireframeDataset
from wireframe.models.meta_builder import MetaBuilder
from wireframe.models.resnet_baseline import ResNetUperNet
from wireframe.models.multitask_learner import MultitaskLearner


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def get_outdir(c, identifier=None):
    # load config
    logdir, model_name = c["io"]["logdir"], c["name"]
    name = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    name += "-%s" % platform.uname()[1]
    name += "-%s" % git_hash()
    name += "-%s" % model_name
    if identifier:
        name += "-%s" % identifier
    # create out
    outdir = osp.abspath(osp.join(osp.expanduser(logdir), name))
    if not osp.exists(outdir):
        os.makedirs(outdir)
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    with open(osp.join(outdir, "config.yaml"), "w") as f:
        c["io"]["logdir"] = osp.abspath(c["io"]["logdir"])
        c["io"]["datadir"] = osp.abspath(c["io"]["datadir"])
        c["io"]["resume_from"] = outdir
        yaml.safe_dump(c, f, default_flow_style=False)
    return outdir


here = osp.dirname(osp.abspath(__file__))


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/hourglass.yaml"
    with open(config_file, "r") as f:
        c = yaml.load(f)
        pprint.pprint(c, indent=4)
    resume_from = args["--from"]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    datadir = c["io"]["datadir"]

    # uncomment for debug DataLoader
    # wireframe.datasets.WireframeDataset(root, split="train")[0]
    # sys.exit(0)

    if c["model"]["name"] == "upernet":
        image_std = (1, 1, 1)
    else:
        image_std = (22.275, 22.124, 23.229)

    kwargs = {"pin_memory": True, "num_workers": c["io"]["num_workers"]}
    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="train", image_std=image_std),
        batch_size=c["model"]["batch_size"],
        shuffle=True,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="valid", image_std=image_std),
        batch_size=c["model"]["batch_size"],
        shuffle=False,
        **kwargs,
    )
    epoch_size = len(train_loader)
    # print("epoch_size (train):", epoch_size)
    # print("epoch_size (valid):", len(val_loader))

    if resume_from:
        checkpoint = torch.load(resume_from)

    # 2. model
    num_class = MultitaskLearner.NUM_CLASS
    if c["model"]["name"] == "stacked_hourglass":
        model = wireframe.models.hg(
            depth=c["model"]["depth"],
            num_stacks=c["model"]["num_stacks"],
            num_blocks=c["model"]["num_blocks"],
            num_classes=num_class,
        )
    elif c["model"]["name"] == "upernet":
        model_builder = MetaBuilder()
        model_encoder = model_builder.build_encoder()
        model_decoder = model_builder.build_decoder(num_class=num_class)
        model = ResNetUperNet(model_encoder, model_decoder)
    else:
        raise NotImplementedError

    model = MultitaskLearner(model)
    model.num_stacks = 1

    model = model.to(device)
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    model.num_stacks = model.module.num_stacks
    if resume_from:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 3. optimizer
    if c["optim"]["name"] == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=c["optim"]["lr"],
            weight_decay=c["optim"]["weight_decay"],
            amsgrad=c["optim"]["amsgrad"],
        )
    elif c["optim"]["name"] == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=c["optim"]["lr"],
            weight_decay=c["optim"]["weight_decay"],
            momentum=c["optim"]["momentum"],
        )
    else:
        raise NotImplementedError

    if resume_from:
        optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = get_outdir(c, args["--identifier"])
    print("outdir:", outdir)

    trainer = wireframe.trainer.Trainer(
        device=device,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=outdir,
        max_iter=c["optim"]["max_iteration"],
        batch_size=c["model"]["batch_size"],
        checkpoint_interval=c["io"]["checkpoint_interval"],
        validation_interval=c["io"]["validation_interval"],
    )
    if resume_from:
        trainer.iteration = checkpoint["iteration"]
        if trainer.iteration % epoch_size != 0:
            print("WARNING: iteration is not a multiple of epoch_size, reset it")
            trainer.iteration -= trainer.iteration % epoch_size
        trainer.best_mean_loss = checkpoint["best_mean_loss"]
        del checkpoint

    if args["--eval"]:
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
