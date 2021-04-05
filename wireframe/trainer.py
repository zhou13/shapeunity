import os
import os.path as osp
import shutil
import threading
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io

import wireframe.utils as utils

plt.rcParams["figure.figsize"] = (24, 24)


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def _launch_tensorboard(board_out, port, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.system(f"tensorboard --logdir={board_out} --port={port} &> {out}/tb.log")
    return


class Trainer(object):
    def __init__(
        self,
        device,
        model,
        optimizer,
        train_loader,
        val_loader,
        out,
        max_iter,
        batch_size,
        checkpoint_interval,
        validation_interval,
    ):
        self.device = device

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

        self.validation_interval = validation_interval
        self.checkpoint_interval = checkpoint_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def _loss(self, losses):
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.model.num_stacks, len(self.loss_labels)])
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.model.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * self.batch_size:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * self.batch_size:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                input_dict = {"data": data, "target": target, "output_heatmap": True}
                result = self.model(input_dict)
                losses = result["losses"]

                total_loss += self._loss(losses)

                H = result["heatmaps"]
                for i in range(H["jmap"].shape[0]):
                    index = batch_idx * self.batch_size + i
                    np.savez(
                        f"{npz}/{index:06}.npz",
                        **{k: v[i].cpu().numpy() for k, v in H.items()},
                    )
                    if index < 12:
                        self._plot_samples(i, index, H, target, f"{viz}/{index:06}")

        self._write_metrics(len(self.val_loader), total_loss, "validation", True)
        self.mean_loss = total_loss / len(self.val_loader)
        if training:
            self.model.train()

    def add_checkpoint(self):
        if self.iteration > 0:
            torch.save(
                {
                    "iteration": self.iteration,
                    "arch": self.model.__class__.__name__,
                    "optim_state_dict": self.optim.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "best_mean_loss": self.best_mean_loss,
                },
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
            )
        if (self.epoch + 1) % self.checkpoint_interval == 0:
            shutil.copy(
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
                osp.join(self.out, f"checkpoint_{self.epoch+1:03}.pth.tar"),
            )

        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_lastest.pth.tar"),
                osp.join(self.out, "checkpoint_best.pth.tar"),
            )

    def train_epoch(self):
        self.model.train()

        time = timer()
        for batch_idx, (data, target) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0

            input_dict = {"data": data, "target": target, "output_heatmap": False}

            result = self.model(input_dict)
            losses = result["losses"]

            loss = self._loss(losses)

            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            self._write_metrics(1, loss.item(), "training", do_print=False)

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            num_images = self.batch_size * self.iteration
            if num_images % self.validation_interval == 0 or num_images == 600:
                self.validate()
                time = timer()

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        return total_loss

    def _plot_samples(self, i, index, result, target, prefix):
        plt.close()
        plt.imshow(io.imread(self.val_loader.dataset.filelist[index] + ".png"))
        plt.savefig(f"{prefix}_img.jpg"), plt.close()

        mask_result = result["jmap"][i].cpu()
        mask_target = target["jmap"][i].cpu()
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            plt.imshow(ia), plt.savefig(f"{prefix}_mask_{ch}a.jpg"), plt.close()
            plt.imshow(ib), plt.savefig(f"{prefix}_mask_{ch}b.jpg"), plt.close()

        line_result = result["lmap"][i].cpu()
        line_target = target["lmap"][i].cpu()
        plt.imshow(line_target), plt.savefig(f"{prefix}_line_a.jpg"), plt.close()
        plt.imshow(line_result), plt.savefig(f"{prefix}_line_b.jpg"), plt.close()

        angle_result = result["ldir"][i].cpu()
        angle_target = target["ldir"][i].cpu()
        utils.quiver(
            line_target * np.cos(angle_target),
            line_target * np.sin(angle_target),
            plt.gca(),
        ), plt.savefig(f"{prefix}_ldir_a.jpg"), plt.close()
        utils.quiver(
            line_result * np.cos(angle_result),
            line_result * np.sin(angle_result),
            plt.gca(),
        ), plt.savefig(f"{prefix}_ldir_b.jpg"), plt.close()

        depth_result = result["dpth"][i].cpu()
        depth_target = target["dpth"][i].cpu()
        plt.imshow(depth_target), plt.savefig(f"{prefix}_depth_a.jpg"), plt.close()
        plt.imshow(depth_result), plt.savefig(f"{prefix}_depth_b.jpg"), plt.close()

    def train(self):
        # if self.iteration == 0:
        #     self.validate()
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        max_epoch = (self.max_iter - 1) // epoch_size + 1
        for self.epoch in range(start_epoch, max_epoch):
            self.train_epoch()
            self.add_checkpoint()

    def test(self, fname):
        image = io.imread(fname).astype(float)[:, :, :3]
        image = (image - [109.730, 103.832, 98.681]) / [22.275, 22.124, 23.229]
        data = torch.from_numpy(np.rollaxis(image, 2)[np.newaxis, ...]).float()

        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            data = data.to(self.device)
            result = self.model(data)
            result = result[-1].cpu()
            np.savez(f"{self.out}/result.npz", result=result.numpy())

        if training:
            self.model.train()
