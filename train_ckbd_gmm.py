import argparse
import math
import random
import sys
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim
import glob

from compressai.models.ckbd_gmm import Cheng2020AnchorCheckerboardGMM
from compressai.models.sensetime import Cheng2020AnchorCheckerboard
from torch.utils.tensorboard import SummaryWriter   
import os
import torch.nn.functional as F
import gc
#from calflops import calculate_flops

#torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
#torch.backends.cudnn.enabled = False

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()
def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )
def test_kodak(kodak_path, net_, args):
    #collect images under kodak_path
    p = 128
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    device =  next(net_.parameters()).device
    img_list = []
    for file in os.listdir(kodak_path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    net = net_.eval()
    for img_name in img_list:
        img_path = os.path.join(kodak_path, img_name)
        img = Image.open(img_path).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        x_padded, padding = pad(x, p)
        count += 1
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            s = time.time()
            out_net = net.forward(x_padded)
            if args.cuda:
                torch.cuda.synchronize()
            e = time.time()
            total_time += (e - s)
            out_net['x_hat'].clamp_(0, 1)
            out_net["x_hat"] = crop(out_net["x_hat"], padding)
            print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
            print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
            print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
            PSNR += compute_psnr(x, out_net["x_hat"])
            MS_SSIM += compute_msssim(x, out_net["x_hat"])
            Bit_rate += compute_bpp(out_net)
        del out_net, x, img
        gc.collect()
    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.4f} ms')
    return {"PSNR": PSNR, 
            "Bit rate": Bit_rate}

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)
class ImageNet(Dataset):
    def __init__(self, root, transform=None):
        #splitdir = Path(root)

        #if not splitdir.is_dir():
        #    raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = glob.glob(root + '/*/*.JPEG')

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
            out['psnr'] = -10 * math.log10(out["mse_loss"].item())
            out["msssim"] = ms_ssim(torch.round(output["x_hat"]*255), torch.round(target*255), data_range=255, size_average=True)
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
        
def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse'
):  

    model.train()
    device = next(model.parameters()).device

    for i, d in tqdm(enumerate(train_dataloader)):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        if torch.isnan(out_criterion["loss"]):
            # pdb.set_trace()
            del out_net, out_criterion
            torch.cuda.empty_cache()
            continue
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 20 == 0:
            if type == 'mse':
                logging.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                    f'\tPSNR: {out_criterion["psnr"]:.2f} |'
                    f'\tMS-SSIM: {out_criterion["msssim"].item():.4f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                logging.info(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        psnr = AverageMeter()
        msssim = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                psnr.update(out_criterion["psnr"])
                msssim.update(out_criterion["msssim"])

        logging.info(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.4f} |"
            f"\tBpp loss: {bpp_loss.avg:.3f} |"
            f"\tPSNR: {psnr.avg:.2f} |"
            f"\tMS-SSIM: {msssim.avg:.4f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        logging.info(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + filename + "checkpoint_latest.pth.tar")
    if epoch % 10 == 0:
         torch.save(state, save_path + filename + "_" + str(epoch) +"_" +  "checkpoint.pth.tar")
    if is_best:
        torch.save(state, save_path + filename + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=os.cpu_count() // 2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=42, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--K", type=int, default=3
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=False
    )
    parser.add_argument(
        "--kodak_path" , type=str, default=None 
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    setup_logger(save_path + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))

    train_transforms = transforms.Compose(
       [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    #train_dataset = ImageNet(args.dataset + "/train.X[1-4]", transform=train_transforms)
    #test_dataset = ImageNet(args.dataset +"/val.X" , transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = Cheng2020AnchorCheckerboardGMM(N=args.N, K = args.K)
    #net = Cheng2020AnchorCheckerboard(N = args.N)
    net = net.to(device)
    #print(calculate_flops(net, (1, 3, 768, 512)))
    


    #if args.cuda and torch.cuda.device_count() > 1:
    #    net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.gamma, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cuda")
        #checkpointをGPUに読み込むとCUDA OUT OF MEMORYになるので、CPUに読み込んでからGPUに転送
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if args.kodak_path is not None:
        print("kodak test mode enabled, testing...")
        print(test_kodak(args.kodak_path, net, args))
        
    best_loss = float("inf")
    for epoch in tqdm(range(last_epoch, args.epochs)):
        # print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logging.info('======Current epoch %s ======'%epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            type
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type)
        writer.add_scalar('test_loss', loss, epoch)
        if args.kodak_path is not None:
            kodak_rd = test_kodak(args.kodak_path, net, args)
            writer.add_scalars('kodak_RD', kodak_rd, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                str(net.__class__.__name__) + "_",
            )


if __name__ == "__main__":
    main(sys.argv[1:])