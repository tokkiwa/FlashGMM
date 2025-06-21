import types
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping
import torch
import torch.nn as nn
import PIL
from torch import Tensor
from torchvision.transforms import ToTensor, ToPILImage
import tqdm
from compressai.entropy_models import EntropyBottleneck 
import math
from compressai.layers import (
    AttentionBlock,
    MaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

from compressai.models.base import SimpleVAECompressionModel
from compressai.models.utils import conv, deconv

#public time_logger
def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

class TwoPassTimeLogger:
    '''
    class variable: time1, time2, time12, total_count, count1, count2
    '''
    def __init__(self):
        self.time1 = []
        self.time2 = []
        self.time12 = []
        self.total_count = 0
        self.count1 = 0
        self.count2 = 0
    def put(self, time):
        #put input to time1 and time2 one by one
        if self.total_count % 2 == 0:
            self.time1.append(time)
            self.time12.append(time)
            self.count1 += 1
            self.total_count += 1
        else:
            self.time2.append(time)
            self.time12.append(time)
            self.count2 += 1
            self.total_count += 1
            
    def print(self):
        print("time1: ", sum(self.time1) / len(self.time1))
        print("time2: ", sum(self.time2) / len(self.time2))
        print("time12: ", sum(self.time12) / len(self.time1))
        print("total_count: ", self.total_count)
        print("count1: ", self.count1)
        print("count2: ", self.count2)
        print("average time per call: ", (sum(self.time12)) / self.total_count)


time_logger_comp = TwoPassTimeLogger()
time_logger_decomp = TwoPassTimeLogger()
    
    
class Cheng2020AnchorCheckerboardGMMv2(SimpleVAECompressionModel):
    def __init__(self, N=192, K = 4, quantizer = "noise", **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": CheckerboardLatentCodec(
                    latent_codec={
                        "y": GaussianMixtureConditionalLatentCodec(K = self.K, quantizer=quantizer),
                    },
                    entropy_parameters=nn.Sequential(
                        nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 10 // 3, N * 10 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 10 // 3, 3 * self.K * N, 1),
                    ), #follwoing Cheng2020 paper. In Lin's multistage, input of final stage is N * 8 //3
                    context_prediction=CheckerboardMaskedConv2d(
                        N, 2 * N, kernel_size=5, stride=1, padding=2
                    ),
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


from compressai.entropy_models import GaussianMixtureConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module
import time
from compressai.latent_codecs.base import LatentCodec

import os
import sys
print("compressaiのインストール先:", os.path.dirname(sys.modules["compressai"].__file__))

class GaussianMixtureConditionalLatentCodec(LatentCodec):
    """Gaussian conditional for compressing latent ``y`` using ``ctx_params``.

    Probability model for Gaussian of ``(scales, means)``.

    Gaussian conditonal entropy model introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: Unlike the original paper, which models only the scale
       (i.e. "width") of the Gaussian, this implementation models both
       the scale and the mean (i.e. "center") of the Gaussian.

    .. code-block:: none

                          ctx_params
                              │
                              ▼
                              │
                           ┌──┴──┐
                           │  EP │
                           └──┬──┘
                              │
               ┌───┐  y_hat   ▼
        y ──►──┤ Q ├────►────····──►── y_hat
               └───┘          GC

    """

    gaussian_mixture_conditional: GaussianMixtureConditional
    entropy_parameters: nn.Module

    def __init__(
        self,
        K: int = 4,
        scale_table: Optional[Union[List, Tuple]] = None,
        gaussian_mixture_conditional: Optional[GaussianMixtureConditional] = None,
        entropy_parameters: Optional[nn.Module] = None,
        quantizer: str = "noise",
        chunks: Tuple[str] = ("scales", "means", "weights"),
        **kwargs,
    ):
        super().__init__()
        self.K = K
        self.quantizer = quantizer
        self.gaussian_mixture_conditional = gaussian_mixture_conditional if gaussian_mixture_conditional is not None else GaussianMixtureConditional(
            K=K,
            scale_table=scale_table,
        )
        print(self.gaussian_mixture_conditional)
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.chunks = tuple(chunks)

    def forward(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights = self._chunk(gaussian_params)
        weights = self._reshape_gmm_weight(weights)
        y_hat, y_likelihoods = self.gaussian_mixture_conditional(y, 
                                                                 scales_hat, 
                                                                 means_hat, 
                                                                 weights
                                                                 )
        # if self.quantizer == "ste":
        #     y_hat = quantize_ste(y - means_hat) + means_hat
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights = self._chunk(gaussian_params)
        weights = self._reshape_gmm_weight(weights)
        #indexes = self.gaussian_conditional.build_indexes(scales_hat)
        start_time = time.time()
        y_strings, y_hat = self.gaussian_mixture_conditional.compress(y, scales_hat, means_hat, weights)
        compression_time = time.time() - start_time
        time_logger_comp.put(compression_time)
        return {"strings": [y_strings], "shape": y.shape[2:4], "y_hat": y_hat}

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, int],
        ctx_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights = self._chunk(gaussian_params)
        weights = self._reshape_gmm_weight(weights)
        #indexes = self.gaussian_conditional.build_indexes(scales_hat)
        start_time = time.time()
        y_hat = self.gaussian_mixture_conditional.decompress(
            *y_strings, scales_hat, means_hat, weights
        )
        decompression_time = time.time() - start_time
        time_logger_decomp.put(decompression_time)
        assert y_hat.shape[2:4] == shape
        return {"y_hat": y_hat}

    def _chunk(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        scales, means = None, None
        if self.chunks == ("scales",):
            scales = params
        if self.chunks == ("means",):
            means = params
        if self.chunks == ("scales", "means"):
            scales, means = params.chunk(2, 1)
        if self.chunks == ("means", "scales"):
            means, scales = params.chunk(2, 1)
        if self.chunks == ("scales", "means", "weights"):
            scales, means, weights = params.chunk(3, 1)
            return scales, means, weights
        return scales, means
    
    def _reshape_gmm_weight(self, weight):
        weight = torch.reshape(weight, (weight.size(0), self.K, weight.size(1) // self.K, weight.size(2), -1))
        weight = nn.functional.softmax(weight, dim=1)
        weight = torch.reshape(weight, (weight.size(0), weight.size(1) * weight.size(2), weight.size(3), -1))
        return weight

class HyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Hyperprior entropy modeling introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

                 ┌──────────┐
            ┌─►──┤ lc_hyper ├──►─┐
            │    └──────────┘    │
            │                    ▼ params
            │                    │
            │                 ┌──┴───┐
        y ──┴───────►─────────┤ lc_y ├───►── y_hat
                              └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                 ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            ┌─►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►─┐
            │    └───┘     └───┘        EB        └───┘    │
            │                                              │
            │                  ┌──────────────◄────────────┘
            │                  │            params
            │               ┌──┴──┐
            │               │  EP │
            │               └──┬──┘
            │                  │
            │   ┌───┐  y_hat   ▼
        y ──┴─►─┤ Q ├────►────····────►── y_hat
                └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """

    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self, latent_codec: Optional[Mapping[str, LatentCodec]] = None, **kwargs
    ):
        super().__init__()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": GaussianMixtureConditionalLatentCodec,
                "hyper": HyperLatentCodec,
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"](y)
        y_out = self.latent_codec["y"](y, hyper_out["params"])
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
            "y_hat": y_out["y_hat"],
        }

    def compress(self, y: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"].compress(y)
        y_out = self.latent_codec["y"].compress(y, hyper_out["params"])
        [z_strings] = hyper_out["strings"]
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": {"y": y_out["shape"], "hyper": hyper_out["shape"]},
            "y_hat": y_out["y_hat"],
        }

    def decompress(
        self, strings: List[List[bytes]], shape: Dict[str, Tuple[int, ...]], **kwargs
    ) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        #assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec["hyper"].decompress([z_strings], shape["hyper"])
        y_out = self.latent_codec["y"].decompress(
            y_strings_, shape["y"], hyper_out["params"]
        )
        return {"y_hat": y_out["y_hat"]}
class CheckerboardLatentCodec(LatentCodec):
    """Reconstructs latent using 2-pass context model with checkerboard anchors.

    Checkerboard context model introduced in [He2021].

    See :py:class:`~compressai.models.sensetime.Cheng2020AnchorCheckerboard`
    for example usage.

    - `forward_method="onepass"` is fastest, but does not use
      quantization based on the intermediate means.
      Uses noise to model quantization.
    - `forward_method="twopass"` is slightly slower, but accurately
      quantizes via STE based on the intermediate means.
      Uses the same operations as [Chandelier2023].
    - `forward_method="twopass_faster"` uses slightly fewer
      redundant operations.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. code-block:: none

        0. Input:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        1. Decode anchors:

        ◌ □ ◌ □
        □ ◌ □ ◌
        ◌ □ ◌ □

        2. Decode non-anchors:

        ■ ◌ ■ ◌
        ◌ ■ ◌ ■
        ■ ◌ ■ ◌

        3. End result:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        LEGEND:
        ■   decoded
        ◌   currently decoding
        □   empty
    """

    latent_codec: Mapping[str, LatentCodec]

    entropy_parameters: nn.Module

    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        entropy_parameters: Optional[nn.Module] = None,
        context_prediction: Optional[nn.Module] = None,
        anchor_parity="even",
        forward_method="onepass",
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {"odd": "even", "even": "odd"}[anchor_parity]
        self.forward_method = forward_method
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.context_prediction = context_prediction or nn.Identity()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": lambda: GaussianConditionalLatentCodec(quantizer="ste"),
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        if self.forward_method == "onepass":
            return self._forward_onepass(y, side_params)
        if self.forward_method == "twopass":
            return self._forward_twopass(y, side_params)
        if self.forward_method == "twopass_faster":
            return self._forward_twopass_faster(y, side_params)
        raise ValueError(f"Unknown forward method: {self.forward_method}")

    def _forward_onepass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Fast estimation with single pass of the entropy parameters network.

        It is faster than the twopass method (only one pass required!),
        but also less accurate.

        This method uses uniform noise to roughly model quantization.
        """
        y_hat = self.quantize(y)
        y_ctx = self._keep_only(self.context_prediction(y_hat), "non_anchor")
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        The first pass gets ``y_hat`` and ``means_hat`` for the anchors.
        This ``y_hat`` is used as context to predict the non-anchors.
        The second pass gets ``y_hat`` for the non-anchors.
        The two ``y_hat`` tensors are then combined. The resulting
        ``y_hat`` models the effects of quantization more realistically.

        To compute ``y_hat_anchors``, we need the predicted ``means_hat``:
        ``y_hat = quantize_ste(y - means_hat) + means_hat``.
        Thus, two passes of ``entropy_parameters`` are necessary.

        """
        B, C, H, W = y.shape

        params = y.new_zeros((B, C * 2, H, W))

        y_hat_anchors = self._forward_twopass_step(
            y, side_params, params, self._y_ctx_zero(y), "anchor"
        )

        y_hat_non_anchors = self._forward_twopass_step(
            y, side_params, params, self.context_prediction(y_hat_anchors), "non_anchor"
        )

        y_hat = y_hat_anchors + y_hat_non_anchors
        y_out = self.latent_codec["y"](y, params)

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass_step(
        self, y: Tensor, side_params: Tensor, params: Tensor, y_ctx: Tensor, step: str
    ) -> Dict[str, Any]:
        # NOTE: The _i variables contain only the current step's pixels.
        assert step in ("anchor", "non_anchor")

        params_i = self.entropy_parameters(self.merge(y_ctx, side_params))

        # Save params for current step. This is later used for entropy estimation.
        self._copy(params, params_i, step)

        # Apply latent_codec's "entropy_parameters()", if it exists. Usually identity.
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params_i = func(params_i)

        # Keep only elements needed for current step.
        # It's not necessary to mask the rest out just yet, but it doesn't hurt.
        params_i = self._keep_only(params_i, step)
        y_i = self._keep_only(y, step)

        # Determine y_hat for current step, and mask out the other pixels.
        _, means_i = self.latent_codec["y"]._chunk(params_i)
        y_hat_i = self._keep_only(quantize_ste(y_i - means_i) + means_i, step)

        return y_hat_i

    def _forward_twopass_faster(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        This version was written based on the paper description.
        It is a tiny bit faster than the twopass method since
        it avoids a few redundant operations. The "probably unnecessary"
        operations can likely be removed as well.
        The speedup is very small, however.
        """
        y_ctx = self._y_ctx_zero(y)
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params = func(params)
        params = self._keep_only(params, "anchor")  # Probably unnecessary.
        _, means_hat = self.latent_codec["y"]._chunk(params)
        y_hat_anchors = quantize_ste(y - means_hat) + means_hat
        y_hat_anchors = self._keep_only(y_hat_anchors, "anchor")

        y_ctx = self.context_prediction(y_hat_anchors)
        y_ctx = self._keep_only(y_ctx, "non_anchor")  # Probably unnecessary.
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)

        # Reuse quantized y_hat that was used for non-anchor context prediction.
        y_hat = y_out["y_hat"]
        self._copy(y_hat, y_hat_anchors, "anchor")  # Probably unnecessary.

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    @torch.no_grad()
    def _y_ctx_zero(self, y: Tensor) -> Tensor:
        """Create a zero tensor with correct shape for y_ctx."""
        y_ctx_meta = self.context_prediction(y.to("meta"))
        return y.new_zeros(y_ctx_meta.shape)

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].compress(y_[i], params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]

        y_hat = self.embed(y_hat_)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        y_strings_ = strings
        #n = len(y_strings_[0])
        #assert len(y_strings_) == 2
        #assert all(len(x) == n for x in y_strings_)
        
        n = 1

        c, h, w = shape
        y_i_shape = (h, w // 2)
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].decompress(
                [y_strings_[i]], y_i_shape, params_i
            )
            y_hat_[i] = y_out["y_hat"]

        y_hat = self.embed(y_hat_)

        return {
            "y_hat": y_hat,
        }

    def unembed(self, y: Tensor) -> Tensor:
        """Separate single tensor into two even/odd checkerboard chunks.

        .. code-block:: none

            ■ □ ■ □         ■ ■   □ □
            □ ■ □ ■   --->  ■ ■   □ □
            ■ □ ■ □         ■ ■   □ □
        """
        n, c, h, w = y.shape
        y_ = y.new_zeros((2, n, c, h, w // 2))
        if self.anchor_parity == "even":
            y_[0, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 1::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 0::2]
        else:
            y_[0, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 0::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 1::2]
        return y_

    def embed(self, y_: Tensor) -> Tensor:
        """Combine two even/odd checkerboard chunks into single tensor.

        .. code-block:: none

            ■ ■   □ □         ■ □ ■ □
            ■ ■   □ □   --->  □ ■ □ ■
            ■ ■   □ □         ■ □ ■ □
        """
        num_chunks, n, c, h, w_half = y_.shape
        assert num_chunks == 2
        y = y_.new_zeros((n, c, h, w_half * 2))
        if self.anchor_parity == "even":
            y[..., 0::2, 0::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 1::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[1, ..., 1::2, :]
        else:
            y[..., 0::2, 1::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 0::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[1, ..., 1::2, :]
        return y

    def _copy(self, dest: Tensor, src: Tensor, step: str) -> None:
        """Copy pixels in the current step."""
        assert step in ("anchor", "non_anchor")
        parity = self.anchor_parity if step == "anchor" else self.non_anchor_parity
        if parity == "even":
            dest[..., 0::2, 0::2] = src[..., 0::2, 0::2]
            dest[..., 1::2, 1::2] = src[..., 1::2, 1::2]
        else:
            dest[..., 0::2, 1::2] = src[..., 0::2, 1::2]
            dest[..., 1::2, 0::2] = src[..., 1::2, 0::2]

    def _keep_only(self, y: Tensor, step: str, inplace: bool = False) -> Tensor:
        """Keep only pixels in the current step, and zero out the rest."""
        return self._mask(
            y,
            parity=self.non_anchor_parity if step == "anchor" else self.anchor_parity,
            inplace=inplace,
        )

    def _mask(self, y: Tensor, parity: str, inplace: bool = False) -> Tensor:
        if not inplace:
            y = y.clone()
        if parity == "even":
            y[..., 0::2, 0::2] = 0
            y[..., 1::2, 1::2] = 0
        elif parity == "odd":
            y[..., 0::2, 1::2] = 0
            y[..., 1::2, 0::2] = 0
        elif parity == "all":
            y[:] = 0
        return y

    def merge(self, *args):
        return torch.cat(args, dim=1)

    def quantize(self, y: Tensor) -> Tensor:
        mode = "noise" if self.training else "dequantize"
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat
class _SetDefaultMixin:
    """Convenience functions for initializing classes with defaults."""

    def _setdefault(self, k, v, f):
        """Initialize attribute ``k`` with value ``v`` or ``f()``."""
        v = v or f()
        setattr(self, k, v)

    # TODO instead of save_direct, override load_state_dict() and state_dict()
    def _set_group_defaults(self, group_key, group_dict, defaults, save_direct=False):
        """Initialize attribute ``group_key`` with items from
        ``group_dict``, using defaults for missing keys.
        Ensures ``nn.Module`` attributes are properly registered.

        Args:
            - group_key:
                Name of attribute.
            - group_dict:
                Dict of items to initialize ``group_key`` with.
            - defaults:
                Dict of defaults for items not in ``group_dict``.
            - save_direct:
                If ``True``, save items directly as attributes of ``self``.
                If ``False``, save items in a ``nn.ModuleDict``.
        """
        group_dict = group_dict if group_dict is not None else {}
        for k, f in defaults.items():
            if k in group_dict:
                continue
            group_dict[k] = f()
        if save_direct:
            for k, v in group_dict.items():
                setattr(self, k, v)
        else:
            group_dict = nn.ModuleDict(group_dict)
        setattr(self, group_key, group_dict)

from compressai.ops import quantize_ste
from compressai.registry import register_module

__all__ = [
    "HyperLatentCodec",
]


@register_module("HyperLatentCodec")
class HyperLatentCodec(LatentCodec):
    """Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

    "Hyper" side-information branch introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: ``HyperLatentCodec`` should be used inside
       ``HyperpriorLatentCodec`` to construct a full hyperprior.

    .. code-block:: none

               ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
        y ──►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►── params
               └───┘     └───┘        EB        └───┘

    """

    entropy_bottleneck: EntropyBottleneck
    h_a: nn.Module
    h_s: nn.Module

    def __init__(
        self,
        entropy_bottleneck: Optional[EntropyBottleneck] = None,
        h_a: Optional[nn.Module] = None,
        h_s: Optional[nn.Module] = None,
        quantizer: str = "noise",
        **kwargs,
    ):
        super().__init__()
        assert entropy_bottleneck is not None
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()
        self.quantizer = quantizer

    def forward(self, y: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quantizer == "ste":
            z_medians = self.entropy_bottleneck._get_medians()
            z_hat = quantize_ste(z - z_medians) + z_medians
        params = self.h_s(z_hat)
        return {"likelihoods": {"z": z_likelihoods}, "params": params}

    def compress(self, y: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        shape = z.size()[-2:]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat)
        return {"strings": [z_strings], "shape": shape, "params": params}

    def decompress(
        self, strings: List[List[bytes]], shape: Tuple[int, int], **kwargs
    ) -> Dict[str, Any]:
        (z_strings,) = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        params = self.h_s(z_hat)
        return {"params": params}

class LatentCodec(nn.Module, _SetDefaultMixin):
    def forward(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def compress(self, y: Tensor, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def decompress(
        self, strings: List[List[bytes]], shape: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError
    
class CheckerboardMaskedConv2d(MaskedConv2d):
    r"""Checkerboard masked 2D convolution; mask future "unseen" pixels.

    Checkerboard mask variant used in
    `"Checkerboard Context Model for Efficient Learned Image Compression"
    <https://arxiv.org/abs/2103.15306>`_, by Dailan He, Yaoyan Zheng,
    Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        _, _, h, w = self.mask.size()
        self.mask[:] = 1
        self.mask[:, :, 0::2, 0::2] = 0
        self.mask[:, :, 1::2, 1::2] = 0
        self.mask[:, :, h // 2, w // 2] = mask_type == "B"

if __name__ == "__main__":
    print(f"{torch.cuda.is_available()=}")
    model = Cheng2020AnchorCheckerboardGMMv2(K=4)
    state_dict = torch.load("/root/shared_smurai/CKBD_GMM_log/ver2/0.0067Cheng2020AnchorCheckerboardGMMv2_checkpoint_best.pth.tar")["state_dict"]
    #rename keys: "_matrix" to "matrices" and "_bias" to "biases"
    new_state_dict = {}
    for key in state_dict.keys():
        if "matrices" in key:
            new_key = key.replace("matrices.", "_matrix")
            new_state_dict[new_key] = state_dict[key]
        elif "biases" in key:
            new_key = key.replace("biases.", "_bias")
            new_state_dict[new_key] = state_dict[key]
        elif "factors" in key:
            new_key = key.replace("factors.", "_factor")
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    image_path = "/root/shared_smurai/kodak/"

    test_image = [ToTensor()(PIL.Image.open(image_path + f"kodim0{i}.png")).unsqueeze(0) for i in range(1,10)]
    test_image += [ToTensor()(PIL.Image.open(image_path + f"kodim{i}.png")).unsqueeze(0) for i in range(10,25)]
    model.update()
    num_repeat = 1
    
    avarge_psnr = AverageMeter()
    avarge_bpp = AverageMeter()

    for _ in tqdm.tqdm(range(num_repeat)):
        for image in tqdm.tqdm(test_image):
            comp = model.compress(image)
            decomp = model.decompress(comp["strings"], comp["shape"])
            #calculate bpp, psnr and take average
            num_pixels = image.size(0) * image.size(2) * image.size(3)  
            print(f"num_pixels: {image.size(0)} * {image.size(2)} * {image.size(3)} = {num_pixels}")
            bpp = sum(len(s[0]) for s in comp["strings"]) * 8.0 / num_pixels
            psnr = compute_psnr(image, decomp["x_hat"])
            avarge_psnr.update(psnr)
            avarge_bpp.update(bpp)
            
    print("compression log: ")
    time_logger_comp.print()
    print("decompression log: ")
    time_logger_decomp.print()
    print(f"avarge psnr: {avarge_psnr.avg}")
    print(f"avarge bpp: {avarge_bpp.avg}")