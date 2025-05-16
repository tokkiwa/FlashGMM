# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn

from torch import Tensor
import torch

from compressai.entropy_models import GaussianMixtureConditional
from compressai.ops import quantize_ste
from compressai.registry import register_module
import time
from .base import LatentCodec

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
        assert quantizer in ["noise", "weighted_mean_ste"], f"quantizer {quantizer} not supported"
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
        if self.quantizer == "noise":
            y_hat, y_likelihoods = self.gaussian_mixture_conditional(y, 
                                                                    scales_hat, 
                                                                    means_hat, 
                                                                    weights
                                                                    )
        elif self.quantizer == "weighted_mean_ste":
            # means_hat and weights here is the shape "B x (K x C) x H x W"
            # decompose them to "B x K x C x H x W", multiply them and get summation along K
            # to get the weighted mean of shape "B x C x H x W"
            means_hat_expanded = means_hat.view(means_hat.size(0), self.K, -1, means_hat.size(2), means_hat.size(3))
            weights_expanded = weights.view(weights.size(0), self.K, -1, weights.size(2), weights.size(3))
            weighted_sum = torch.sum(means_hat_expanded * weights_expanded, dim=1)
            y_hat = quantize_ste(y - weighted_sum) + weighted_sum
            means_hat_expanded = means_hat_expanded - weighted_sum.unsqueeze(1).repeat(1, self.K, 1, 1, 1)
            means_hat = means_hat_expanded.view(means_hat.size(0), -1, means_hat.size(2), means_hat.size(3))
            y_hat, y_likelihoods = self.gaussian_mixture_conditional(y_hat,
                                                                scales_hat, 
                                                                means_hat, 
                                                                weights
                                                                )
            
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat, weights = self._chunk(gaussian_params)
        weights = self._reshape_gmm_weight(weights)
        #indexes = self.gaussian_conditional.build_indexes(scales_hat)
        start_time = time.time()
        if self.quantizer == "noise":
            y_strings, y_hat = self.gaussian_mixture_conditional.compress(y, scales_hat, means_hat, weights)
        elif self.quantizer == "weighted_mean_ste":
            # means_hat and weights here is the shape "B x (K x C) x H x W"
            # decompose them to "B x K x C x H x W", multiply them and get summation along K
            # to get the weighted mean of shape "B x C x H x W"
            means_hat_expanded = means_hat.view(means_hat.size(0), self.K, -1, means_hat.size(2), means_hat.size(3))
            weights_expanded = weights.view(weights.size(0), self.K, -1, weights.size(2), weights.size(3))
            weighted_sum = torch.sum(means_hat_expanded * weights_expanded, dim=1)
            y_hat = quantize_ste(y - weighted_sum)
            means_hat_expanded = means_hat_expanded - weighted_sum.unsqueeze(1).repeat(1, self.K, 1, 1, 1)
            means_hat = means_hat_expanded.view(means_hat.size(0), -1, means_hat.size(2), means_hat.size(3))
            y_strings, y_hat = self.gaussian_mixture_conditional.compress(y_hat, scales_hat, means_hat, weights)
        print(f"time taken to GMM compression: {time.time() - start_time}" )
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
        if self.quantizer == "noise":
            start_time = time.time()
            y_hat = self.gaussian_mixture_conditional.decompress(
                *y_strings, scales_hat, means_hat, weights
            )
            print(f"time taken to GMM decompression: {time.time() - start_time}" )
        elif self.quantizer == "weighted_mean_ste":
            # means_hat and weights here is the shape "B x (K x C) x H x W"
            # decompose them to "B x K x C x H x W", multiply them and get summation along K
            # to get the weighted mean of shape "B x C x H x W"
            means_hat_expanded = means_hat.view(means_hat.size(0), self.K, -1, means_hat.size(2), means_hat.size(3))
            weights_expanded = weights.view(weights.size(0), self.K, -1, weights.size(2), weights.size(3))
            weighted_sum = torch.sum(means_hat_expanded * weights_expanded, dim=1)
            means_hat_expanded = means_hat_expanded - weighted_sum.unsqueeze(1).repeat(1, self.K, 1, 1, 1)
            means_hat = means_hat_expanded.view(means_hat.size(0), -1, means_hat.size(2), means_hat.size(3))
            y_hat = self.gaussian_mixture_conditional.decompress(
                *y_strings, scales_hat, means_hat, weights
            )
            y_hat = y_hat + weighted_sum
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
