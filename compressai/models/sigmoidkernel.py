import torch.nn as nn
import torch


from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

from .waseda import Cheng2020Attention

class SigmoidCDFCompressor(Cheng2020Attention):
    def __init__(self, N=192,K = 3, **kwargs):
        super().__init__(N, **kwargs)
        self.K = K
        self.M = N
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, 3 * self.M * self.K, 1),
        )

        
    def _cdf_sigmoid_kernel(self, x, params, K):
        """
        Return the modeled CDF of of the sigmoid kernel linear model.
        definition: CDF(x|\mu, \sigma, w) = \sum_{i=1}^{K} w_i * sigmoid((x - \mu_i) / \sigma_i)
        conduct batched calculation.
        """
        mu, sigma, w = params.chunk(3, dim=1) #[B, K * M * 3, H, W] -> 3 * [B, K * M, H, W]
        sigma = nn.functional.relu(sigma) + 1e-4

        x = x.unsqueeze(1).expand(-1, K, -1, -1, -1) # [B, M, H, W] -> [B, K, M, H, W]
        mu = mu.reshape(x.size()) # [B, K*M, H, W] -> [B, K, M, H, W]
        sigma = sigma.reshape(x.size()) #of size [B, K, M, H, W]
        w = w.reshape(x.size()) #of size [B, K, M, H, W]
        w = nn.functional.softmax(w,dim=1)
        return (w * torch.sigmoid((x - mu) / sigma)).sum(1) #of size [B, M, H, W]

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat) 

        y_hat = self.entropy_bottleneck.quantize(
            y, "noise" if self.training else "dequantize"
        ) #disabled gaussian conditional. so use entrpy bottleneck to quantize y
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )#of size [B, C * 3K, H, W]
        #direct cdf likelihood calculation
        half = torch.full_like(y, 0.5).to(y.device)
        y_likelihoods = \
            self._cdf_sigmoid_kernel(y_hat + half, gaussian_params, self.K) \
            - self._cdf_sigmoid_kernel(y_hat - half, gaussian_params, self.K)
        y_likelihoods = self.entropy_bottleneck.likelihood_lower_bound(y_likelihoods)
        
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net