import torch
import torch.nn as nn
import torch.nn.functional as F


class ReversibleNormalization(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(ReversibleNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = 1
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
        return self.mean, self.stdev

    def normalize(self, x):
        mean, stdev = self._get_statistics(x)
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (
                self.affine_weight + self.eps * self.affine_weight
            )
        x = x * self.stdev + self.mean
        return x


class EndogenousAnchoredTemporalAddressing(nn.Module):
    def __init__(self, endo_dim, exo_dim, k_lookback, dropout=0.1):
        super().__init__()
        self.endo_dim = endo_dim
        self.exo_dim = exo_dim
        self.k_lookback = k_lookback
        self.dropout = dropout

        self.endo_expander = nn.Sequential(
            nn.Conv1d(
                self.endo_dim,
                self.exo_dim * self.endo_dim,
                kernel_size=1,
                groups=self.endo_dim,
            ),
            nn.Dropout(self.dropout),
        )

    def forward(self, endo, exo, return_components=False):
        bsz, endo_dim, time_steps = endo.shape
        exo_dim = exo.shape[1]

        endo_expanded = self.endo_expander(endo)
        query = endo_expanded.reshape(bsz, endo_dim, exo_dim, time_steps).unsqueeze(-1)

        exo_padded = F.pad(exo, (self.k_lookback - 1, 0))
        exo_windows = exo_padded.unfold(dimension=2, size=self.k_lookback, step=1)
        key_value = exo_windows.unsqueeze(1)

        product = query * key_value
        weights = F.softmax(torch.abs(product), dim=-1)
        sign_direction = torch.sign(product)
        exo_encoded = (weights * sign_direction * key_value).sum(dim=-1)

        endo_original = endo.unsqueeze(2)
        combined = torch.cat([exo_encoded, endo_original], dim=2)
        if not return_components:
            return combined

        return combined, {
            "query": query,
            "key_value": key_value,
            "product": product,
            "weights": weights,
            "sign_direction": sign_direction,
            "signed_weights": weights * sign_direction,
            "exo_encoded": exo_encoded,
            "endo_original": endo_original,
        }


class DynamicDistributionShiftTracer(nn.Module):
    def __init__(self, n_channels, k_lookback=16, dropout=0.3):
        super().__init__()
        self.k = k_lookback
        self.n_channels = n_channels
        self.kernel_generator = nn.Conv1d(
            in_channels=n_channels * 6,
            out_channels=n_channels * k_lookback,
            kernel_size=1,
            groups=n_channels,
        )
        nn.init.xavier_normal_(self.kernel_generator.weight, gain=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, channels, time_steps = x.shape
        lookback = self.k

        global_mean = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, time_steps)
        x_squared = torch.pow(x, 2)
        avg_x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        avg_x2 = F.avg_pool1d(x_squared, kernel_size=5, stride=1, padding=2)
        rolling_var = torch.clamp(avg_x2 - torch.pow(avg_x, 2), min=1e-6)
        rolling_std = torch.sqrt(rolling_var)
        third_moment = torch.pow(x - avg_x, 3)
        rolling_max = F.max_pool1d(x, kernel_size=5, stride=1, padding=2)
        rolling_min = -F.max_pool1d(-x, kernel_size=5, stride=1, padding=2)
        x_drop = self.dropout(x)

        stats_stack = torch.stack(
            [x_drop, global_mean, rolling_std, third_moment, rolling_max, rolling_min],
            dim=2,
        )
        gen_in = stats_stack.reshape(bsz, channels * 6, time_steps)

        kernels_flat = self.kernel_generator(gen_in)
        kernels = kernels_flat.reshape(bsz, channels, lookback, time_steps)

        x_padded = F.pad(x, (lookback - 1, 0))
        x_windows = x_padded.unfold(dimension=2, size=lookback, step=1).permute(0, 1, 3, 2)

        attn_weights = F.softmax(torch.abs(kernels), dim=2)
        out = (x_windows * attn_weights * torch.sign(kernels)).sum(dim=2)
        return out.permute(0, 2, 1)


class SimpleDDST(nn.Module):
    def __init__(self, n_channels, k_lookback=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(6, k_lookback),
            padding=(0, k_lookback // 2),
            groups=n_channels,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        _, _, time_steps = x.shape

        global_mean = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, time_steps)
        x_squared = torch.pow(x, 2)
        avg_x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        avg_x2 = F.avg_pool1d(x_squared, kernel_size=5, stride=1, padding=2)
        rolling_var = torch.clamp(avg_x2 - torch.pow(avg_x, 2), min=1e-6)
        rolling_std = torch.sqrt(rolling_var)
        third_moment = torch.pow(x - avg_x, 3)
        rolling_max = F.max_pool1d(x, kernel_size=5, stride=1, padding=2)
        rolling_min = -F.max_pool1d(-x, kernel_size=5, stride=1, padding=2)
        x_drop = self.dropout(x)

        gen_in = torch.stack(
            [x_drop, global_mean, rolling_std, third_moment, rolling_max, rolling_min],
            dim=2,
        )

        out = self.conv(gen_in)
        out = out[:, :, :, :time_steps]
        out = self.activation(out).squeeze(2)
        return (x + out).permute(0, 2, 1)


class GatedMultivariateInteractionLayer(nn.Module):
    def __init__(self, n_channels, d_model=32, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.var_projector = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Linear(d_model, n_channels)
        self.channel_gate = nn.Parameter(torch.zeros(n_channels))

    def forward(self, pred_ci):
        pred_ci = pred_ci.permute(0, 2, 1)
        batch_size, time_steps, channels = pred_ci.shape
        x_flat = pred_ci.reshape(batch_size * time_steps, channels)
        global_ctx = self.var_projector(x_flat)
        correction_flat = self.out_proj(global_ctx)
        correction = correction_flat.reshape(batch_size, time_steps, channels)
        gate = torch.tanh(self.channel_gate).view(1, 1, channels)
        return correction * gate


RevIN = ReversibleNormalization
Temp_Causal_Address = EndogenousAnchoredTemporalAddressing
SimpleDistributionShiftTracer = SimpleDDST
GatedInterAct = GatedMultivariateInteractionLayer
EATA = EndogenousAnchoredTemporalAddressing
DDST = DynamicDistributionShiftTracer
GMIL = GatedMultivariateInteractionLayer
