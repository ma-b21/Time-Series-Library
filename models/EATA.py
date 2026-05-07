import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from layers.EATA_utils import (
    DynamicDistributionShiftTracer,
    EndogenousAnchoredTemporalAddressing,
    GatedMultivariateInteractionLayer,
    ReversibleNormalization,
    SimpleDDST,
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.exo_dim = configs.enc_in
        self.k_lookback = configs.k_lookback
        self.method = configs.method
        self.hidden = configs.hidden
        self.endo_dim = 1 if self.features == "MS" else self.exo_dim
        self.emb_dim = self.exo_dim + 1
        self.alpha_param = nn.Parameter(torch.zeros(self.pred_len))

        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.bias = configs.bias
        self.rnn_output = 2 * self.d_model

        self.revin = ReversibleNormalization(self.exo_dim)

        self.endogenous_trend = nn.Linear(self.seq_len, self.pred_len)
        self.eata = EndogenousAnchoredTemporalAddressing(
            self.endo_dim, self.exo_dim, self.k_lookback, self.dropout
        )
        self.eata_projection = nn.Linear(self.emb_dim, self.d_model)
        self.eata_norm = nn.LayerNorm(self.d_model)

        self.temporal_encoder = nn.GRU(
            self.d_model, self.d_model, batch_first=True, bidirectional=True
        )
        if self.method == "Dynamic":
            self.ddst = DynamicDistributionShiftTracer(
                self.emb_dim, self.k_lookback, self.dropout
            )
        elif self.method == "Simple":
            self.ddst = nn.Sequential(
                SimpleDDST(self.emb_dim, self.k_lookback, self.dropout),
                nn.Linear(self.emb_dim, self.rnn_output),
            )
        else:
            self.ddst = None

        self.ddst_projection = nn.Linear(self.emb_dim, self.rnn_output)
        self.gmil = (
            GatedMultivariateInteractionLayer(
                self.rnn_output, self.d_model // 2, self.dropout
            )
            if configs.interact
            else None
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(self.rnn_output, self.hidden, self.bias),
            nn.GELU(),
            Rearrange("B T H -> B (T H)"),
            nn.Linear(self.hidden * self.seq_len, self.pred_len, self.bias),
        )
        self.trend_residual_fusion = nn.Linear(
            2 * self.pred_len, self.pred_len, self.bias
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forecast(self, x):
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)
        batch_size, channels, time_steps = x.shape
        endo = x[:, -1:, :] if self.features == "MS" else x
        endo_dim = 1 if self.features == "MS" else channels
        endo_ci = endo.reshape(batch_size * endo_dim, 1, time_steps).squeeze(1)

        trend = self.endogenous_trend(endo_ci)

        eata_out = self.eata(endo, x)
        eata_flat = eata_out.reshape(batch_size * endo_dim, channels + 1, time_steps).permute(0, 2, 1)

        temporal_in = self.eata_projection(eata_flat)
        temporal_in = self.eata_norm(temporal_in)
        temporal_out, _ = self.temporal_encoder(temporal_in)

        if self.ddst is not None:
            ddst_out = self.ddst(eata_flat.permute(0, 2, 1))
            ddst_out = self.ddst_projection(ddst_out)
            temporal_out = temporal_out + ddst_out

        if self.gmil is not None:
            gmil_input = temporal_out.permute(0, 2, 1)
            gmil_out = self.gmil(gmil_input)
            temporal_out = temporal_out + gmil_out

        residual = self.forecast_head(temporal_out)
        final_comb = torch.cat([trend, residual], dim=1)
        final_pred = self.trend_residual_fusion(final_comb)
        final_pred = final_pred.reshape(batch_size, endo_dim, self.pred_len).permute(0, 2, 1)

        if self.features == "M":
            final_pred = self.revin.denormalize(final_pred)
        elif self.features == "MS":
            target_idx = -1
            if self.revin.affine:
                bias = self.revin.affine_bias[target_idx]
                weight = self.revin.affine_weight[target_idx]
                eps = self.revin.eps
                final_pred = (final_pred - bias) / (weight + eps * weight)

            mean = self.revin.mean[:, :, -1:]
            stdev = self.revin.stdev[:, :, -1:]
            final_pred = final_pred * stdev + mean

        return final_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
