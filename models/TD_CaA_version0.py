import torch
import torch.nn as nn
from layers.TD_CaA_layers import RevIN, Statistic_Features, Lite_Temp_Causal_Address, Full_Temp_Causal_Address, \
    VectorizedCausalProjection, GatedExoProjector


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.features = configs.features # 'M', 'S', 'MS'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.k_lookback = configs.k_lookback 
        self.moving_avg = configs.moving_avg
        self.fft = configs.fft
        
        # Hyperparameters
        self.m_expand = 64
        hd = configs.d_model
        dropout = configs.dropout
        time_dim = 8
        
        self.revin = RevIN(
            self.enc_in if self.features != 'S' else 1
        )
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self.is_lite_mode = (self.features == 'S')

        if self.is_lite_mode:
            # === Lite Mode Components ===
            self.raw_proj = nn.Conv1d(1, self.m_expand, kernel_size=3, padding=1)
            
            self.tca_lite = Lite_Temp_Causal_Address(
                d_model=self.m_expand, 
                k_lookback=self.k_lookback, 
                dropout=dropout
            )
            
            self.lite_fusion = nn.Sequential(
                nn.Linear(self.m_expand, self.m_expand),
                nn.GELU()
            )
            # 单流 BiGRU
            self.backbone_rnn = nn.GRU(self.m_expand, self.m_expand, batch_first=True, bidirectional=True)
            self.head = nn.Linear(self.m_expand * 2 * self.seq_len, self.pred_len)
            
        else:
            # === Full Mode Components ===
            self.time_proj = nn.Sequential(nn.Linear(4, time_dim), nn.ReLU(), nn.Dropout(dropout))
            
            self.tca_full = Full_Temp_Causal_Address(
                endo_dim=1,
                exo_dim=self.m_expand, 
                time_dim=time_dim, 
                k_lookback=self.k_lookback,
                roll_win=self.moving_avg, 
                FFT=self.fft, 
                dropout=dropout,
                seq_len=self.seq_len
            )
            
            # 双流 Backbone
            self.rba_layer = VectorizedCausalProjection(self.m_expand, self.m_expand, self.k_lookback, dropout)
            self.backbone_rnn = nn.GRU(self.m_expand, self.m_expand, batch_first=True, bidirectional=True)
            self.mix = nn.Linear(self.m_expand * 3, hd)
            self.head = nn.Linear(hd * self.seq_len, self.pred_len) # Seq head
            self.output_proj = nn.Linear(2 * self.pred_len, self.pred_len)
            
            if self.features == 'M':
                self.exo_adapter = GatedExoProjector(n_vars=self.enc_in, d_model=64, dropout=dropout)
            else:
                self.exo_adapter = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, M_in = x_enc.shape
        x_norm = self.revin.normalize(x_enc)
        
        if self.is_lite_mode:
            if M_in > 1:
                x_in_raw = x_norm[:, :, -1:]
            else:
                x_in_raw = x_norm
            
            x_in = x_in_raw.permute(0, 2, 1)  #[B, 1, T]
            
            trend = self.linear_trend(x_in.squeeze(1))  #[B, P]
            
            feat_raw = self.raw_proj(x_in)
            feat_tca = self.tca_lite(x_in)
            
            feat_fused = feat_raw + feat_tca
            feat_fused = self.lite_fusion(feat_fused.permute(0, 2, 1))
            
            enc_out, _ = self.backbone_rnn(feat_fused)
            
            seasonal = self.head(enc_out.reshape(B, -1))
            final = seasonal + trend
            
            final = final.unsqueeze(-1)
            return self.revin.denormalize(final)
        else:
            aug_endo, exo_emb = None, None
            x_trend_in = None
            curr_bs = 0
            
            if self.features == 'MS':
                curr_bs = B
                
                endo_data = x_norm[:, :, -1:]
                endo_flat = endo_data.permute(0, 2, 1)
                x_trend_in = endo_flat.squeeze(1)
                
                if M_in > 1:
                    exo_data = x_norm[:, :, :-1]
                    exo_flat = exo_data.permute(0, 2, 1).reshape(B * (M_in-1), 1, T)
                else:
                    exo_flat = None
                
                # Time Embs
                time_emb_endo = None
                time_emb_exo = None
                if x_mark_enc is not None:
                    xm_endo = x_mark_enc[:, :, :4]
                    time_emb_endo = self.time_proj(xm_endo).permute(0, 2, 1)
                    
                    if exo_flat is not None:
                        xm_exo = xm_endo.unsqueeze(1).repeat(1, M_in-1, 1, 1).reshape(B*(M_in-1), T, -1)
                        time_emb_exo = self.time_proj(xm_exo).permute(0, 2, 1)
                
                aug_endo, exo_emb = self.tca_full(
                    endo=endo_flat, x_mark_enc=time_emb_endo,
                    exo=exo_flat, x_mark_exo=time_emb_exo
                )

            else: # Features == 'M'
                curr_bs = B * M_in
                x_flat = x_norm.permute(0, 2, 1).reshape(B * M_in, 1, T)
                x_trend_in = x_flat.squeeze(1)  #[BM, T]
                
                time_emb = None
                if x_mark_enc is not None:  # time emb
                    xm = x_mark_enc[:, :, :4].unsqueeze(1).repeat(1, M_in, 1, 1).reshape(B*M_in, T, -1)
                    time_emb = self.time_proj(xm).permute(0, 2, 1)
                
                # Self-Addressing
                aug_endo, exo_emb = self.tca_full(
                    endo=x_flat, x_mark_enc=time_emb,
                    exo=None
                )

            fused_feat = aug_endo + exo_emb
            
            attn_out = self.rba_layer(query_feat=fused_feat, context_block=fused_feat)
            rnn_out, _ = self.backbone_rnn(fused_feat.permute(0, 2, 1))
            
            combined = torch.cat([attn_out, rnn_out.permute(0,2,1)], dim=1)
            code = self.mix(combined.permute(0, 2, 1)).reshape(curr_bs, -1)
            seasonal_pred = self.head(code)
            
            trend_pred = self.linear_trend(x_trend_in)
            
            final = torch.cat([trend_pred, seasonal_pred], dim=1)
            final_ci_pred = self.output_proj(final) + trend_pred
            
            if self.features == 'MS':
                pred_final = final_ci_pred.unsqueeze(-1)
                
                target_mean = self.revin.mean[:, :, -1:]
                target_std = self.revin.stdev[:, :, -1:]
                if self.revin.affine:
                    target_weight = self.revin.affine_weight[-1].view(1, 1, 1)
                    target_bias = self.revin.affine_bias[-1].view(1, 1, 1)
                    pred_final = (pred_final - target_bias) / (target_weight + self.revin.eps * target_weight)
                return pred_final * target_std + target_mean
            
            else: # 'M'
                pred_out = final_ci_pred.reshape(B, M_in, self.pred_len).permute(0, 2, 1)
                if self.exo_adapter is not None:
                    pred_out = pred_out + self.exo_adapter(pred_out)
                return self.revin.denormalize(pred_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
