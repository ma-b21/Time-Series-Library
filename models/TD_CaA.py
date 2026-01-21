import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from layers.TD_CaA_utils import RevIN, Temp_Causal_Address, DynamicDistributionShiftTracer, GatedInterAct, SimpleDistributionShiftTracer

###TODO 
#比较RNN和TCA，因为回看seqlen时表现好，有点像RNN呢
#SimpleDistreiTracer表现差，说明project效果好，这里可以做消融实验


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.features = configs.features # 'M','MS'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.exo_dim = configs.enc_in
        self.k_lookback = configs.k_lookback 
        self.method = configs.method  # "Dynamic", "Simple"
        self.hidden = configs.hidden
        self.endo_dim = 1 if self.features == "MS" else self.exo_dim
        self.emb_dim = self.exo_dim + 1  #编码后每一个内生变量有exo_dim个外生编码以及其自身
        
        # Hyperparameters
        d_model = configs.d_model
        dropout = configs.dropout
        bias = configs.bias
        
        self.revin = RevIN(self.exo_dim)
        # backbone
        ## embedding
        
        ### trend trace
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len) #[B*E, P]
        
        ### channel indepence
        self.TCA = Temp_Causal_Address(self.endo_dim, self.exo_dim, self.k_lookback, dropout)  #[B, E, F+1, T]
        self.sequencial = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)  #[B*E, T, 2*(F+1)]
        if self.method == "Dynamic": 
            self.stat_tracer = DynamicDistributionShiftTracer(self.emb_dim, self.k_lookback, dropout)
        elif self.method == "Simple":
            self.stat_tracer = SimpleDistributionShiftTracer(self.emb_dim, self.k_lookback, dropout)
        else:
            self.stat_tracer = None  # 轻量级 
            
        ### channel Interactive
        self.interact = GatedInterAct(self.emb_dim, d_model, dropout) if configs.interact else None #[B*E, T, F+1]
        
        ## Simply Linear Sequential  Layer
        current_dim = 2 * self.emb_dim 
        if self.stat_tracer is not None:
            current_dim += self.emb_dim # stat_tracer 输出维度是 emb_dim
        if self.interact is not None:
            current_dim += self.emb_dim # interact 输出维度是 emb_dim
        
        self.Linear = nn.Sequential(
            nn.Linear(current_dim, self.hidden, bias),
            Rearrange('B T H -> B (T H)'),
            nn.Linear(self.hidden * self.seq_len, self.pred_len, bias)
        )  #[B*E, P]
        
        self.linear_head = nn.Linear(2 * self.pred_len, self.pred_len, bias)
        
    def forecast(self, x):
        # preprocess
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)
        B, F, T = x.shape
        endo = x[:, -1:, :] if self.features == "MS" else x
        E = 1 if self.features == "MS" else F
        endo_CI = endo.reshape(B * E, 1, T)
        endo_CI = endo_CI.squeeze(1) #[B*E, T]    
        
        # backbone
        trend = self.linear_trend(endo_CI) #[B*E, P]
        
        features_list = []
        x_TCA = self.TCA(endo, x) #[B, E, F+1, T]
        x_TCA = x_TCA.reshape(B * E, 1, F+1, T)
        x_TCA = x_TCA.squeeze(1) #[B*E, F+1, T]
        x_gru_in = x_TCA.permute(0, 2, 1) # [B*E, T, F+1]
        x_seq, _ = self.sequencial(x_gru_in)  # [B*E, T, 2*(F+1)]
        features_list.append(x_seq)

        if self.stat_tracer is not None:
            # [B*E, F+1, T] ->  [B*E, T, F+1]
            x_stat = self.stat_tracer(x_TCA) 
            features_list.append(x_stat)
          
        if self.interact is not None:
            x_interact = self.interact(x_TCA) 
            features_list.append(x_interact)
            
        emb_comb = torch.cat(features_list, dim=-1) 
    
        residual = self.Linear(emb_comb)   # [B*E, T, Middle_Dim] -> [B*E, P]
        
        final_comb = torch.cat([trend, residual], dim=1)
        
        final_pred = self.linear_head(final_comb)
        
        # [B*E, P] -> [B, E, P]
        final_pred = final_pred.reshape(B, E, self.pred_len)
        
        # Denormalize
        final_pred = final_pred.permute(0, 2, 1)
        if self.features == 'M':
            final_pred = self.revin.denormalize(final_pred)
            
        elif self.features == 'MS':
            target_idx = -1 
            if self.revin.affine:
                bias = self.revin.affine_bias[target_idx]
                weight = self.revin.affine_weight[target_idx]
                eps = self.revin.eps
                final_pred = (final_pred - bias) / (weight + eps * weight)
            
            # 获取最后一个特征的统计量 [1, 1, F] -> 切片 -> [1, 1, 1]
            mean = self.revin.mean[:, :, target_idx:target_idx+1]
            stdev = self.revin.stdev[:, :, target_idx:target_idx+1]
            
            final_pred = final_pred * stdev + mean
        
        return final_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
        
        
        
        
        
        
        
        
        
