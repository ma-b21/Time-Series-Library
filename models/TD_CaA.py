import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from layers.TD_CaA_utils import RevIN, Temp_Causal_Address, DynamicDistributionShiftTracer, GatedInterAct, SimpleDistributionShiftTracer

###TODO 
#比较RNN和TCA，因为回看seqlen时表现好，有点像RNN呢
#SimpleDistreiTracer表现差，说明project效果好，这里可以做消融实验
# new1 升维度，没变输入逻辑
# new2 没有升维度，只变了输入逻辑
# new3 new2+Gelu Yes
# new4 new3 + alpha NO
# new5 new3 + 1-alpha/ alpha  NO
# new6 new3+ corr NO



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
        self.alpha_param = nn.Parameter(torch.zeros(self.pred_len))
        
        
        # Hyperparameters
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.bias = configs.bias
        self.rnn_output = 2 * self.d_model
        
        self.revin = RevIN(self.exo_dim)
        # backbone
        ## embedding
        
        ### trend trace
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len) #[B*E, P]
        
        ### channel indepence
        self.TCA = Temp_Causal_Address(self.endo_dim, self.exo_dim, self.k_lookback, self.dropout)  #[B, E, F+1, T]
        self.linear_deep = nn.Linear(self.emb_dim, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        
        self.sequencial = nn.GRU(self.d_model, self.d_model, batch_first=True, bidirectional=True)  #[B*E, T, 2*D]
        if self.method == "Dynamic": 
            self.stat_tracer = DynamicDistributionShiftTracer(self.emb_dim, self.k_lookback, self.dropout)
        elif self.method == "Simple":
            self.stat_tracer = nn.Sequential(
                SimpleDistributionShiftTracer(self.emb_dim, self.k_lookback, self.dropout),
                nn.Linear(self.emb_dim, self.rnn_output)
            )
        else:
            self.stat_tracer = None  # 轻量级 
            
        self.stat_linear = nn.Linear(self.emb_dim, self.rnn_output)
        ### channel Interactive
        self.interact = GatedInterAct(self.rnn_output, self.d_model // 2, self.dropout) if configs.interact else None #[B*E, T, F+1]
        
        self.Linear = nn.Sequential(
            nn.Linear(self.rnn_output, self.hidden, self.bias),
            nn.GELU(),
            Rearrange('B T H -> B (T H)'),
            nn.Linear(self.hidden * self.seq_len, self.pred_len, self.bias)
        )  #[B*E, P]
        
        
        self.corr = GatedInterAct(self.endo_dim, self.d_model, self.dropout)
        self.linear_head = nn.Linear(2 * self.pred_len, self.pred_len, self.bias)
        
    def forecast(self, x):
        # preprocess
        x = self.revin.normalize(x)
        x = x.permute(0, 2, 1)
        B, F, T = x.shape
        endo = x[:, -1:, :] if self.features == "MS" else x
        E = 1 if self.features == "MS" else F
        endo_CI = endo.reshape(B * E, 1, T).squeeze(1) #[B*E, T]    
        
        # backbone
        trend = self.linear_trend(endo_CI) #[B*E, P]
        
        x_TCA = self.TCA(endo, x) #[B, E, F+1, T]
        x_TCA_flat = x_TCA.reshape(B * E, F+1, T).permute(0, 2, 1)
        # new*
        x_gru_in = self.linear_deep(x_TCA_flat)
        x_gru_in = self.layernorm(x_gru_in)
        x_seq, _ = self.sequencial(x_gru_in)  # [B*E, T, 2*D]

        if self.stat_tracer is not None:
            # [B*E, F+1, T] ->  [B*E, T, 2*D]
            x_stat = self.stat_tracer(x_TCA_flat.permute(0, 2, 1))
            x_stat = self.stat_linear(x_stat) 
            x_seq = x_seq + x_stat
          
        if self.interact is not None:
            inter_input = x_seq.permute(0, 2, 1)
            x_interact = self.interact(inter_input) 
            x_seq = x_seq + x_interact
            
    
         # [B*E, T, Middle_Dim] -> [B*E, P]
        residual = self.Linear(x_seq)
        
        
        final_comb = torch.cat([trend, residual], dim=1)

        # alpha = torch.sigmoid(self.alpha_param)
        # final_pred = (1 - alpha) * self.linear_head(final_comb) + alpha * trend
        final_pred = self.linear_head(final_comb)
        
        # [B*E, P] -> [B, P, E]
        final_pred = final_pred.reshape(B, E, self.pred_len).permute(0, 2, 1)
        
        # Denormalize
        if self.features == 'M':
            final_pred = self.revin.denormalize(final_pred)
            
        elif self.features == 'MS':
            target_idx = -1 
            if self.revin.affine:
                bias = self.revin.affine_bias[target_idx]
                weight = self.revin.affine_weight[target_idx]
                eps = self.revin.eps
                final_pred = (final_pred - bias) / (weight + eps * weight)
            
            # 获取最后一个特征的统计量 [1, 1, F]-> [1, 1, 1]
            mean = self.revin.mean[:, :, target_idx:target_idx+1]
            stdev = self.revin.stdev[:, :, target_idx:target_idx+1]
            
            final_pred = final_pred * stdev + mean
        
        correlation = self.corr(final_pred.permute(0, 2, 1))
        
        return final_pred + correlation

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
        