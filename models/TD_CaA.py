import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from layers.TD_CaA_utils import RevIN, Temp_Causal_Address, DynamicDistributionShiftTracer, GatedInterAct, SimpleDistributionShiftTracer



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
        
        self.linear_head = nn.Linear(2 * self.pred_len, self.pred_len, self.bias)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用 Xavier 初始化解决线性层敏感问题
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            # 卷积层通常适合 Kaiming 初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.GRU):
            # GRU/LSTM 最需要正交初始化 (Orthogonal Initialization)
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data) # 关键！正交初始化对 RNN 收敛极其重要
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
        
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
        
        # 输出x的一个batch，以及x_TCA对应的batch
        # print(x[0][:,:20].detach().cpu().numpy().T)
        x_TCA = self.TCA(endo, x) #[B, E, F+1, T]
        # print(x_TCA.squeeze(1)[0][:,:20].detach().cpu().numpy().T)
        # raise ValueError("Debugging: Check the output of TCA layer.")
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

            mean = self.revin.mean[:, :, -1:]
            stdev = self.revin.stdev[:, :, -1:]
            
            final_pred = final_pred * stdev + mean
        
        return final_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
        