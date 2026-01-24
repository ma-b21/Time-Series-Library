import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from layers.TD_CaA_utils import RevIN, Temp_Causal_Address, DynamicDistributionShiftTracer, GatedInterAct, SimpleDistributionShiftTracer

###TODO 
#比较RNN和TCA，因为回看seqlen时表现好，有点像RNN呢
#SimpleDistreiTracer表现差，说明project效果好，这里可以做消融实验
# new1 升维度，没变输入逻辑
# new2 没有升维度，只变了输入逻辑


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
        self.alpha = nn.Parameter(torch.zeros(self.pred_len))
        
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
            # self.stat_tracer = DynamicDistributionShiftTracer(self.emb_dim, self.k_lookback, dropout)
            #new2
            self.stat_tracer = nn.Sequential(
                DynamicDistributionShiftTracer(self.emb_dim, self.k_lookback, dropout),
                nn.Linear(self.emb_dim, self.emb_dim * 2)
            )
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
            # nn.Linear(current_dim, self.hidden, bias),
            # new2
            nn.Linear(self.emb_dim * 2, self.hidden, bias),
            Rearrange('B T H -> B (T H)'),
            nn.Linear(self.hidden * self.seq_len, self.pred_len, bias)
        )  #[B*E, P]
        
        self.corr = GatedInterAct(self.endo_dim, d_model, dropout)
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
        
        # features_list = []
        x_TCA = self.TCA(endo, x) #[B, E, F+1, T]
        x_TCA = x_TCA.reshape(B * E, 1, F+1, T)
        x_TCA = x_TCA.squeeze(1) #[B*E, F+1, T]
        x_gru_in = x_TCA.permute(0, 2, 1) # [B*E, T, F+1]
        x_seq, _ = self.sequencial(x_gru_in)  # [B*E, T, 2*(F+1)]
        # features_list.append(x_seq)

        if self.stat_tracer is not None:
            # [B*E, F+1, T] ->  [B*E, T, F+1]
            x_stat = self.stat_tracer(x_TCA) 
            #new2
            x_seq += x_stat
            # features_list.append(x_stat)
          
        if self.interact is not None:
            # x_interact = self.interact(x_TCA)
            #new2
            x_interact = self.interact(x_seq.permute(0, 2, 1)) 
            
            # features_list.append(x_interact)
            
        # emb_comb = torch.cat(features_list, dim=-1) 
    
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
        
        
        
        
        
        
        
        
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange
# from layers.TD_CaA_utils import RevIN, Temp_Causal_Address, DynamicDistributionShiftTracer, GatedInterAct, SimpleDistributionShiftTracer

# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.configs = configs
#         self.features = configs.features 
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.exo_dim = configs.enc_in
#         self.k_lookback = configs.k_lookback 
#         self.method = configs.method 
#         self.hidden = configs.hidden 
#         self.d_model = configs.d_model  # 必须保证这个够大，比如 256/512
#         self.dropout = configs.dropout
        
#         self.endo_dim = 1 if self.features == "MS" else self.exo_dim
#         self.emb_dim = self.exo_dim + 1 
        
#         # 1. 动态融合系数
#         self.alpha_param = nn.Parameter(torch.zeros(self.pred_len))
        
#         self.revin = RevIN(self.exo_dim)
        
#         # 2. 趋势项
#         self.linear_trend = nn.Linear(self.seq_len, self.pred_len) 
        
#         # 3. [核心修复] Query 增强器
#         # 旧模型靠FFT增强Query，新模型靠 Conv1d 提取局部模式增强 Query
#         # 这一步把贫瘠的 endo 变成 32 维的丰富特征
#         self.query_enhancer = nn.Sequential(
#             nn.Conv1d(self.endo_dim, 32, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Dropout(self.dropout)
#         )
        
#         # 4. [核心修复] 补回丢失的时间感
#         # 既然没有外部 Time Embedding，我们自己学一个 Positional Embedding
#         self.pos_emb = nn.Parameter(torch.randn(1, 32, self.seq_len))

#         # 5. TCA
#         # 注意：这里我们修改 TCA 的逻辑，让它接受增强后的特征
#         # 原始 Input 是 emb_dim, 但我们需要某种方式让增强特征起作用
#         # 为了不改动 TCA utils，我们在 forward 里做 trick
#         self.TCA = Temp_Causal_Address(self.endo_dim, self.exo_dim, self.k_lookback, self.dropout)
        
#         # 6. 主干投影与建模
#         # 将 TCA 找回来的信息 (emb_dim) 映射到高维 (d_model)
#         self.input_proj = nn.Linear(self.emb_dim, self.d_model)
#         self.layer_norm = nn.LayerNorm(self.d_model)
        
#         # 双向 GRU
#         self.sequencial = nn.GRU(self.d_model, self.d_model, batch_first=True, bidirectional=True)
#         self.rnn_out_dim = self.d_model * 2

#         # 7. 特殊模块 (Tracer & Interact)
#         if self.method == "Dynamic": 
#             self.stat_tracer = DynamicDistributionShiftTracer(self.emb_dim, self.k_lookback, self.dropout)
#         elif self.method == "Simple":
#             self.stat_tracer = SimpleDistributionShiftTracer(self.emb_dim, self.k_lookback, self.dropout)
#         else:
#             self.stat_tracer = None
            
#         if self.stat_tracer is not None:
#              self.tracer_proj = nn.Linear(self.emb_dim, self.rnn_out_dim)

#         if configs.interact:
#             # 严格匹配 GRU 输出维度
#             self.interact = GatedInterAct(self.rnn_out_dim, self.d_model // 2, self.dropout)
#         else:
#             self.interact = None
        
#         # 8. 回归头
#         self.Linear = nn.Sequential(
#             nn.Linear(self.rnn_out_dim, self.hidden),
#             nn.GELU(),
#             Rearrange('B T H -> B (T H)'),
#             nn.Linear(self.hidden * self.seq_len, self.pred_len)
#         )
        
#         self.corr = GatedInterAct(self.endo_dim, self.d_model, self.dropout)
#         self.linear_head = nn.Linear(2 * self.pred_len, self.pred_len)
        
#     def forecast(self, x):
#         # === 1. Preprocess ===
#         x = self.revin.normalize(x)
#         x = x.permute(0, 2, 1) # [B, F, T]
#         B, F, T = x.shape
#         endo = x[:, -1:, :] if self.features == "MS" else x
#         E = 1 if self.features == "MS" else F
#         endo_CI = endo.reshape(B * E, 1, T).squeeze(1) #[B*E, T]
        
#         # === 2. Trend ===
#         trend = self.linear_trend(endo_CI) 
        
#         # === 3. [关键步骤] 增强 Query ===
#         # 我们不直接传 endo 给 TCA，而是传一个有丰富语义的 endo_enhanced
#         # 原版 TCA 的入参是 (endo, exo)。
#         # 这里的 trick 是：TCA 内部用 endo 做 Query。
#         # 但 TCA 的第一层通常是 Conv 投影。
#         # 如果我们不动 utils，只能传原始 endo。
#         # **为了救回效果，这里必须即便传入原始endo，后续处理也要足够强**
        
#         # 正常调用 TCA 得到上下文
#         x_TCA = self.TCA(endo, x) # [B, E, F+1, T]
#         x_TCA_flat = x_TCA.reshape(B * E, F+1, T).permute(0, 2, 1) # [B*E, T, F+1]
        
#         # === 4. High-Dim Projection (容量扩充) ===
#         # 这里至关重要：把贫瘠的 F+1 维特征，拉升到 d_model
#         x_deep = self.input_proj(x_TCA_flat) # [B*E, T, d_model]
#         x_deep = self.layer_norm(x_deep)     # 加 LayerNorm 稳定梯度
        
#         # === 5. GRU Modeling ===
#         # x_backbone: [B*E, T, 2 * d_model]
#         x_backbone, _ = self.sequencial(x_deep) 
        
#         # === 6. Feature Fusion (Tracer) ===
#         if self.stat_tracer is not None:
#             # Tracer 作用在原始物理量上
#             x_stat_raw = self.stat_tracer(x_TCA_flat.permute(0, 2, 1)) # -> [T, F+1]
#             x_stat_proj = self.tracer_proj(x_stat_raw) # -> [T, 2*d_model]
#             x_backbone = x_backbone + x_stat_proj # Residual
          
#         # === 7. Interact (修复维度 bug) ===
#         if self.interact is not None:
#             # x_backbone: [B*E, T(96), C(512)]
#             # GatedInterAct 需要 [B, C, T]
#             inp_interact = x_backbone.permute(0, 2, 1) # [B*E, 512, 96]
            
#             # Out: [B*E, 512, 96] (内部会转回去又转回来)
#             out_interact = self.interact(inp_interact) 
            
#             # 转回 [B*E, 96, 512] 以便相加
#             out_interact = out_interact
            
#             x_backbone = x_backbone + out_interact
            
#         # === 8. Regression Head ===
#         residual = self.Linear(x_backbone)   # [B*E, P]
        
#         final_comb = torch.cat([trend, residual], dim=1)
        
#         # Weighting
#         alpha = torch.sigmoid(self.alpha_param)
#         final_pred = (1 - alpha) * self.linear_head(final_comb) + alpha * trend
        
#         # === 9. Post-Process ===
#         final_pred = final_pred.reshape(B, E, self.pred_len).permute(0, 2, 1) # [B, P, E]
        
#         if self.features == 'M':
#             final_pred = self.revin.denormalize(final_pred)
#         elif self.features == 'MS':
#             target_idx = -1 
#             if self.revin.affine:
#                 bias = self.revin.affine_bias[target_idx]
#                 weight = self.revin.affine_weight[target_idx]
#                 eps = self.revin.eps
#                 final_pred = (final_pred - bias) / (weight + eps * weight)
#             mean = self.revin.mean[:, :, target_idx:target_idx+1]
#             stdev = self.revin.stdev[:, :, target_idx:target_idx+1]
#             final_pred = final_pred * stdev + mean
        
#         # === 10. Correlation Correction ===
#         # final_pred: [B, P, E]
#         # self.corr expects Channel First: [B, E, P]
#         # self.corr outputs Time First: [B, P, E]
#         correlation = self.corr(final_pred.permute(0, 2, 1))
        
#         return final_pred + correlation

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         return self.forecast(x_enc)
            
