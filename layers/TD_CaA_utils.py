import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def _get_statistics(self, x):
        dim2reduce = tuple(range(x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        return self.mean, self.stdev
    def normalize(self, x):
        mean, stdev = self._get_statistics(x)
        x = (x - mean) / stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    def denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.affine_weight)
        x = x * self.stdev + self.mean
        return x


class Temp_Causal_Address(nn.Module):
    def __init__(self, endo_dim, exo_dim, k_lookback, dropout=0.1):
        super().__init__()
        self.endo_dim = endo_dim  # E
        self.exo_dim = exo_dim    # F
        self.k_lookback = k_lookback
        self.dropout = dropout
        
        # 核心投影层：将每个endo变量投影为F个特征，这F个特征将充当Query
        # 输入: [B, E, T] -> 输出: [B, E*F, T]
        # 使用 groups=endo_dim 保证第i个endo只生成第i组的F个Query，保持独立性
        self.endo_expander = nn.Sequential(
            nn.Conv1d(self.endo_dim, self.exo_dim * self.endo_dim, kernel_size=1, groups=self.endo_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, endo, exo):
        """
        endo: [B, E, T]
        exo:  [B, F, T]
        """
        B, E, T = endo.shape
        F_dim = exo.shape[1] # F
        
        # [B, E, T] -> [B, E*F, T]
        endo_expanded = self.endo_expander(endo)
     
        # [B, E*F, T] -> [B, E, F, T] -> [B, E, F, T, 1]
        query = endo_expanded.reshape(B, E, F_dim, T).unsqueeze(-1)
        
        # [B, F, T] -> [B, F, T+K-1]
        exo_padded = F.pad(exo, (self.k_lookback-1, 0))
        
        # [B, F, T+K-1] -> [B, F, T, K]
        exo_windows = exo_padded.unfold(dimension=2, size=self.k_lookback, step=1)
        
        # [B, 1, F, T, K]
        key_value = exo_windows.unsqueeze(1)
        
        # Query:      [B, E, F, T, 1]
        # Key/Value:  [B, 1, F, T, K]
        # ->  [B, E, F, T, K]
        product = query * key_value 
        
        weights = F.softmax(torch.abs(product), dim=-1) # [B, E, F, T, K]
        sign_direction = torch.sign(product)
        # [B, E, F, T, K] -> [B, E, F, T]
        exo_encoded = (weights * sign_direction * key_value).sum(dim=-1)
        
        # endo: [B, E, T] -> [B, E, 1, T]
        endo_original = endo.unsqueeze(2)
        
        #  [B, E, F+1, T]
        combined = torch.cat([exo_encoded, endo_original], dim=2)
        
        return combined

# class DynamicDistributionShiftTracer(nn.Module):
#     def __init__(self, query_dim, key_dim, k_lookback=16, dropout=0.3):
#         super().__init__()
#         self.k = k_lookback
#         self.gen_input_dim = query_dim * 6 
#         self.out_channels = key_dim * k_lookback 
#         self.kernel_generator = nn.Conv1d(self.gen_input_dim, self.out_channels, kernel_size=1)
#         nn.init.xavier_normal_(self.kernel_generator.weight, gain=0.01)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, query_feat, context_block):
#         BE, F, T = query_feat.shape
#         K = self.k
#         g_mean = torch.mean(query_feat, dim=2, keepdim=True).expand(-1, -1, T)
#         x_squared = torch.pow(query_feat, 2)
#         avg_x = F.avg_pool1d(query_feat, kernel_size=5, stride=1, padding=2)
#         avg_x2 = F.avg_pool1d(x_squared, kernel_size=5, stride=1, padding=2)
#         rolling_var = F.relu(avg_x2 - torch.pow(avg_x, 2)) + 1e-6
#         rolling_std = torch.sqrt(rolling_var)
#         third_moment = torch.pow(query_feat - avg_x, 3)
#         rolling_max = F.max_pool1d(query_feat, kernel_size=5, stride=1, padding=2)
#         rolling_min = -F.max_pool1d(-query_feat, kernel_size=5, stride=1, padding=2)
        
#         gen_in = torch.cat([self.dropout(query_feat), g_mean, rolling_std, third_moment, rolling_max, rolling_min], dim=1)
#         kernels = self.kernel_generator(gen_in).view(BE, F, K, T)   # 对每个时间点生成一个前看K布的权重向量
#         ctx_padded = F.pad(context_block, (K-1, 0))   # [BE, F+1, T+k-1]
#         ctx_windows = ctx_padded.unfold(dimension=2, size=K, step=1).permute(0, 1, 3, 2) # [BE, F+1, K, T]
#         attn_weights = F.softmax(torch.abs(kernels), dim=2)
#         y_conservative = (ctx_windows * attn_weights * torch.sign(kernels)).sum(dim=2)
#         return y_conservative  #[BE, F+1, T]
    
class DynamicDistributionShiftTracer(nn.Module):
    def __init__(self, n_channels, k_lookback=16, dropout=0.3):
        super().__init__()
        self.k = k_lookback
        self.n_channels = n_channels
        
        # 输入统计特征有6种 (mean, std, x^3, max, min, raw)
        # 我们希望对每个channel独立处理，所以这里 Input Dim 设为 6 (每组6个统计量)
        # Output Dim 设为 K (每个时间步生成K个权重)
        # 使用 groups=n_channels 或者是对最后一维操作的 Linear，这里用 Conv1d + groups 实现高效并行
        
        # 这种写法：Input = F * 6, Output = F * K, Groups = F
        # 意味着：第 i 组的 6 个输入由第 i 组的卷积核处理，生成第 i 组的 K 个输出
        self.kernel_generator = nn.Conv1d(
            in_channels=n_channels * 6, 
            out_channels=n_channels * k_lookback, 
            kernel_size=1, 
            groups=n_channels  # <--- 关键！保持通道独立性
        )
        
        # 初始化非常重要，保证初始状态下稍微偏向于“当前时刻”或者“均匀分布”
        nn.init.xavier_normal_(self.kernel_generator.weight, gain=0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, F, T]  (同时作为 Query 和 Context)
        """
        B, C, T = x.shape
        K = self.k
        
        # --- 1. 提取统计特征 (Statistical Features) ---
        # 均值
        g_mean = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, T)
        
        # 滚动方差/标准差 (Rolling Std)
        x_squared = torch.pow(x, 2)
        avg_x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        avg_x2 = F.avg_pool1d(x_squared, kernel_size=5, stride=1, padding=2)
        rolling_var = torch.clamp(avg_x2 - torch.pow(avg_x, 2), min=1e-6)
        rolling_std = torch.sqrt(rolling_var)
        
        # 三阶矩 (Skewness proxy)
        third_moment = torch.pow(x - avg_x, 3)
        
        # 滚动极值
        rolling_max = F.max_pool1d(x, kernel_size=5, stride=1, padding=2)
        rolling_min = -F.max_pool1d(-x, kernel_size=5, stride=1, padding=2)
        
        # 原始特征 (加入 dropout)
        x_drop = self.dropout(x)
        
        # 拼接: [B, C*6, T]
        # 注意顺序：为了配合 Conv1d 的 groups=C，
        # 我们需要数据排列成: [Ch1_feat1...feat6, Ch2_feat1...feat6, ...]
        # 但简单的 torch.cat(dim=1) 得到的是 [Ch1...ChF_feat1, Ch1...ChF_feat2...]
        # 所以由 dim=1 拼接是不够的，我们需要对齐通过 groups 的逻辑
    
        
        stats_list = [x_drop, g_mean, rolling_std, third_moment, rolling_max, rolling_min]
        # Stack 起来: [B, C, 6, T]
        stats_stack = torch.stack(stats_list, dim=2) 
        
        # Flatten 回 Conv1d 需要的格式: [B, C*6, T]
        # 这样排列顺序就是: F1的6个特征, F2的6个特征... 
        # 正好对应 groups=C 的卷积核顺序
        gen_in = stats_stack.reshape(B, C * 6, T)

        # --- 2. 生成动态权重 (Dynamic Kernel Generation) ---
        # Conv1d output: [B, C*K, T] (由于使用了groups=F)
        kernels_flat = self.kernel_generator(gen_in)
        
        # Reshape: [B, F, K, T]
        kernels = kernels_flat.reshape(B, C, K, T)
        
        # --- 3. 提取历史窗口 (Context Unfolding) ---
        # [B, F, T] -> [B, F, T+K-1]
        x_padded = F.pad(x, (K-1, 0))
        
        # [B, F, T+K-1] -> [B, F, T, K] -> [B, F, K, T]
        x_windows = x_padded.unfold(dimension=2, size=K, step=1).permute(0, 1, 3, 2)
        
        # --- 4. 加权聚合 ---
        # Softmax over K dimension
        attn_weights = F.softmax(torch.abs(kernels), dim=2) # [B, C, K, T]
        
        # Apply weights and sum over K
        # sign(kernels) 允许模型捕捉负相关性 (例如反转趋势)
        out = (x_windows * attn_weights * torch.sign(kernels)).sum(dim=2) 
        
        return out.permute(0, 2, 1) # [B, T, F]
    
class SimpleDistributionShiftTracer(nn.Module):
    def __init__(self, n_channels, k_lookback=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_channels, 
            out_channels=n_channels, 
            kernel_size=(6, k_lookback), 
            padding=(0, k_lookback // 2), # Padding 保证时间维度 T 不变 (假设 k是奇数)
            groups=n_channels  #保证变量之间互不干扰
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU() 
        # 或者用 Tanh 如果你想做一种 scale 调整

    def forward(self, x):
        """
        x: [B, F, T]
        """
        B, C, T = x.shape 
        
        # --- 1. 计算这一堆统计特征 (同前) ---
        g_mean = torch.mean(x, dim=2, keepdim=True).expand(-1, -1, T)
        x_squared = torch.pow(x, 2)
        avg_x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        avg_x2 = F.avg_pool1d(x_squared, kernel_size=5, stride=1, padding=2)
        rolling_var = torch.clamp(avg_x2 - torch.pow(avg_x, 2), min=1e-6)
        rolling_std = torch.sqrt(rolling_var)
        third_moment = torch.pow(x - avg_x, 3)
        rolling_max = F.max_pool1d(x, kernel_size=5, stride=1, padding=2)
        rolling_min = -F.max_pool1d(-x, kernel_size=5, stride=1, padding=2)
        x_drop = self.dropout(x)
        
        # Stack 起来 -> [B, F, 6, T]
        # B: Batch
        # F: Channel (Conv2d 的 Input Channel)
        # 6: Height (Conv2d 的 H)
        # T: Width (Conv2d 的 W)
        gen_in = torch.stack([x_drop, g_mean, rolling_std, third_moment, rolling_max, rolling_min], dim=2)
      
        # 卷积核 (6, k) 会把 Height 维度的 6 直接卷成 1
        # 时间维度根据 padding 保持为 T
        # 输出: [B, F, 1, T]
        out = self.conv(gen_in)
        out = out[:, :, :, :T]
        out = self.activation(out).squeeze(2)
        
        return (x + out).permute(0, 2, 1)
    

class GatedInterAct(nn.Module):
    def __init__(self, n_channels, d_model=32, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.var_projector = nn.Sequential(nn.Linear(n_channels, d_model), nn.LeakyReLU(), nn.Dropout(dropout))
        self.out_proj = nn.Linear(d_model, n_channels)
        self.channel_gate = nn.Parameter(torch.zeros(n_channels)) 
    def forward(self, pred_ci):
        pred_ci = pred_ci.permute(0, 2, 1)
        B, T, M = pred_ci.shape
        x_flat = pred_ci.reshape(B*T, M) 
        global_ctx = self.var_projector(x_flat) 
        correction_flat = self.out_proj(global_ctx)
        correction = correction_flat.reshape(B, T, M)
        gate = torch.tanh(self.channel_gate).view(1, 1, M)
        return correction * gate

