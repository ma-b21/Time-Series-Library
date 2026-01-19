# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class RevIN(nn.Module):
#     def __init__(self, num_features: int, eps=1e-5, affine=True):
#         super(RevIN, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.affine = affine
#         if self.affine:
#             self._init_params()

#     def _init_params(self):
#         self.affine_weight = nn.Parameter(torch.ones(self.num_features))
#         self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

#     def _get_statistics(self, x):
#         dim2reduce = tuple(range(x.ndim - 1))  
#         self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
#         self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
#         return self.mean, self.stdev

#     def normalize(self, x):
#         mean, stdev = self._get_statistics(x)
#         x = (x - mean) / stdev
#         if self.affine:
#             x = x * self.affine_weight + self.affine_bias
#         return x

#     def denormalize(self, x):
#         if self.affine:
#             x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.affine_weight)
#         x = x * self.stdev + self.mean
#         return x
    

# class Statistic_Features(nn.Module):
    
#     def __init__(self, time_dim, dropout, FFT= 1, roll_win=10):
#         """
#         generate 4 statisric features and time_dim time emcoded features
#         :param time_dim: ooutput of time information emcoded features
#         :param roll_win: statistic features rolling windows
#         """
#         super().__init__()
#         self.roll_win = roll_win
#         self.FFT = True if FFT == 1 else False
#         self.time_proj = nn.Sequential(
#             nn.Linear(4, time_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#     def get_statistic_feature(self, data, x_mark_enc):
#         """
#         data: [B*E, 1, T]
#         x_mark_enc: [B, T, 4]
#         return: statistic features with row input data [B, 5*E + time_dim, T]
#         """
#         B, E, T = data.size()
#         global_mean = torch.mean(data, dim=2, keepdim=True).expand(-1, -1, T)
        
#         data_sqr = data.pow(2)
#         padding = (self.roll_win - 1) // 2
        
#         avg_data = F.avg_pool1d(data, kernel_size=self.roll_win, stride=1, padding=padding)
#         avg_data2 = F.avg_pool1d(data_sqr, kernel_size=self.roll_win, stride=1, padding=padding)
        
#         roll_var = F.relu(avg_data2 - avg_data.pow(2)) + 1e-6
#         roll_std = torch.sqrt(roll_var)
        
#         roll_max = F.max_pool1d(data, kernel_size=self.roll_win, stride=1, padding=padding)
#         roll_min = -F.max_pool1d(-data, kernel_size=self.roll_win, stride=1, padding=padding)
        
#         if x_mark_enc is not None:
#             x_mark_enc = x_mark_enc.transpose(1,2)
#             time_feat = self.time_proj(x_mark_enc) #[B, T, time_dim]
#             time_feat = time_feat.permute(0, 2, 1) #[B, time_dim, T]
#         else:
#             time_feat = torch.zeros(B, self.time_dim, T, device=data.device)
                    
#         # --- 关键修改: FFT 部分 ---
#         if self.FFT:
#             # 1. 使用 fft 而不是 rfft，结果长度保持为 T
#             fft_complex = torch.fft.fft(data, n=T, dim=-1) # [B, 1, T] 这是一个复数张量
            
#             # 2. 必须拆分 实部 和 虚部，才能作为 Float 张量拼接
#             fft_real = fft_complex.real # [B, 1, T]
#             fft_imag = fft_complex.imag # [B, 1, T]
            
#             # 3. 将实虚部拼好，作为 FFT 特征
#             fft_feat = torch.cat([fft_real, fft_imag], dim=1) # [B, 2, T]
#         else:
#             fft_feat = None
        
#         # 拼接列表
#         feature_list = [data, global_mean, roll_std, roll_max, roll_min, time_feat]
#         if self.FFT:
#             feature_list.append(fft_feat)

#         stat_features = torch.cat(feature_list, dim=1) if self.FFT else torch.cat([data, global_mean, roll_std, roll_max, roll_min, time_feat], dim=1)

#         return stat_features
    
    

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
    

class Statistic_Features(nn.Module):
    def __init__(self, time_dim, dropout, FFT=1, roll_win=10, seq_len=96):
        """
        Modified to use robust Spectrum Extraction instead of raw Real/Imag concatenation
        """
        super().__init__()
        self.roll_win = roll_win
        self.FFT = True if FFT == 1 else False
        self.time_proj = nn.Sequential(
            nn.Linear(4, time_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # [NEW] 借鉴 Framework 2 的频谱处理方式
        # 将频域信息压缩到一个固定的特征维度，而不是直接拼接原始复数部分
        if self.FFT:
            self.freq_dim = 8 # 设定一个合理的频域特征维度
            fft_len = seq_len // 2 + 1
            self.freq_proj = nn.Linear(fft_len, self.freq_dim)
        else:
            self.freq_dim = 0
            
        
    def get_statistic_feature(self, data, x_mark_enc):
        """
        data: [B*E, 1, T]
        x_mark_enc: [B, T, 4] -> 需要处理成 [B*E, T, 4] 在外部或者内部
        return: statistic features [B*E, 5 + freq_dim + time_dim, T]
        """
        B_E, C, T = data.size()
        
        # 1. 基础统计量
        global_mean = torch.mean(data, dim=2, keepdim=True).expand(-1, -1, T)
        
        data_sqr = data.pow(2)
        padding = (self.roll_win - 1) // 2
        
        avg_data = F.avg_pool1d(data, kernel_size=self.roll_win, stride=1, padding=padding)
        avg_data2 = F.avg_pool1d(data_sqr, kernel_size=self.roll_win, stride=1, padding=padding)
        
        roll_var = F.relu(avg_data2 - avg_data.pow(2)) + 1e-6
        roll_std = torch.sqrt(roll_var)
        
        roll_max = F.max_pool1d(data, kernel_size=self.roll_win, stride=1, padding=padding)
        roll_min = -F.max_pool1d(-data, kernel_size=self.roll_win, stride=1, padding=padding)
        
        # 2. 时间特征投影
        if x_mark_enc is not None:
            # 假设输入已经是 [B*E, T, 4] 或者 [B, T, 4]
            # 这里为了通用性，如果维度不匹配，需要在外部处理好，这里假设外部已经处理好 B*E
            if x_mark_enc.dim() == 3 and x_mark_enc.shape[0] != B_E: 
                 # 简单的容错，防止并在batch
                 pass 
            
            # x_mark_enc: [B*E, 4, T] (在模型里transpose过) -> [B*E, T, 4]
            x_mark_enc_in = x_mark_enc.transpose(1, 2)
            time_feat = self.time_proj(x_mark_enc_in) #[B*E, T, time_dim]
            time_feat = time_feat.permute(0, 2, 1)    #[B*E, time_dim, T]
        else:
            time_feat = torch.zeros(B_E, self.time_dim, T, device=data.device)
                    
        # 3. [NEW] 鲁棒的 FFT 特征提取 (借鉴 F2)
        if self.FFT:
            # RFFT 对实数序列更合适，输出长度为 T//2 + 1
            x_fft = torch.fft.rfft(data, dim=-1, norm='ortho') # [B*E, 1, fft_len]
            amplitude = torch.abs(x_fft) # 取幅度谱，忽略相位噪声
            
            # 投影回特征空间 [B*E, 1, fft_len] -> [B*E, 1, freq_dim]
            freq_feat_global = self.freq_proj(amplitude) 
            
            # 将全局频率特征广播到时间维度 (类似于 global_mean)
            freq_feat = freq_feat_global.permute(0, 2, 1).repeat(1, 1, T) # [B*E, freq_dim, T]
        else:
            freq_feat = None
        
        # 拼接列表
        feature_list = [data, global_mean, roll_std, roll_max, roll_min, time_feat]
        if self.FFT:
            feature_list.append(freq_feat)

        stat_features = torch.cat(feature_list, dim=1) 
        return stat_features