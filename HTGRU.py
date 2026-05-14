"""
IoT Time-Series Anomaly Detection Using Residual GRU-Attention (RGAAD)
Implementation based on the paper:
"IoT Time-Series Anomaly Detection Using a Hybrid Transformer-GRU Fusion Model"

Key Components:
1. DataPreprocessor - 标准化和滑动窗口处理
2. ResidualGRUAttention - 核心模型架构（GRU+Attention+VAE）
3. RGAAD_Trainer - 训练和异常检测流程
4. Evaluator - 性能评估模块
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


# ---------------------------- 数据预处理模块 ----------------------------
class DataPreprocessor:
    """处理原始时间序列数据的标准化和窗口分割"""

    def __init__(self, window_size=10):
        """
        Args:
            window_size (int): 滑动窗口长度（论文默认值10）
        """
        self.window_size = window_size
        self.scaler = StandardScaler()

    def fit_transform(self, train_data):
        """训练数据标准化和窗口化处理
        Args:
            train_data (np.array): 原始训练数据 [num_timesteps, num_features]
        Returns:
            torch.Tensor: 窗口化序列 [num_windows, window_size, num_features]
        """
        # 标准化（仅用训练数据拟合）
        self.scaler.fit(train_data)
        normalized = self.scaler.transform(train_data)

        # 滑动窗口处理
        sequences = []
        for i in range(len(normalized) - self.window_size + 1):
            seq = normalized[i:i + self.window_size]
            sequences.append(seq)
        return torch.FloatTensor(np.array(sequences))

    def transform(self, data):
        """对新数据应用预处理（使用训练数据的均值和方差）"""
        normalized = self.scaler.transform(data)
        sequences = []
        for i in range(len(normalized) - self.window_size + 1):
            seq = normalized[i:i + self.window_size]
            sequences.append(seq)
        return torch.FloatTensor(np.array(sequences))


# ---------------------------- 核心模型架构 ----------------------------
class ResidualGRUAttention(nn.Module):
    """RGAAD核心模型：双向GRU + 多头注意力 + 变分自编码器"""

    def __init__(self, input_dim, latent_dim=64, num_heads=4):
        """
        Args:
            input_dim (int): 输入特征维度（如SMD数据集为38）
            latent_dim (int): 潜在空间维度（论文默认64）
            num_heads (int): 注意力头数（论文默认4）
        """
        super().__init__()
        self.window_size = None  # 动态设置
        self.input_dim = input_dim

        # 编码器：双向GRU（对应论文图3蓝色模块）
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=latent_dim,
            bidirectional=True,
            batch_first=True
        )

        # 多头注意力机制（对应论文3.2节）
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim * 2,
            num_heads=num_heads,
            batch_first=True
        )

        # 变分自编码器结构（论文3.3节）
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)

        # 解码器：GRU + 残差连接（对应论文图3绿色模块）
        self.decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim * 2,
            batch_first=True
        )
        self.output_layer = nn.Linear(latent_dim * 2, input_dim)

    def forward(self, x):
        """前向传播流程（对应论文图3箭头方向）"""
        batch_size = x.size(0)
        if self.window_size is None:
            self.window_size = x.size(1)

        # ---- 编码阶段 ----
        # 双向GRU处理
        gru_out, _ = self.gru(x)  # [batch, window_size, latent_dim*2]

        # 注意力机制（动态特征加权）
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)

        # ---- 潜在空间 ----
        # 计算分布参数（论文公式(4)）
        mu = self.fc_mu(attn_out[:, -1, :])  # 取最后时间步
        logvar = self.fc_logvar(attn_out[:, -1, :])
        z = self.reparameterize(mu, logvar)  # [batch, latent_dim]

        # ---- 解码阶段 ----
        # 通过GRU解码（论文3.3节）
        decoder_input = z.unsqueeze(1).repeat(1, self.window_size, 1)
        decoder_out, _ = self.decoder_gru(decoder_input)
        reconstruction = self.output_layer(decoder_out)

        # 残差连接（论文强调的关键设计）
        return reconstruction + x, mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧（VAE核心）"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# ---------------------------- 训练与评估模块 ----------------------------
class RGAAD_Trainer:
    """封装训练逻辑和异常检测功能"""

    def __init__(self, model, lr=0.001, beta=0.1):
        """
        Args:
            model (nn.Module): RGAAD模型实例
            lr (float): 学习率（论文默认0.001）
            beta (float): KL散度权重（论文默认0.1）
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.beta = beta

    def train_epoch(self, train_loader):
        """单epoch训练"""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            # 前向传播
            recon_x, mu, logvar = self.model(batch)

            # 复合损失函数（论文公式(5)）
            recon_loss = F.mse_loss(recon_x, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl_loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def detect_anomalies(self, data_loader, k=3):
        """异常检测（论文3.3节动态阈值方法）"""
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for batch in data_loader:
                recon_x, _, _ = self.model(batch)
                # 计算窗口内平均绝对误差
                errors = torch.mean(torch.abs(batch - recon_x), dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())

        # 动态阈值（论文公式(6)）
        errors = np.array(all_errors)
        threshold = errors.mean() + k * errors.std()
        return (errors > threshold).astype(int), errors


class Evaluator:
    """性能评估模块（计算Precision/Recall/F1）"""

    @staticmethod
    def evaluate(y_true, y_pred):
        """计算评估指标"""
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }


# ---------------------------- 主程序示例 ----------------------------
if __name__ == "__main__":
    # 示例数据参数（以SMD数据集为例）
    WINDOW_SIZE = 10
    INPUT_DIM = 38
    BATCH_SIZE = 64

    # 1. 数据准备
    train_data = np.random.randn(1000, INPUT_DIM)
    test_data = np.random.randn(200, INPUT_DIM)
    test_labels = np.random.randint(0, 2, 200 - WINDOW_SIZE + 1)

    # 2. 数据预处理
    preprocessor = DataPreprocessor(WINDOW_SIZE)
    train_sequences = preprocessor.fit_transform(train_data)
    test_sequences = preprocessor.transform(test_data)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_sequences, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_sequences, batch_size=BATCH_SIZE
    )

    # 3. 初始化模型和训练器
    model = ResidualGRUAttention(INPUT_DIM)
    trainer = RGAAD_Trainer(model)

    # 4. 训练循环（论文训练10-20个epoch）
    for epoch in range(10):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # 5. 异常检测和评估
    preds, scores = trainer.detect_anomalies(test_loader)
    metrics = Evaluator.evaluate(test_labels, preds)

    print("\nEvaluation Results:")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1-score: {metrics['f1']:.2%}")