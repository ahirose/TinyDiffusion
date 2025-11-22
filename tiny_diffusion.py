"""
TinyDiffusion - 最もシンプルなDiffusionモデル実装
TinyDiffusion - The Simplest Diffusion Model Implementation

=============================================================================
【日本語】Diffusionモデルとは？
=============================================================================
Diffusionモデルは、画像生成のための深層学習モデルです。
基本的なアイデアは非常にシンプルです：

1. Forward Process（拡散過程）:
   - きれいな画像に少しずつノイズを加えていく
   - 最終的には完全なノイズ（ランダムな点々）になる

2. Reverse Process（逆拡散過程）:
   - ノイズから少しずつノイズを取り除いていく
   - 最終的にきれいな画像が生成される

学習では、「ノイズを予測する」ことを学びます。
生成時は、ランダムノイズから始めて、予測したノイズを引き算していきます。

=============================================================================
【English】What is a Diffusion Model?
=============================================================================
A Diffusion model is a deep learning model for image generation.
The basic idea is very simple:

1. Forward Process (Diffusion):
   - Gradually add noise to a clean image
   - Eventually it becomes pure noise (random dots)

2. Reverse Process (Denoising):
   - Gradually remove noise from the noisy image
   - Eventually a clean image is generated

During training, the model learns to "predict noise".
During generation, we start from random noise and subtract the predicted noise.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# ノイズスケジュール / Noise Schedule
# =============================================================================
# 【日本語】
# βスケジュールは、各タイムステップでどれくらいのノイズを加えるかを決めます。
# β（ベータ）が小さいと少しのノイズ、大きいと多くのノイズが加わります。
#
# 【English】
# The β schedule determines how much noise to add at each timestep.
# Small β means little noise, large β means more noise.
# =============================================================================

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    線形βスケジュール / Linear beta schedule

    【日本語】
    最もシンプルなスケジュール。βを線形に増加させます。
    timesteps=1000の場合、t=0でβ=0.0001、t=999でβ=0.02となります。

    【English】
    The simplest schedule. β increases linearly.
    For timesteps=1000, β=0.0001 at t=0, β=0.02 at t=999.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    コサインβスケジュール / Cosine beta schedule

    【日本語】
    より滑らかなスケジュール。画像の端がきれいになりやすいです。
    数式: β_t = 1 - (α_bar_t / α_bar_{t-1})
    ここで α_bar_t = cos((t/T + s) / (1 + s) * π/2)^2

    【English】
    A smoother schedule. Tends to produce cleaner image edges.
    Formula: β_t = 1 - (α_bar_t / α_bar_{t-1})
    where α_bar_t = cos((t/T + s) / (1 + s) * π/2)^2
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# =============================================================================
# Diffusionモデルの数学 / Mathematics of Diffusion
# =============================================================================
# 【日本語】
# Forward Process（ノイズを加える）:
#   q(x_t | x_0) = N(x_t; √(α_bar_t) * x_0, (1 - α_bar_t) * I)
#
#   つまり：x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε
#   ここで ε はランダムノイズ（標準正規分布）
#
# Reverse Process（ノイズを取り除く）:
#   p(x_{t-1} | x_t) を学習する
#   実際には、x_t に含まれるノイズ ε を予測する
#
# 【English】
# Forward Process (adding noise):
#   q(x_t | x_0) = N(x_t; √(α_bar_t) * x_0, (1 - α_bar_t) * I)
#
#   Which means: x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε
#   where ε is random noise (standard normal distribution)
#
# Reverse Process (removing noise):
#   Learn p(x_{t-1} | x_t)
#   In practice, predict the noise ε contained in x_t
# =============================================================================


class DiffusionModel:
    """
    Diffusionモデルのコア / Core Diffusion Model

    【日本語】
    このクラスはDiffusionの「プロセス」を管理します：
    - ノイズのスケジュール（αやβの計算）
    - Forward process（ノイズを加える）
    - Reverse process（ノイズを取り除く）

    注意：このクラスはニューラルネットワークではありません。
    ニューラルネットワーク（UNet）は別に定義します。

    【English】
    This class manages the diffusion "process":
    - Noise schedule (calculating α and β)
    - Forward process (adding noise)
    - Reverse process (removing noise)

    Note: This class is NOT a neural network.
    The neural network (UNet) is defined separately.
    """

    def __init__(self, timesteps=1000, beta_schedule='linear', device='cpu'):
        """
        【日本語】
        Args:
            timesteps: 拡散ステップ数（通常1000）
            beta_schedule: 'linear' または 'cosine'
            device: 'cpu' または 'cuda'

        【English】
        Args:
            timesteps: Number of diffusion steps (typically 1000)
            beta_schedule: 'linear' or 'cosine'
            device: 'cpu' or 'cuda'
        """
        self.timesteps = timesteps
        self.device = device

        # βスケジュールを設定 / Set beta schedule
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        else:
            self.betas = cosine_beta_schedule(timesteps).to(device)

        # =================================================================
        # 重要な定数を事前計算 / Pre-compute important constants
        # =================================================================
        # 【日本語】
        # α_t = 1 - β_t （各ステップで残る画像の割合）
        # α_bar_t = Π_{i=1}^{t} α_i （最初からtまでの累積積）
        #
        # 【English】
        # α_t = 1 - β_t (fraction of image remaining at each step)
        # α_bar_t = Π_{i=1}^{t} α_i (cumulative product from 1 to t)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 便利な定数 / Convenient constants
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Reverse process用の定数 / Constants for reverse process
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward Process: きれいな画像にノイズを加える
        Forward Process: Add noise to a clean image

        【日本語】
        数式: x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε

        これは「reparameterization trick」と呼ばれる技法で、
        任意のタイムステップtの画像を一発で計算できます。
        （1ステップずつノイズを加える必要がない！）

        【English】
        Formula: x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε

        This is called the "reparameterization trick".
        We can compute the image at any timestep t directly.
        (No need to add noise step by step!)

        Args:
            x_0: きれいな画像 / Clean image [B, C, H, W]
            t: タイムステップ / Timestep [B]
            noise: 使用するノイズ（指定しない場合はランダム生成）
                   Noise to use (randomly generated if not specified)

        Returns:
            x_t: ノイズを加えた画像 / Noisy image
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # tに対応する係数を取得 / Get coefficients for timestep t
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # 形状を調整 [B] -> [B, 1, 1, 1] / Reshape [B] -> [B, 1, 1, 1]
        while sqrt_alpha_cumprod_t.dim() < x_0.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        # x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        Reverse Process（1ステップ）: ノイズを少し取り除く
        Reverse Process (one step): Remove a little noise

        【日本語】
        1. モデルでノイズを予測
        2. 予測したノイズを使って x_{t-1} を計算

        【English】
        1. Predict noise using the model
        2. Calculate x_{t-1} using the predicted noise

        Args:
            model: ノイズ予測ネットワーク / Noise prediction network
            x_t: 現在のノイズ画像 / Current noisy image
            t: 現在のタイムステップ / Current timestep

        Returns:
            x_{t-1}: 少しノイズが減った画像 / Slightly denoised image
        """
        # バッチサイズを取得 / Get batch size
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # モデルでノイズを予測 / Predict noise with model
        predicted_noise = model(x_t, t_tensor)

        # 必要な係数を取得 / Get necessary coefficients
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]

        # =================================================================
        # x_{t-1} の平均を計算 / Calculate mean of x_{t-1}
        # =================================================================
        # 【日本語】
        # 数式: μ = (1/√α_t) * (x_t - (β_t/√(1-α_bar_t)) * ε_θ)
        # ここで ε_θ は予測されたノイズ
        #
        # 【English】
        # Formula: μ = (1/√α_t) * (x_t - (β_t/√(1-α_bar_t)) * ε_θ)
        # where ε_θ is the predicted noise

        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)

        mean = (1 / sqrt_alpha_t) * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
        )

        # t > 0 の場合はノイズを追加（確率的サンプリング）
        # Add noise if t > 0 (stochastic sampling)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            x_t_minus_1 = mean + variance * noise
        else:
            # t = 0 の場合はノイズなし / No noise when t = 0
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, model, shape):
        """
        画像を生成 / Generate images

        【日本語】
        完全なノイズから始めて、1ステップずつノイズを取り除き、
        最終的にきれいな画像を生成します。

        【English】
        Start from pure noise, remove noise step by step,
        and generate a clean image at the end.

        Args:
            model: 学習済みノイズ予測ネットワーク / Trained noise prediction network
            shape: 生成する画像の形状 / Shape of images to generate [B, C, H, W]

        Returns:
            x_0: 生成された画像 / Generated images
        """
        # 完全なランダムノイズから開始 / Start from pure random noise
        x_t = torch.randn(shape, device=self.device)

        # t = T-1, T-2, ..., 1, 0 の順でノイズを取り除く
        # Remove noise in order: t = T-1, T-2, ..., 1, 0
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(model, x_t, t)

            # 進捗表示 / Show progress
            if t % 100 == 0:
                print(f"  Sampling step {self.timesteps - t}/{self.timesteps}")

        return x_t


# =============================================================================
# シンプルなU-Net / Simple U-Net
# =============================================================================
# 【日本語】
# U-Netは「ノイズを予測する」ニューラルネットワークです。
# 入力：ノイズ画像 x_t と タイムステップ t
# 出力：予測されたノイズ ε_θ
#
# U-Netの構造：
# - Encoder（ダウンサンプリング）: 画像を小さくしながら特徴を抽出
# - Middle: 最も圧縮された状態で処理
# - Decoder（アップサンプリング）: 画像を元のサイズに戻す
# - Skip connections: EncoderとDecoderを直接つなぐ（細かい情報を保持）
#
# 【English】
# U-Net is a neural network that "predicts noise".
# Input: Noisy image x_t and timestep t
# Output: Predicted noise ε_θ
#
# U-Net structure:
# - Encoder (downsampling): Extract features while making image smaller
# - Middle: Process at the most compressed state
# - Decoder (upsampling): Restore image to original size
# - Skip connections: Directly connect Encoder and Decoder (preserve fine details)
# =============================================================================


class SinusoidalPositionEmbedding(nn.Module):
    """
    時間埋め込み / Time Embedding

    【日本語】
    タイムステップ t を高次元のベクトルに変換します。
    Transformerで使われるPositional Encodingと同じ考え方です。

    sin/cos関数を使うことで、異なるtが異なるパターンを持ち、
    モデルが「今どのステップか」を理解しやすくなります。

    【English】
    Convert timestep t into a high-dimensional vector.
    Same idea as Positional Encoding used in Transformers.

    Using sin/cos functions, different t values have different patterns,
    making it easier for the model to understand "which step it is now".
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        # 周波数を計算 / Calculate frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # t と掛け合わせて sin/cos を適用 / Multiply with t and apply sin/cos
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class Block(nn.Module):
    """
    基本ブロック / Basic Block

    【日本語】
    Conv -> GroupNorm -> SiLU のシンプルな構造
    時間情報も条件として加えられます

    【English】
    Simple structure: Conv -> GroupNorm -> SiLU
    Time information can also be added as conditioning
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # 時間埋め込みの射影 / Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        x = self.conv(x)
        x = self.norm(x)

        # 時間情報を加える / Add time information
        if self.time_mlp is not None and t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            x = x + t_emb[:, :, None, None]

        x = self.act(x)
        return x


class SimpleUNet(nn.Module):
    """
    シンプルなU-Net / Simple U-Net

    【日本語】
    教育目的のため、最小限の構造にしています：
    - 2段階のダウンサンプリング
    - 2段階のアップサンプリング
    - Skip connections

    入力: [B, C, H, W] の画像と [B] のタイムステップ
    出力: [B, C, H, W] の予測ノイズ

    【English】
    Minimal structure for educational purposes:
    - 2 levels of downsampling
    - 2 levels of upsampling
    - Skip connections

    Input: [B, C, H, W] image and [B] timestep
    Output: [B, C, H, W] predicted noise
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()

        # =================================================================
        # 時間埋め込み / Time embedding
        # =================================================================
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # =================================================================
        # Encoder（ダウンサンプリング）/ Encoder (Downsampling)
        # =================================================================
        # 【日本語】画像を小さくしながら特徴を抽出
        # 【English】Extract features while making image smaller

        self.enc1 = Block(in_channels, base_channels, time_emb_dim)
        self.enc2 = Block(base_channels, base_channels * 2, time_emb_dim)

        self.pool = nn.MaxPool2d(2)  # 画像サイズを半分に / Halve image size

        # =================================================================
        # Middle / 中間層
        # =================================================================
        self.middle = Block(base_channels * 2, base_channels * 2, time_emb_dim)

        # =================================================================
        # Decoder（アップサンプリング）/ Decoder (Upsampling)
        # =================================================================
        # 【日本語】画像を元のサイズに戻す（Skip connectionで細かい情報を保持）
        # 【English】Restore image to original size (preserve fine details with skip connections)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = Block(base_channels * 4, base_channels, time_emb_dim)  # *4 because of skip connection

        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec2 = Block(base_channels * 2, base_channels, time_emb_dim)  # *2 because of skip connection

        # =================================================================
        # 出力層 / Output layer
        # =================================================================
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        【日本語】
        Forward pass:
        1. タイムステップを埋め込みに変換
        2. Encoderで特徴抽出（skip connectionsを保存）
        3. Middle層で処理
        4. Decoderで復元（skip connectionsを結合）
        5. 出力層でノイズを予測

        【English】
        Forward pass:
        1. Convert timestep to embedding
        2. Extract features with Encoder (save skip connections)
        3. Process with Middle layer
        4. Restore with Decoder (concatenate skip connections)
        5. Predict noise with output layer
        """
        # 時間埋め込み / Time embedding
        t_emb = self.time_embedding(t.float())

        # Encoder
        e1 = self.enc1(x, t_emb)           # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1), t_emb)  # [B, 128, H/2, W/2]

        # Middle
        m = self.middle(self.pool(e2), t_emb)  # [B, 128, H/4, W/4]

        # Decoder with skip connections
        d1 = self.up1(m)                    # [B, 128, H/2, W/2]
        d1 = torch.cat([d1, e2], dim=1)     # [B, 256, H/2, W/2] (skip connection)
        d1 = self.dec1(d1, t_emb)           # [B, 64, H/2, W/2]

        d2 = self.up2(d1)                   # [B, 64, H, W]
        d2 = torch.cat([d2, e1], dim=1)     # [B, 128, H, W] (skip connection)
        d2 = self.dec2(d2, t_emb)           # [B, 64, H, W]

        # 出力 / Output
        return self.out(d2)                 # [B, out_channels, H, W]


# =============================================================================
# 損失関数 / Loss Function
# =============================================================================

def diffusion_loss(model, diffusion, x_0, t=None):
    """
    Diffusionの損失関数（シンプル版）
    Diffusion loss function (simple version)

    【日本語】
    損失 = |ε - ε_θ|^2
    つまり、本当のノイズと予測したノイズの差の二乗

    手順：
    1. ランダムなタイムステップ t を選ぶ
    2. ランダムなノイズ ε を生成
    3. x_0 にノイズを加えて x_t を作る
    4. モデルで ε_θ を予測
    5. ε と ε_θ の差を計算

    【English】
    Loss = |ε - ε_θ|^2
    That is, the squared difference between true noise and predicted noise

    Steps:
    1. Choose a random timestep t
    2. Generate random noise ε
    3. Add noise to x_0 to create x_t
    4. Predict ε_θ with the model
    5. Calculate the difference between ε and ε_θ

    Args:
        model: U-Net（ノイズ予測ネットワーク）/ U-Net (noise prediction network)
        diffusion: DiffusionModelインスタンス / DiffusionModel instance
        x_0: きれいな画像 / Clean images [B, C, H, W]
        t: タイムステップ（指定しない場合はランダム）/ Timestep (random if not specified)

    Returns:
        loss: 平均二乗誤差 / Mean squared error
    """
    batch_size = x_0.shape[0]
    device = x_0.device

    # ランダムなタイムステップを選ぶ / Choose random timesteps
    if t is None:
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

    # ランダムなノイズを生成 / Generate random noise
    noise = torch.randn_like(x_0)

    # x_t を計算（Forward process）/ Calculate x_t (Forward process)
    x_t = diffusion.q_sample(x_0, t, noise=noise)

    # ノイズを予測 / Predict noise
    predicted_noise = model(x_t, t)

    # 損失を計算（MSE）/ Calculate loss (MSE)
    loss = F.mse_loss(predicted_noise, noise)

    return loss


# =============================================================================
# 使用例 / Usage Example
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TinyDiffusion - シンプルなDiffusionモデルのデモ")
    print("TinyDiffusion - Simple Diffusion Model Demo")
    print("=" * 60)

    # デバイス設定 / Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # モデルとDiffusionを初期化 / Initialize model and diffusion
    print("\n--- Initializing ---")
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    diffusion = DiffusionModel(timesteps=1000, device=device)

    # パラメータ数を表示 / Show parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ダミーデータでテスト / Test with dummy data
    print("\n--- Testing Forward Process ---")
    x_0 = torch.randn(4, 1, 32, 32).to(device)  # バッチサイズ4、1チャンネル、32x32
    t = torch.randint(0, 1000, (4,)).to(device)

    x_t = diffusion.q_sample(x_0, t)
    print(f"Input shape: {x_0.shape}")
    print(f"Noisy image shape: {x_t.shape}")

    # 損失計算テスト / Test loss calculation
    print("\n--- Testing Loss Calculation ---")
    loss = diffusion_loss(model, diffusion, x_0)
    print(f"Loss: {loss.item():.4f}")

    # サンプリングテスト（短縮版）/ Test sampling (shortened)
    print("\n--- Testing Sampling (10 steps only for demo) ---")
    diffusion_short = DiffusionModel(timesteps=10, device=device)
    samples = diffusion_short.sample(model, (2, 1, 32, 32))
    print(f"Generated samples shape: {samples.shape}")

    print("\n" + "=" * 60)
    print("Demo completed! / デモ完了！")
    print("See train.py for training and generate.py for generation.")
    print("学習は train.py、生成は generate.py を参照してください。")
    print("=" * 60)
