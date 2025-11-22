"""
TinyDiffusion 学習スクリプト / Training Script

=============================================================================
【日本語】学習の流れ
=============================================================================
1. データセットを読み込む（MNIST：手書き数字）
2. 各エポックで：
   a. バッチごとにきれいな画像を取得
   b. ランダムなタイムステップ t を選ぶ
   c. ランダムなノイズを加えて x_t を作る
   d. モデルでノイズを予測
   e. 予測と実際のノイズの差（損失）を計算
   f. バックプロパゲーションで学習

=============================================================================
【English】Training Flow
=============================================================================
1. Load dataset (MNIST: handwritten digits)
2. For each epoch:
   a. Get clean images for each batch
   b. Choose random timestep t
   c. Add random noise to create x_t
   d. Predict noise with the model
   e. Calculate loss (difference between prediction and actual noise)
   f. Learn through backpropagation
=============================================================================
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import argparse
import random

from tiny_diffusion import SimpleUNet, DiffusionModel, diffusion_loss


# =============================================================================
# 合成データセット / Synthetic Dataset
# =============================================================================
# 【日本語】
# MNISTがダウンロードできない場合や、より単純な実験をしたい場合に使用
# シンプルな図形（四角形、円、線）を生成します
#
# 【English】
# Used when MNIST cannot be downloaded or for simpler experiments
# Generates simple shapes (squares, circles, lines)
# =============================================================================

class SyntheticShapesDataset(Dataset):
    """
    合成図形データセット / Synthetic Shapes Dataset

    【日本語】
    シンプルな図形を生成するデータセット：
    - 四角形
    - 円
    - 対角線

    【English】
    Dataset that generates simple shapes:
    - Squares
    - Circles
    - Diagonal lines
    """

    def __init__(self, size=10000, image_size=28):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.zeros(1, self.image_size, self.image_size)

        shape_type = random.randint(0, 2)

        if shape_type == 0:
            # 四角形 / Square
            x = random.randint(2, self.image_size - 12)
            y = random.randint(2, self.image_size - 12)
            size = random.randint(5, 10)
            img[0, y:y+size, x:x+size] = 1.0

        elif shape_type == 1:
            # 円（近似）/ Circle (approximation)
            cx = random.randint(7, self.image_size - 7)
            cy = random.randint(7, self.image_size - 7)
            r = random.randint(3, 6)
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i - cy) ** 2 + (j - cx) ** 2 <= r ** 2:
                        img[0, i, j] = 1.0

        else:
            # 対角線 / Diagonal line
            thickness = random.randint(1, 3)
            direction = random.randint(0, 1)
            for i in range(self.image_size):
                if direction == 0:
                    j = i
                else:
                    j = self.image_size - 1 - i
                for t in range(-thickness//2, thickness//2 + 1):
                    if 0 <= j + t < self.image_size:
                        img[0, i, j + t] = 1.0

        # [-1, 1] に正規化 / Normalize to [-1, 1]
        img = img * 2 - 1

        return img, 0  # ラベルは使わないのでダミー / Dummy label


def train(
    epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    timesteps=1000,
    device='auto',
    save_dir='checkpoints',
    sample_interval=5,
    use_synthetic=False
):
    """
    Diffusionモデルを学習 / Train the Diffusion model

    【日本語】
    Args:
        epochs: 学習エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        timesteps: Diffusionのステップ数
        device: 'auto', 'cuda', または 'cpu'
        save_dir: チェックポイント保存先
        sample_interval: サンプル画像を生成するエポック間隔
        use_synthetic: 合成データを使用するか（MNISTの代わり）

    【English】
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        timesteps: Number of diffusion steps
        device: 'auto', 'cuda', or 'cpu'
        save_dir: Directory to save checkpoints
        sample_interval: Interval (epochs) to generate sample images
        use_synthetic: Use synthetic data instead of MNIST
    """

    # =================================================================
    # デバイス設定 / Device setup
    # =================================================================
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # =================================================================
    # データセット準備 / Prepare dataset
    # =================================================================
    if use_synthetic:
        # 【日本語】
        # 合成データセット：シンプルな図形（四角形、円、線）
        # MNISTがダウンロードできない場合に便利
        #
        # 【English】
        # Synthetic dataset: Simple shapes (squares, circles, lines)
        # Useful when MNIST cannot be downloaded

        print("\n--- Using Synthetic Shapes Dataset ---")
        dataset = SyntheticShapesDataset(size=10000, image_size=28)
    else:
        # 【日本語】
        # MNISTは28x28のグレースケール画像（手書き数字0-9）
        # シンプルで学習が速いので、教育に最適です
        #
        # 【English】
        # MNIST is 28x28 grayscale images (handwritten digits 0-9)
        # Simple and fast to train, perfect for education

        print("\n--- Loading MNIST Dataset ---")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
        ])

        try:
            dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )
        except RuntimeError as e:
            print(f"\nMNIST download failed: {e}")
            print("Falling back to synthetic dataset...")
            print("MNISTダウンロード失敗。合成データに切り替えます...\n")
            dataset = SyntheticShapesDataset(size=10000, image_size=28)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows互換性のため / For Windows compatibility
        drop_last=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # =================================================================
    # モデル初期化 / Initialize model
    # =================================================================
    print("\n--- Initializing Model ---")
    model = SimpleUNet(
        in_channels=1,   # グレースケール / Grayscale
        out_channels=1,
        base_channels=64,
        time_emb_dim=128
    ).to(device)

    diffusion = DiffusionModel(timesteps=timesteps, device=device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # =================================================================
    # オプティマイザ設定 / Setup optimizer
    # =================================================================
    # 【日本語】
    # AdamWは最も一般的な最適化アルゴリズムの一つです
    # 学習率1e-3はDiffusionモデルでよく使われる値です
    #
    # 【English】
    # AdamW is one of the most common optimization algorithms
    # Learning rate 1e-3 is commonly used for Diffusion models

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # =================================================================
    # 学習ループ / Training loop
    # =================================================================
    print("\n--- Starting Training ---")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Timesteps: {timesteps}")
    print()

    # チェックポイントディレクトリ作成 / Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):
            # ==========================================================
            # 1. データをデバイスに移動 / Move data to device
            # ==========================================================
            images = images.to(device)

            # ==========================================================
            # 2. 損失を計算 / Calculate loss
            # ==========================================================
            # 【日本語】
            # diffusion_loss は以下を行います：
            # - ランダムな t を選択
            # - ランダムなノイズ ε を生成
            # - x_t = √(α_bar_t) * x_0 + √(1-α_bar_t) * ε を計算
            # - モデルで ε_θ を予測
            # - Loss = MSE(ε, ε_θ) を計算
            #
            # 【English】
            # diffusion_loss does the following:
            # - Select random t
            # - Generate random noise ε
            # - Calculate x_t = √(α_bar_t) * x_0 + √(1-α_bar_t) * ε
            # - Predict ε_θ with the model
            # - Calculate Loss = MSE(ε, ε_θ)

            loss = diffusion_loss(model, diffusion, images)

            # ==========================================================
            # 3. バックプロパゲーション / Backpropagation
            # ==========================================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 進捗表示 / Show progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        # エポック終了時の平均損失 / Average loss at end of epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} completed | Average Loss: {avg_loss:.4f}")

        # ==========================================================
        # サンプル画像を生成 / Generate sample images
        # ==========================================================
        if (epoch + 1) % sample_interval == 0 or epoch == epochs - 1:
            print(f"\n--- Generating samples at epoch {epoch+1} ---")
            generate_samples(model, diffusion, epoch+1, save_dir, device)

        # ==========================================================
        # チェックポイント保存 / Save checkpoint
        # ==========================================================
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'timesteps': timesteps,
        }
        torch.save(checkpoint, os.path.join(save_dir, 'latest.pt'))

        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f'epoch_{epoch+1}.pt'))

    # =================================================================
    # 学習曲線を保存 / Save learning curve
    # =================================================================
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss / 学習損失')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    print("\n" + "=" * 60)
    print("Training completed! / 学習完了！")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 60)

    return model, diffusion


@torch.no_grad()
def generate_samples(model, diffusion, epoch, save_dir, device, n_samples=16):
    """
    サンプル画像を生成して保存 / Generate and save sample images

    【日本語】
    学習中の進捗を確認するために、定期的にサンプル画像を生成します。
    純粋なノイズから始めて、ステップごとにノイズを取り除いていきます。

    【English】
    Generate sample images periodically to check training progress.
    Start from pure noise and remove noise step by step.
    """
    model.eval()

    # 短縮版のDiffusion（デモ用）/ Shortened diffusion (for demo)
    # フルの1000ステップは時間がかかるので、100ステップで近似
    # Full 1000 steps take time, so approximate with 100 steps
    short_diffusion = DiffusionModel(timesteps=100, device=device)

    # サンプル生成 / Generate samples
    samples = short_diffusion.sample(model, (n_samples, 1, 28, 28))

    # [-1, 1] -> [0, 1] に変換 / Convert [-1, 1] -> [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    # グリッド表示 / Display as grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < n_samples:
            ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle(f'Generated Samples at Epoch {epoch}\nエポック {epoch} での生成サンプル')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
    plt.close()

    print(f"Samples saved to: {save_dir}/samples_epoch_{epoch}.png")

    model.train()


if __name__ == "__main__":
    # =================================================================
    # コマンドライン引数 / Command line arguments
    # =================================================================
    parser = argparse.ArgumentParser(
        description='Train TinyDiffusion model / TinyDiffusionモデルの学習'
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs / エポック数')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size / バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate / 学習率')
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps / ステップ数')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--sample-interval', type=int, default=5, help='Sample generation interval')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic shapes dataset / 合成図形データを使用')

    args = parser.parse_args()

    print("=" * 60)
    print("TinyDiffusion Training / TinyDiffusion 学習")
    print("=" * 60)

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        timesteps=args.timesteps,
        device=args.device,
        save_dir=args.save_dir,
        sample_interval=args.sample_interval,
        use_synthetic=args.synthetic
    )
