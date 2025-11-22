"""
TinyDiffusion 生成スクリプト / Generation Script

=============================================================================
【日本語】生成の流れ
=============================================================================
1. 学習済みモデルを読み込む
2. 完全なランダムノイズ（標準正規分布）から開始
3. t = T-1, T-2, ..., 1, 0 の順で：
   a. モデルでノイズを予測
   b. 予測したノイズを使って x_{t-1} を計算
4. 最終的に x_0 （きれいな画像）が得られる

=============================================================================
【English】Generation Flow
=============================================================================
1. Load trained model
2. Start from pure random noise (standard normal distribution)
3. For t = T-1, T-2, ..., 1, 0:
   a. Predict noise with the model
   b. Calculate x_{t-1} using the predicted noise
4. Finally obtain x_0 (clean image)
=============================================================================
"""

import torch
import matplotlib.pyplot as plt
import argparse
import os

from tiny_diffusion import SimpleUNet, DiffusionModel


@torch.no_grad()
def generate(
    checkpoint_path='checkpoints/latest.pt',
    n_samples=16,
    timesteps=1000,
    device='auto',
    output_path='generated_samples.png',
    show_process=False
):
    """
    学習済みモデルで画像を生成 / Generate images with trained model

    【日本語】
    Args:
        checkpoint_path: 学習済みモデルのパス
        n_samples: 生成する画像の数
        timesteps: サンプリングステップ数（学習時と同じ推奨）
        device: 'auto', 'cuda', または 'cpu'
        output_path: 出力画像のパス
        show_process: 生成過程を表示するか

    【English】
    Args:
        checkpoint_path: Path to trained model
        n_samples: Number of images to generate
        timesteps: Number of sampling steps (same as training recommended)
        device: 'auto', 'cuda', or 'cpu'
        output_path: Output image path
        show_process: Whether to show generation process
    """

    # =================================================================
    # デバイス設定 / Device setup
    # =================================================================
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # =================================================================
    # モデル読み込み / Load model
    # =================================================================
    print(f"\n--- Loading model from {checkpoint_path} ---")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
        print("まず train.py を実行してモデルを学習してください。")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SimpleUNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        time_emb_dim=128
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 学習時のタイムステップを取得（可能な場合）/ Get training timesteps if available
    if 'timesteps' in checkpoint:
        timesteps = checkpoint['timesteps']
        print(f"Using timesteps from checkpoint: {timesteps}")

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # =================================================================
    # Diffusion設定 / Setup Diffusion
    # =================================================================
    diffusion = DiffusionModel(timesteps=timesteps, device=device)

    # =================================================================
    # 画像生成 / Generate images
    # =================================================================
    print(f"\n--- Generating {n_samples} images ---")
    print("This may take a while... / しばらくお待ちください...")

    if show_process:
        # 生成過程を表示 / Show generation process
        samples, intermediates = sample_with_intermediates(
            model, diffusion, n_samples, device
        )
        save_generation_process(intermediates, 'generation_process.png')
    else:
        samples = diffusion.sample(model, (n_samples, 1, 28, 28))

    # =================================================================
    # 結果を保存 / Save results
    # =================================================================
    # [-1, 1] -> [0, 1] に変換 / Convert [-1, 1] -> [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    # グリッド表示 / Display as grid
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    if n_samples == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < n_samples:
                axes[i][j].imshow(samples[idx, 0].cpu().numpy(), cmap='gray')
            axes[i][j].axis('off')

    plt.suptitle('Generated Images / 生成された画像')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n--- Results saved to {output_path} ---")
    print("Generation completed! / 生成完了！")


@torch.no_grad()
def sample_with_intermediates(model, diffusion, n_samples, device, save_steps=10):
    """
    中間状態を保存しながらサンプリング / Sample while saving intermediate states

    【日本語】
    生成過程を可視化するために、途中の状態を保存します。
    これにより、ノイズがどのように画像に変わっていくかを確認できます。

    【English】
    Save intermediate states to visualize the generation process.
    This allows us to see how noise transforms into an image.
    """
    # 完全なランダムノイズから開始 / Start from pure random noise
    x_t = torch.randn(n_samples, 1, 28, 28, device=device)

    intermediates = [x_t.clone()]
    save_interval = diffusion.timesteps // save_steps

    for t in reversed(range(diffusion.timesteps)):
        x_t = diffusion.p_sample(model, x_t, t)

        # 定期的に中間状態を保存 / Save intermediate states periodically
        if t % save_interval == 0:
            intermediates.append(x_t.clone())

        if t % 100 == 0:
            print(f"  Sampling step {diffusion.timesteps - t}/{diffusion.timesteps}")

    intermediates.append(x_t.clone())  # 最終結果 / Final result

    return x_t, intermediates


def save_generation_process(intermediates, output_path):
    """
    生成過程を画像として保存 / Save generation process as image

    【日本語】
    各ステップでの画像を並べて表示します。
    左から右に向かってノイズが減少していく様子がわかります。

    【English】
    Display images at each step side by side.
    Shows how noise decreases from left to right.
    """
    n_steps = len(intermediates)
    n_samples = min(4, intermediates[0].shape[0])

    fig, axes = plt.subplots(n_samples, n_steps, figsize=(2 * n_steps, 2 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        for j, intermediate in enumerate(intermediates):
            img = intermediate[i, 0].cpu().numpy()
            # [-1, 1] -> [0, 1]
            img = (img + 1) / 2
            img = img.clip(0, 1)

            axes[i][j].imshow(img, cmap='gray')
            axes[i][j].axis('off')

            if i == 0:
                if j == 0:
                    axes[i][j].set_title('Noise\nノイズ', fontsize=8)
                elif j == n_steps - 1:
                    axes[i][j].set_title('Result\n結果', fontsize=8)

    plt.suptitle('Generation Process: Noise → Image\n生成過程: ノイズ → 画像')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generation process saved to: {output_path}")


def interactive_demo():
    """
    インタラクティブデモ / Interactive demo

    【日本語】
    学習済みモデルがない場合でも、Diffusionの仕組みを
    視覚的に理解するためのデモです。

    【English】
    A demo to visually understand how Diffusion works,
    even without a trained model.
    """
    print("=" * 60)
    print("Diffusion Process Demo / Diffusion過程のデモ")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =================================================================
    # Forward Process（ノイズを加える）のデモ
    # Demo of Forward Process (adding noise)
    # =================================================================
    print("\n--- Forward Process Demo ---")
    print("Adding noise to a clean image step by step...")
    print("きれいな画像に段階的にノイズを加えていきます...")

    # ダミーの「きれいな画像」を作成 / Create a dummy "clean image"
    # 簡単な模様 / Simple pattern
    x_0 = torch.zeros(1, 1, 28, 28, device=device)
    x_0[:, :, 10:18, 10:18] = 1.0  # 中央に四角 / Square in center

    diffusion = DiffusionModel(timesteps=1000, device=device)

    # 異なるタイムステップでノイズを加える / Add noise at different timesteps
    timesteps_to_show = [0, 100, 250, 500, 750, 999]

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(12, 3))

    for i, t in enumerate(timesteps_to_show):
        t_tensor = torch.tensor([t], device=device)
        x_t = diffusion.q_sample(x_0, t_tensor)

        img = x_t[0, 0].cpu().numpy()
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f't={t}')
        axes[i].axis('off')

    plt.suptitle('Forward Process: Clean Image → Noise\n拡散過程: きれいな画像 → ノイズ')
    plt.tight_layout()
    plt.savefig('forward_process_demo.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Forward process demo saved to: forward_process_demo.png")
    print()
    print("=" * 60)
    print("Demo completed! / デモ完了！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate images with TinyDiffusion / TinyDiffusionで画像を生成'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pt',
                        help='Path to checkpoint / チェックポイントのパス')
    parser.add_argument('--n-samples', type=int, default=16,
                        help='Number of samples to generate / 生成する画像数')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Sampling timesteps / サンプリングステップ数')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    parser.add_argument('--output', type=str, default='generated_samples.png',
                        help='Output path / 出力パス')
    parser.add_argument('--show-process', action='store_true',
                        help='Show generation process / 生成過程を表示')
    parser.add_argument('--demo', action='store_true',
                        help='Run interactive demo / デモを実行')

    args = parser.parse_args()

    print("=" * 60)
    print("TinyDiffusion Generation / TinyDiffusion 生成")
    print("=" * 60)

    if args.demo:
        interactive_demo()
    else:
        generate(
            checkpoint_path=args.checkpoint,
            n_samples=args.n_samples,
            timesteps=args.timesteps,
            device=args.device,
            output_path=args.output,
            show_process=args.show_process
        )
