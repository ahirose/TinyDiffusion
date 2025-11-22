# TinyDiffusion

**最もシンプルなDiffusionモデル実装 / The Simplest Diffusion Model Implementation**

教育目的で作成された、理解しやすいDiffusionモデルの実装です。
An easy-to-understand Diffusion model implementation created for educational purposes.

---

## 目次 / Table of Contents

1. [Diffusionモデルとは？ / What is a Diffusion Model?](#diffusionモデルとは--what-is-a-diffusion-model)
2. [プロジェクト構成 / Project Structure](#プロジェクト構成--project-structure)
3. [使い方 / How to Use](#使い方--how-to-use)
4. [仕組みの詳細解説 / Detailed Explanation](#仕組みの詳細解説--detailed-explanation)
5. [コードの読み方 / How to Read the Code](#コードの読み方--how-to-read-the-code)

---

## Diffusionモデルとは？ / What is a Diffusion Model?

### 日本語

Diffusionモデルは、**ノイズを除去することを学習する**ことで画像を生成する深層学習モデルです。

#### 基本的なアイデア

想像してみてください：
1. きれいな写真があります
2. その写真に少しずつ砂をかけていきます
3. 最終的には、写真は完全に砂に埋もれて見えなくなります

Diffusionモデルは、この**逆の過程**を学習します：
1. 砂だらけの状態（ノイズ）から始めて
2. 少しずつ砂（ノイズ）を取り除き
3. 最終的にきれいな画像を復元します

#### なぜ「Diffusion（拡散）」？

物理学の「拡散」現象から名前が来ています。インクが水に広がるように、情報（画像）がノイズに「拡散」していくイメージです。

### English

A Diffusion model is a deep learning model that generates images by **learning to remove noise**.

#### Basic Idea

Imagine this:
1. You have a clean photo
2. You gradually sprinkle sand on it
3. Eventually, the photo is completely buried in sand and invisible

A Diffusion model learns **the reverse process**:
1. Start from a sandy state (noise)
2. Gradually remove sand (noise)
3. Eventually recover a clean image

#### Why "Diffusion"?

The name comes from the physics phenomenon of "diffusion". Like ink spreading in water, information (image) "diffuses" into noise.

---

## プロジェクト構成 / Project Structure

```
TinyDiffusion/
├── tiny_diffusion.py  # コアとなる実装（詳細コメント付き）
│                       # Core implementation (with detailed comments)
│
├── train.py           # 学習スクリプト
│                       # Training script
│
├── generate.py        # 画像生成スクリプト
│                       # Image generation script
│
├── requirements.txt   # 依存パッケージ
│                       # Dependencies
│
└── README.md          # このファイル
                        # This file
```

---

## 使い方 / How to Use

### 1. 環境構築 / Setup

```bash
# 依存パッケージをインストール / Install dependencies
pip install -r requirements.txt
```

### 2. 学習 / Training

```bash
# 基本的な学習（10エポック）/ Basic training (10 epochs)
python train.py

# カスタム設定での学習 / Training with custom settings
python train.py --epochs 20 --batch-size 128 --lr 1e-4

# GPUを使用 / Use GPU
python train.py --device cuda
```

**出力 / Output:**
- `checkpoints/latest.pt` - 最新のモデル / Latest model
- `checkpoints/samples_epoch_*.png` - 各エポックでの生成サンプル / Samples at each epoch
- `checkpoints/loss_curve.png` - 学習曲線 / Learning curve

### 3. 画像生成 / Generation

```bash
# 基本的な生成 / Basic generation
python generate.py

# 生成過程を可視化 / Visualize generation process
python generate.py --show-process

# 多くの画像を生成 / Generate more images
python generate.py --n-samples 64

# デモモード（学習済みモデルなしでも動作）
# Demo mode (works without trained model)
python generate.py --demo
```

### 4. コードの動作確認 / Test the Code

```bash
# tiny_diffusion.py の動作確認 / Test tiny_diffusion.py
python tiny_diffusion.py
```

---

## 仕組みの詳細解説 / Detailed Explanation

### Forward Process（拡散過程）/ Forward Process

#### 日本語

きれいな画像 `x_0` にノイズを加えていく過程です。

**数式:**
```
x_t = √(ᾱ_t) × x_0 + √(1 - ᾱ_t) × ε
```

- `x_t`: 時刻 t でのノイズ画像
- `x_0`: 元のきれいな画像
- `ᾱ_t`: 「信号」がどれだけ残っているか（t が大きいほど小さい）
- `ε`: ランダムノイズ（標準正規分布）

**直感的な理解:**
- `t = 0`: ほぼ元の画像（ノイズがほとんどない）
- `t = T/2`: 画像とノイズが半々くらい
- `t = T-1`: ほぼ完全なノイズ（元の画像がほとんど見えない）

#### English

The process of adding noise to a clean image `x_0`.

**Formula:**
```
x_t = √(ᾱ_t) × x_0 + √(1 - ᾱ_t) × ε
```

- `x_t`: Noisy image at time t
- `x_0`: Original clean image
- `ᾱ_t`: How much "signal" remains (smaller as t increases)
- `ε`: Random noise (standard normal distribution)

**Intuitive understanding:**
- `t = 0`: Almost original image (almost no noise)
- `t = T/2`: About half image and half noise
- `t = T-1`: Almost pure noise (original image barely visible)

---

### Reverse Process（逆拡散過程）/ Reverse Process

#### 日本語

ノイズ画像から元の画像を復元する過程です。これが**学習の目標**です。

**アイデア:**
1. ノイズ画像 `x_t` を見て、そこに含まれるノイズ `ε` を予測する
2. 予測したノイズを引き算して、少しきれいな `x_{t-1}` を得る
3. これを `t = T-1` から `t = 0` まで繰り返す

**数式（簡略版）:**
```
x_{t-1} = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ) + σ_t × z
```

- `ε_θ`: ニューラルネットワークが予測したノイズ
- `z`: 確率的なランダムノイズ（t > 0 の場合）

#### English

The process of recovering the original image from a noisy image. This is the **learning objective**.

**Idea:**
1. Look at noisy image `x_t` and predict the noise `ε` it contains
2. Subtract the predicted noise to get slightly cleaner `x_{t-1}`
3. Repeat from `t = T-1` to `t = 0`

**Formula (simplified):**
```
x_{t-1} = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ) + σ_t × z
```

- `ε_θ`: Noise predicted by the neural network
- `z`: Stochastic random noise (when t > 0)

---

### U-Net（ニューラルネットワーク）/ U-Net (Neural Network)

#### 日本語

U-Netは「ノイズを予測する」ニューラルネットワークです。

**なぜU-Net？**
- 入力と出力が同じサイズ（画像 → ノイズ画像）
- Skip connections で細かい情報を保持
- 画像のセグメンテーションなどで実績がある

**構造:**
```
入力画像 (28×28)
    ↓
[Encoder] 特徴抽出 + ダウンサンプリング
    ↓
(14×14) → (7×7)
    ↓
[Middle] 最も圧縮された状態で処理
    ↓
[Decoder] アップサンプリング + Skip connections
    ↓
(14×14) → (28×28)
    ↓
出力（予測ノイズ）
```

**時間情報の組み込み:**
- タイムステップ `t` を Sinusoidal Embedding で高次元ベクトルに変換
- 各ブロックで画像特徴に加算

#### English

U-Net is a neural network that "predicts noise".

**Why U-Net?**
- Input and output are the same size (image → noise image)
- Skip connections preserve fine details
- Proven track record in image segmentation, etc.

**Structure:**
```
Input image (28×28)
    ↓
[Encoder] Feature extraction + downsampling
    ↓
(14×14) → (7×7)
    ↓
[Middle] Process at most compressed state
    ↓
[Decoder] Upsampling + Skip connections
    ↓
(14×14) → (28×28)
    ↓
Output (predicted noise)
```

**Incorporating time information:**
- Convert timestep `t` to high-dimensional vector using Sinusoidal Embedding
- Add to image features at each block

---

### 損失関数 / Loss Function

#### 日本語

**Simple Loss（シンプル損失）:**
```
L = E[||ε - ε_θ(x_t, t)||²]
```

つまり：「本当のノイズ」と「予測したノイズ」の差の二乗平均

**なぜこれで学習できる？**
- モデルは様々な `t` で様々なノイズレベルの画像を見る
- 各レベルで「どのくらいノイズがあるか」を学習
- 結果として、任意のノイズレベルから少しずつノイズを取り除けるようになる

#### English

**Simple Loss:**
```
L = E[||ε - ε_θ(x_t, t)||²]
```

In other words: Mean squared difference between "true noise" and "predicted noise"

**Why does this work?**
- Model sees images with various noise levels at various `t`
- Learns "how much noise is present" at each level
- As a result, can gradually remove noise from any noise level

---

## コードの読み方 / How to Read the Code

### 推奨する読む順序 / Recommended Reading Order

1. **`tiny_diffusion.py`** を最初に読む
   - `DiffusionModel` クラス: 拡散過程の数学
   - `SimpleUNet` クラス: ニューラルネットワーク
   - `diffusion_loss` 関数: 損失計算

2. **`train.py`** で学習フローを理解
   - データ読み込み
   - 学習ループ
   - チェックポイント保存

3. **`generate.py`** で生成フローを理解
   - モデル読み込み
   - サンプリング過程

### 各ファイルの重要な関数 / Key Functions in Each File

#### tiny_diffusion.py

| 関数/クラス | 説明（日本語） | Description (English) |
|------------|--------------|----------------------|
| `linear_beta_schedule` | βスケジュール（線形） | Linear β schedule |
| `DiffusionModel.q_sample` | ノイズを加える | Add noise to image |
| `DiffusionModel.p_sample` | ノイズを取り除く（1ステップ） | Remove noise (1 step) |
| `DiffusionModel.sample` | 完全な画像生成 | Complete image generation |
| `SimpleUNet.forward` | ノイズ予測 | Predict noise |
| `diffusion_loss` | 損失計算 | Calculate loss |

#### train.py

| 関数 | 説明（日本語） | Description (English) |
|-----|--------------|----------------------|
| `train` | メイン学習ループ | Main training loop |
| `generate_samples` | 学習中のサンプル生成 | Generate samples during training |

#### generate.py

| 関数 | 説明（日本語） | Description (English) |
|-----|--------------|----------------------|
| `generate` | 画像生成 | Generate images |
| `sample_with_intermediates` | 中間状態を保存しながら生成 | Generate with intermediate states |
| `interactive_demo` | Forward processのデモ | Demo of forward process |

---

## 参考文献 / References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Understanding Diffusion Models: A Unified Perspective (Luo, 2022)](https://arxiv.org/abs/2208.11970)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

---

## ライセンス / License

MIT License

---

**Happy Learning! / 学習を楽しんでください！**
