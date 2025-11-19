# preprocess_user_fft_ids.py
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm


# =====================================================
# Step 1: 检测用户序列的主周期（基于 item ID 序列）
# =====================================================
def detect_period_idseq(seq):
    """
    seq: [i1, i2, ..., iL]  (item IDs, int)
    使用 ID 序列直接进行 FFT，检测周期峰值
    返回: 周期 P（可能为 None）
    """
    L = len(seq)
    if L < 8:
        return None

    # 转成 float32 做 FFT
    arr = torch.tensor(seq, dtype=torch.float32)

    # 一维 FFT
    fft_vals = torch.fft.rfft(arr)          # [F]
    power = fft_vals.abs() ** 2             # 能量谱

    # 去掉直流分量（频率 0 对应整体平均水平）
    power[0] = 0

    peak = torch.argmax(power).item()
    if peak <= 1:
        return None

    # 周期 = 序列长度 / 主频索引
    P = L / peak
    return P


# =====================================================
# Step 2: 计算三种频率参考（低 / 中 / 高频）
# =====================================================
def compute_freq_refs(seq, P=None):
    """
    seq: 用户行为序列 (list[int])
    P: 主周期（可能为 None）

    返回:
      - freq_global: 全序列 FFT 频谱
      - freq_short : 最近 K 步 FFT 频谱（短期行为）
      - freq_period: 周期窗口 FFT 频谱（中频行为）
    """
    L = len(seq)
    arr = torch.tensor(seq, dtype=torch.float32)

    # 1) 全局频谱（低频趋势）
    fft_global = torch.fft.rfft(arr).abs()      # [F_g]

    # 2) 高频：最近 K 步（短期记忆）
    K = min(16, L)  # 你可以调，比如 32
    fft_short = torch.fft.rfft(arr[-K:]).abs()  # [F_s]

    # 3) 中频：按周期截一个窗口（如果有）
    if P is not None and P >= 8:
        start = int(max(0, L - P))
        fft_period = torch.fft.rfft(arr[start:]).abs()
    else:
        # 如果没有有效周期，就用短期频谱代替
        fft_period = fft_short.clone()

    return {
        "freq_global": fft_global.numpy(),
        "freq_short": fft_short.numpy(),
        "freq_period": fft_period.numpy(),
        "F_len": len(fft_global)
    }


# =====================================================
# Step 3: 主处理流程
#   读取 data.pkl → 遍历用户 → 计算周期 + 频谱参考 → 存 pkl
# =====================================================
def preprocess_user_fft(input_pkl: str, output_pkl: str):
    print(f"[INFO] Loading dataset: {input_pkl}")

    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    # 你的数据结构：
    #   data["train"][u] = [i1, i2, ...]
    #   data["val"][u]   = [iv]
    #   data["test"][u]  = [it]
    train = data["train"]
    val = data["val"]

    user_freq_info = {}
    num_users = len(train)

    print(f"[INFO] #users in train: {num_users}")
    print("[INFO] Using sequence = train[u] + val[u] 作为用户历史（不包含 test）")

    for uid in tqdm(train.keys(), desc="Processing users"):
        seq_train = train[uid]   # list[int]
        seq_val = val[uid]       # list[int]

        # 用户完整历史（和模型训练/测试阶段对齐）
        full_seq = list(seq_train) + list(seq_val)

        if len(full_seq) < 4:
            # 序列太短，不做 FFT
            continue

        # 1) 检测主周期
        P = detect_period_idseq(full_seq)

        # 2) 计算三段频率参考
        refs = compute_freq_refs(full_seq, P)

        user_freq_info[uid] = {
            "P": P,
            "freq_global": refs["freq_global"],
            "freq_short": refs["freq_short"],
            "freq_period": refs["freq_period"],
            "F_len": refs["F_len"],
            "seq_len": len(full_seq),
        }

    print(f"[INFO] Processed users: {len(user_freq_info)} / {num_users}")
    print(f"[INFO] Saving user FFT info to: {output_pkl}")

    with open(output_pkl, "wb") as f:
        pickle.dump(user_freq_info, f)

    print("[OK] Done.")


# =====================================================
# Step 4: main 函数 + 命令行入口
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="预处理用户行为序列，计算 FFT 频谱 + 周期信息，并保存为 pkl"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="原始数据集 pkl 路径，例如: dataset.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="user_freq_info.pkl",
        help="输出的用户频谱信息 pkl 路径，默认: user_freq_info.pkl",
    )

    args = parser.parse_args()

    preprocess_user_fft(args.input, args.output)


if __name__ == "__main__":
    main()
