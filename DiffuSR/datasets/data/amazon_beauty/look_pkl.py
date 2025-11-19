import pickle
import argparse

def convert_pkl_to_txt(input_pkl, output_txt, max_items=300):
    """
    将 pkl 数据集转为 txt 格式，保持结构清晰可读。
    max_items: 每个大字段最多打印多少条数据，避免文件过大
    """
    print(f"[INFO] Loading PKL: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    print("[INFO] Writing TXT:", output_txt)
    with open(output_txt, "w", encoding="utf-8") as out:
        out.write("====== PKL → TXT 转换结果 ======\n\n")

        for key in data:
            out.write(f"\n===== KEY: {key} =====\n")

            value = data[key]

            # dict 类型
            if isinstance(value, dict):
                out.write(f"(dict, size={len(value)})\n")

                # 打印前 max_items 个 sample
                for i, (k,v) in enumerate(value.items()):
                    if i >= max_items:
                        out.write(f"... (剩余 {len(value)-max_items} 项未显示)\n")
                        break
                    out.write(f"{k}: {v}\n")

            # list 类型
            elif isinstance(value, list):
                out.write(f"(list, size={len(value)})\n")
                for i, v in enumerate(value[:max_items]):
                    out.write(f"{i}: {v}\n")
                if len(value) > max_items:
                    out.write(f"... (剩余 {len(value)-max_items} 项未显示)\n")

            # 其他类型
            else:
                out.write(f"({type(value)})\n")
                out.write(str(value) + "\n")

    print(f"[OK] TXT 文件生成完毕：{output_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="use_ifo.pkl", help="输入 pkl 文件路径")
    parser.add_argument("--output", default="dataset.txt", help="输出 txt 文件路径")
    parser.add_argument("--max_items", type=int, default=300,
                        help="每个字段最多打印多少条数据")

    args = parser.parse_args()
    convert_pkl_to_txt(args.input, args.output, args.max_items)


if __name__ == "__main__":
    main()
