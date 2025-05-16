import os
import json


def flatten_dict(d, parent_key='', sep='_'):
    """ネストされた辞書を1階層に平坦化"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def to_text_sequence(data):
    """Donut形式の text_sequence に変換（リストや辞書も文字列化）"""
    parts = []
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(f"<s_{k}>{v}</s_{k}>")
    return " ".join(parts)

def convert_dataset_jsons(dataset_dir):
    os.makedirs(dataset_dir , exist_ok=True)
    
    for split in ["train", "validation", "test"]:
        input_dir = os.path.join(dataset_dir, split, "ground_truth")

        # 出力先を同じ階層に text_sequence フォルダで出力
        output_split = os.path.join(os.path.dirname(dataset_dir), split)
        os.makedirs(output_split , exist_ok=True)
        output_dir = os.path.join(output_split, "text_sequence")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"⚠️ Skipping missing split directory: {input_dir}")
            continue

        for filename in os.listdir(input_dir):
            if not filename.endswith(".json"):
                continue

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    gt_parse = data["gt_parse"]
                    flat_data = flatten_dict(gt_parse)
                    text_sequence = to_text_sequence(flat_data)

                converted = {
                    "ground_truth": text_sequence
                    
                }

                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(converted, out_f, ensure_ascii=False, indent=2)

                print(f"✅ Converted {split}/{filename}")

            except Exception as e:
                print(f"❌ Failed {split}/{filename}: {e}")

# --- 実行 ---

if __name__ == "__main__":
    # ==== パラメータ指定 ====
    DATASET_DIR = "./cord_dataset_donut"   # 出力元フォルダ（画像・元jsonが入っているディレクトリ）
    convert_dataset_jsons(DATASET_DIR)
