import os
import json
import shutil
from datasets import load_dataset, DatasetDict
from PIL import Image as PILImage

def ner_to_text_sequence(words, ner_tags):
    """NERタグ付きトークン列から Donut 用 text_sequence を生成"""
    items = []
    current_item = {}

    for word, tag in zip(words, ner_tags):
        if tag == "O":
            continue

        tag_type = tag[2:].lower()  # 例: item, qty, price
        prefix = tag[:2]

        if prefix == "B-":
            if current_item:
                items.append(current_item)
                current_item = {}
            current_item[tag_type] = word
        elif prefix == "I-":
            if tag_type in current_item:
                current_item[tag_type] += f" {word}"
            else:
                current_item[tag_type] = word

    if current_item:
        items.append(current_item)

    # XML形式で組み立て
    sequence = "<s_receipt>"
    for item in items:
        sequence += "<s_item>"
        for field, value in item.items():
            sequence += f"<{field}>{value}</{field}>"
        sequence += "</s_item>"
    sequence += "</s_receipt>"

    return sequence


class ImageOrder:
    def __init__(self, image_dir_path):
        self.image_dir_path = image_dir_path
        self.imagenum = 0

    def makename(self):
        name = f"cordimg_{self.imagenum:05d}"
        self.imagenum += 1
        return name

def convert_dataset(dataset_name, output_dir, max_samples_per_split=200):
    # データセットを読み込み
    dataset = load_dataset(dataset_name)

    print(f"✅ Loaded dataset: {dataset_name}")

    for split in dataset.keys():
        print(f"🔧 Processing split: {split}")

        image_dir = os.path.join(output_dir, split, "images")
        json_dir = os.path.join(output_dir, split, "ground_truth")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        image_namer = ImageOrder(image_dir)

        count = 0
        for example in dataset[split]:
            if count >= max_samples_per_split:
                break
            print(example)

            image = example["image"]

            if not image:
                print(f"⚠️ No image found for example example['image']")
                break

            file_id ="CORDDATA_" + image_namer.makename()

            # 保存：画像
            dst_image_path = os.path.join(image_dir, f"{file_id}.jpg")
            image.save(dst_image_path, format="JPEG")

            # 保存：JSON（Donut形式）
            json_data = example['ground_truth']
            json_data = json.loads(json_data)      # type: dict

            dst_json_path = os.path.join(json_dir, f"{file_id}.json")
            with open(dst_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            count += 1

        print(f"✅ Saved {count} samples to {os.path.join(output_dir, split)}")


if __name__ == "__main__":
    # ==== パラメータ指定 ====
    DATASET_NAME = "naver-clova-ix/cord-v2"  # データセット名
    OUTPUT_DIR = "./cord_dataset_donut"  # 出力先フォルダ

    convert_dataset(DATASET_NAME, OUTPUT_DIR)
    print("✅ 変換完了。Donut 用フォーマットで保存されました。")
