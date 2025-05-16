import json
from pathlib import Path
from typing import cast

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2.functional import pil_to_tensor, to_grayscale, to_pil_image

from src.domain.receiptItem import ReceiptItem
from src.domain.model import Model


class Dataset(TorchDataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        data: list[ReceiptItem],
        model: Model,
        *,
        training: bool = True,
    ) -> None:
        self.data = data
        self.model = model
        self.training = training

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        receipt_item = self.data[index]

        image = Image.open(receipt_item.image_path)

        pixel_values = self._image_to_tensor(image, random_padding=self.training)
        labels = self._target_string_to_tensor(receipt_item.xml)

        return pixel_values, labels, receipt_item.xml

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def load(
        cls,
        path: Path,
        model: Model,
        *,
        training: bool = True,
    ) -> "Dataset":
        receipt_items: list[ReceiptItem] = []

        for json_path in sorted(path.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as f:
                label_json = json.load(f)

            # 対応する画像パスを推測（.png or .jpg）
            image_path = json_path.with_suffix(".png")
            if not image_path.exists():
                image_path = json_path.with_suffix(".jpg")
            if not image_path.exists():
                print(f"⚠ 画像が見つかりません: {json_path.name}")
                continue  # または raise FileNotFoundError

            # ReceiptItem を直接生成
            receipt_item = ReceiptItem(
                menu_nms=label_json.get("menu_nms", []),
                menu_price=label_json.get("menu_price", ""),
                menu_itemsubtotal=label_json.get("menu_itemsubtotal", ""),
                sub_total_subtotal_price=label_json.get("sub_total_subtotal_price", ""),
                sub_total_discount_price=label_json.get("sub_total_discount_price", ""),
                sub_total_tax_price=label_json.get("sub_total_tax_price", ""),
                total_total_price=label_json.get("total_total_price", ""),
                total_creditcardprice=label_json.get("total_creditcardprice", ""),
                total_menuqty_cnt=label_json.get("total_menuqty_cnt", ""),
            )

            receipt_items.append(receipt_item)

        # 最後にデータセットを返す
        print(f"✅ Loaded {len(receipt_items)} items from {path}")
        return cls(receipt_items, model, training=training)

    def _gray_scaling_image(self, image: Image.Image) -> Image.Image:
        return to_pil_image(to_grayscale(pil_to_tensor(image)))

    def _image_to_tensor(self, image: Image.Image, *, random_padding: bool) -> Tensor:
        preprocess_image = self._gray_scaling_image(image)
        pixel_values = cast(
            Tensor,
            self.model.processor(
                preprocess_image.convert("RGB"),
                random_padding=random_padding,
                return_tensors="pt",
            ).pixel_values,
        )
        return pixel_values.squeeze()

    def _target_string_to_tensor(self, target: str) -> Tensor:
        ignore_id = -100
        input_ids = cast(
            Tensor,
            self.model.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.model.model.config.decoder.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ).input_ids,
        ).squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.model.tokenizer.pad_token_id] = ignore_id

        return labels
