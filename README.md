# donut_playground
githubの公式リポジトリをクローンして、試してみたやつ。


### DATASET 形式

    dataset/
    └── cord/
        ├── train/
        │   ├── receipt_0001.jpg
        │   ├── receipt_0001.json
        │   └── ...
        └── validation/
            ├── receipt_0501.jpg
            ├── receipt_0501.json
            └── ...


###　学習用コピペ

    　python train.py --config config/train_donut-base.yaml