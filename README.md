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

###　学習用のyaml

　基本はそのままで大丈夫だが、以下の項目を追加

    input_size: [1280, 960]
    align_long_axis: False
    train_batch_size: 8
    eval_batch_size: 8
    num_workers: 4
    learning_rate: 1e-5
    adam_epsilon: 1e-8

###　学習用コピペ

    　python3 train.py

    HAGGING FACEの形式では学習ができない（データセットの読み込みができない）直せば動かなくもないかもしれないが　PYTORCH形式でも動かせるので、pytoch形式で学習するようにした。
    