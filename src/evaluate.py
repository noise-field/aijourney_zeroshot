import argparse
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from classifier import ZeroshotClassifier

QUERY = "Текст: {text}\nКатегория: "

# Other labels have 3 or less items
LABELS = [
    "Наука",
    "Технологии",
    "Оружие",
    "Космос",
    "Гаджеты",
    "Транспорт"
]
REFUSE = "Не знаю"

def load_text_data(data_path: Path):
    texts_path = data_path/"texts"
    text_data = list()

    for file in texts_path.iterdir():
        text_data.append((
            file.name.split(".")[0],  # get filename
            file.read_text(encoding="utf-8").strip().split("\n")[0]  # get the text
        ))

    text_data_df = pd.DataFrame(text_data, columns=["textid", "text"])
    return text_data_df

def load_metadata(data_path: Path):
    return pd.read_csv(data_path/"newmetadata.csv", sep="\t")

def get_data_sample(data_path, sample_size, seed):
    texts = load_text_data(data_path)
    meta = load_metadata(data_path)
    data = texts.merge(meta, on="textid")
    data = data[["text", "textrubric"]]
    data = data[data.textrubric.isin(LABELS)]
    if sample_size < 1:
        return data
    
    data = data.sample(data.shape[0], random_state=seed)
    return data[:sample_size]

def evaluate(data_path: Path, model_path: Union[str,Path], sample_size: int=0, seed: int=0):
    data = get_data_sample(data_path, sample_size, seed)
    print("Label distribution:")
    print(data.textrubric.value_counts())
    predictions = []
    print("Loading model...")
    clf = ZeroshotClassifier(model_path, 1024)
    for _, text in tqdm(data.text.items()):
        pred = clf.classify(text, QUERY, LABELS)
        if pred.confident_enough:
            predictions.append(pred.label)
        else:
            predictions.append(REFUSE)
    print(classification_report(data.textrubric, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    parser.add_argument("sample_size", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    evaluate(Path(args.data_path), args.model_path, args.sample_size, args.seed)

