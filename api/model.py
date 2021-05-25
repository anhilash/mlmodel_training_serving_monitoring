import argparse
import gzip
import json
from typing import Iterable
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import joblib
import logging


class Model:
    def __init__(self):
        self.vec = CountVectorizer()
        self.tfidf = TfidfTransformer()
        self.model = MultinomialNB()

    def train(self, train_data: Iterable[dict]):
        counts = self.vec.fit_transform([x["text"] for x in train_data])
        tfidf = self.tfidf.fit_transform(counts)
        self.model.fit(tfidf, [x["label"] for x in train_data])
        with open("./model.joblib", 'wb') as model_jb:
            joblib.dump((self.vec, self.tfidf, self.model), model_jb)

    def predict(self, data: Iterable[dict]):
        return self.model.predict(
            self.tfidf.transform(self.vec.transform([x["text"] for x in data]))
        )


def load_dataset(path):
    data = []
    with gzip.open(path, "rb") as f_in:
        for line in f_in:
            data.append(json.loads(line))
    return data


def main(args):
    logging.info("in main******")
    train_data = load_dataset(args.train)
    model = Model()
    model.train(train_data)

    test_data = load_dataset(args.test)

    example = test_data[0]
    prediction = model.predict([example])[0]
    print(type(model))
    print("\nExample:")
    print(example)
    print(f"\nPredicted label: {prediction}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default="train.jsonl.gz",
        help="path to training data (.jsonl.gz file)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="test.jsonl.gz",
        help="path to test data (.jsonl.gz file)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
