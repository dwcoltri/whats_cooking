#!/usr/bin/env python3
# Jesse, it's time to cook

import argparse
import pandas
import nltk
import re
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def process(cmd_line_args):
    # Ensure we have this for lemmatization
    nltk.download("wordnet")
    # Read in testing and training file as pandas dataframes
    training_data_frame = pandas.read_json("training_file.json")
    testing_data_frame = pandas.read_json(cmd_line_args.test_file)

    # Lemmatize
    training_data_frame["ingredients_string"] = lemmatize(training_data_frame)
    testing_data_frame["ingredients_string"] = lemmatize(testing_data_frame)

    # TFIDF-ize
    training_tfidf, testing_tfidf = tfidf(training_data_frame, testing_data_frame)

    # PREDICT!
    testing_data_frame = predict(
        training_tfidf, testing_tfidf, training_data_frame, testing_data_frame
    )

    output(testing_data_frame, cmd_line_args)


def predict(training_tfidf, testing_tfidf, training_data_frame, testing_data_frame):
    classifier = GridSearchCV(
        LogisticRegression(solver="liblinear", multi_class="auto"),
        param_grid={"C": [1, 10]},
        cv=5,
    )
    classifier = classifier.fit(training_tfidf, training_data_frame["cuisine"])
    testing_data_frame["cuisine"] = classifier.predict(testing_tfidf)
    return testing_data_frame


def tfidf(training_data_frame, testing_data_frame):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=0.57,
        binary=False,
        token_pattern=r"\w+",
        sublinear_tf=False,
    )
    training_tfidf = vectorizer.fit_transform(
        training_data_frame["ingredients_string"]
    ).todense()
    testing_tfidf = vectorizer.transform(testing_data_frame["ingredients_string"])
    return training_tfidf, testing_tfidf


def lemmatize(data_frame):
    lemmatized_ingredients = []
    lemmatized_ingredients = [
        " ".join(
            [
                nltk.stem.WordNetLemmatizer().lemmatize(re.sub("[^A-Za-z]", " ", line))
                for line in lists
            ]
        ).strip()
        for lists in data_frame["ingredients"]
    ]
    return lemmatized_ingredients


def output(data_frame, cmd_line_args):
    data_frame.sort_index()
    # id, cuisine, ingredients
    columns = ["id", "cuisine"]
    if cmd_line_args.ingredients:
        columns.append("ingredients")
    data_frame.to_csv("submission.csv", index=False, columns=columns)


def _with_cmd_line_args(f):
    def wrapper(*args, **kwargs):
        p = argparse.ArgumentParser()
        p.add_argument(
            "-t", "--test-file", help="Path to the json file to test", required=True
        )
        p.add_argument(
            "-i",
            "--ingredients",
            help="Optionally include ingredient list",
            action="store_true",
        )
        return f(p.parse_args(), *args, **kwargs)

    return wrapper


@_with_cmd_line_args
def main(cmd_line_args):
    print("Jesse, WE HAVE TO COOK!")
    process(cmd_line_args)


if __name__ == "__main__":
    main()
