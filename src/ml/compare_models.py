import os
import csv
import numpy as np
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from src.ml.model_training import (
    preprocess_and_vectorize,
    train_and_save_model,
    fit_and_evaluate,
    combine_text_and_meta
)
from src.ml_algorithms.random_forest import RandomForestScratch
from src.ml_algorithms.svm_scratch import SVMScratch
from src.ml_algorithms.naive_bayes_scratch import NaiveBayesScratch


def find_best_vectorization(
    classifier_factory,
    algo_name: str,
    vecs: dict,
    y: np.ndarray,
    meta_X,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
):

    best_score = -np.inf
    best_key = None

    for key, (vec_obj, X_text) in vecs.items():
        X_full = combine_text_and_meta(X_text, meta_X)
        clf = classifier_factory()
        _, metrics = fit_and_evaluate(
            X_full, y, clf,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        acc = metrics['accuracy']
        print(f"{algo_name} + {key} -> acc={acc}")

        if acc > best_score:
            best_score = acc
            best_key = key

    return best_key, best_score


def main(
    multi_csv: str = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'data_to_be_cleansed.csv'
    ),
    binary_xlsx: str = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'Depression_Text.xlsx'
    ),
    models_dir: str = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'models'
    ),
    results_file: str = 'model_comparison_results.csv'
):
    multi_algos = {
        'RF_Multi_Scratch': partial(RandomForestScratch,
            n_estimators=600, max_depth=60,
            min_samples_leaf=2, max_features=0.8,
            bootstrap=True, random_state=42
        ),
        'SVM_Multi_Scratch': partial(SVMScratch,
            lambda_param=1e-4, epochs=200, batch_size=128
        ),
        'NB_Multi_Scratch': partial(NaiveBayesScratch, alpha=1.0),
        # 'RF_Multi': RandomForestClassifier,
        # 'SVM_Multi': SVC,
        # 'NB_Multi': MultinomialNB
    }
    binary_algos = {
        # 'RF_Binary': RandomForestClassifier,
        # 'SVM_Binary': SVC,
        # 'NB_Binary': MultinomialNB
    }

    os.makedirs(models_dir, exist_ok=True)
    summary = []

    # --- multi-class experiments ---
    vecs_mc, y_mc = preprocess_and_vectorize(multi_csv)
    for name, factory in multi_algos.items():
        best_key, _ = find_best_vectorization(factory, name, vecs_mc, y_mc, meta_X=None)
        out_file = os.path.join(models_dir, f"{name}_{best_key}.pkl")
        if os.path.exists(out_file):
            print(f"Model '{out_file}' exists, skipping {name}.")
            continue

        res = train_and_save_model(multi_csv, factory(), name, best_key, models_dir)
        summary.append(res)
        print(f"Trained {name}, best vec={best_key}, acc={res['accuracy']}")

    # --- binary experiments ---
    vecs_bin, y_bin = preprocess_and_vectorize(binary_xlsx)
    for name, factory in binary_algos.items():
        best_key, _ = find_best_vectorization(factory, name, vecs_bin, y_bin, meta_X=None)
        out_file = os.path.join(models_dir, f"{name}_{best_key}.pkl")
        if os.path.exists(out_file):
            print(f"Model '{out_file}' exists, skipping {name}.")
            continue

        res = train_and_save_model(binary_xlsx, factory(), name, best_key, models_dir)
        summary.append(res)
        print(f"Trained {name}, best vec={best_key}, acc={res['accuracy']}")

    write_header = not os.path.exists(results_file)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'algorithm', 'vectorizer', 'accuracy', 'f1_score',
                'precision', 'recall', 'training_time_sec', 'model_filepath'
            ])

        for r in summary:
            writer.writerow([
                r.get('algorithm'),
                r.get('vectorizer'),
                r.get('accuracy'),
                r.get('f1_score'),
                r.get('precision'),
                r.get('recall'),
                r.get('training_time_sec'),
                r.get('model_filepath')
            ])

    print("All models processed and summary saved to", results_file)


if __name__ == '__main__':
    main()
