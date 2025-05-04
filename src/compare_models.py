# src/compare_models.py

import os
import time
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.dataCleaning import clean_data
from src.model_training import load_labels
from src.model_serialization import save_model
from src.vectorization_selection import find_best_vectorization
from src.dataVectorizations import vectorize_texts

# Імпорти ваших scratch-реалізацій
from src.ml_algorithms.random_forest import RandomForestScratch
from src.ml_algorithms.svm_scratch import SVMScratch
from src.ml_algorithms.logistic_regression_scratch import LogisticRegressionScratch
# from src.ml_algorithms.naive_bayes_scratch import NaiveBayesScratch


def _train_for_algo(name, factory, vecs, y, models_dir):
    """
    Навчання одного алгоритму на всіх векторизаціях, пошук найкращої та збереження.
    Повертає (name, best_vectorization_key, best_accuracy, duration_sec).
    """
    start_time = time.time()

    best_key, best_score, (best_clf, best_vec) = find_best_vectorization(
        classifier_factory=factory,
        algo_name=name,
        vecs=vecs,
        y=y
    )
    # Зберігаємо модель + векторизатор
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, f"{name}_{best_key}.pkl")
    save_model((best_clf, best_vec), save_path)

    duration = time.time() - start_time
    return name, best_key, best_score, duration


def main(
    csv_file: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_to_be_cleansed.csv'),
    models_dir: str = os.path.join(os.path.dirname(__file__), '..', 'models')
):
    # 1) Підготовка даних та одноразова векторизація
    texts = clean_data(csv_file)
    y = load_labels(csv_file)
    vecs = vectorize_texts(texts)

    # 2) Фабрики алгоритмів (pickl-совні через partial)
    algorithms = {
        'RandomForestScratch': partial(
            RandomForestScratch,
            n_estimators=800,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.8,
            bootstrap=True,
            random_state=42
        ),
        'SVMScratch': partial(
            SVMScratch,
            lambda_param=0.0001,
            epochs=100,
            batch_size=128
        ),
        'LogisticRegressionScratch': partial(
            LogisticRegressionScratch,
            learning_rate=0.01,
            n_iters=1000,
            lambda_param=0.01
        ),
        # 'NaiveBayesScratch': partial(
        #     NaiveBayesScratch,
        #     alpha=1.0
        # ),
    }

    # 3) Перевіряємо, які моделі вже готові, а які потребують тренування
    os.makedirs(models_dir, exist_ok=True)
    to_train = []
    summary = []
    for name, factory in algorithms.items():
        existing = [f for f in os.listdir(models_dir)
                    if f.startswith(name + '_') and f.endswith('.pkl')]
        if existing:
            # модель уже існує — пропускаємо тренування
            filename = existing[0]
            best_key = filename.split(name + '_')[1].replace('.pkl','')
            summary.append((name, best_key, None, None))
            print(f"Found existing {name} model: {filename} — skipping.")
        else:
            to_train.append((name, factory))

    # 4) Паралельне тренування моделей, що лишилися
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_train_for_algo, name, factory, vecs, y, models_dir): name
            for name, factory in to_train
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                name, best_key, best_score, duration = fut.result()
                summary.append((name, best_key, best_score, duration))
                print(f"{name}: best vector={best_key}, acc={best_score:.4f}, time={duration:.1f}s")
            except Exception as e:
                print(f"Error training {name}: {e}")

    # 5) Вивід підсумкової таблиці
    print("\nSummary of all algorithms:")
    for name, vec, score, dur in summary:
        if score is None:
            print(f"  {name:<25} -> {vec:<8} (loaded existing)")
        else:
            print(f"  {name:<25} -> {vec:<8} accuracy={score:.4f} time={dur:.1f}s")


if __name__ == '__main__':
    main()
