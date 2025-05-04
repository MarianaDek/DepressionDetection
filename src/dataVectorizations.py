# src/dataVectorizations.py

import numpy as np
import re

def vectorize_texts(cleaned_texts, w2v_size=100, random_state=42):
    """
    Ручні реалізації:
      1. Count Vectorization
      2. N-grams Vectorization (уніграми + біграми)
      3. TF-IDF Vectorization
      4. Word2Vec Document Embeddings (рандомні вектори слів)

    Повертає словник: метод -> (vocab, feature_matrix)
    vocab для Word2Vec містить {word: vector}
    """
    # ------------------------
    # 1) Count Vectorization
    # ------------------------
    count_vocab = {}
    for doc in cleaned_texts:
        for token in doc.split():
            if token not in count_vocab:
                count_vocab[token] = len(count_vocab)
    X_count = np.zeros((len(cleaned_texts), len(count_vocab)), dtype=int)
    for i, doc in enumerate(cleaned_texts):
        for token in doc.split():
            X_count[i, count_vocab[token]] += 1
    print("Count Vectorization shape:", X_count.shape)

    # ---------------------------------
    # 2) N-grams Vectorization
    # ---------------------------------
    ngram_vocab = {}
    for doc in cleaned_texts:
        tokens = doc.split()
        # уніграми
        for t in tokens:
            if t not in ngram_vocab:
                ngram_vocab[t] = len(ngram_vocab)
        # біграми
        for j in range(len(tokens) - 1):
            bg = tokens[j] + ' ' + tokens[j+1]
            if bg not in ngram_vocab:
                ngram_vocab[bg] = len(ngram_vocab)
    X_ngrams = np.zeros((len(cleaned_texts), len(ngram_vocab)), dtype=int)
    for i, doc in enumerate(cleaned_texts):
        tokens = doc.split()
        for t in tokens:
            X_ngrams[i, ngram_vocab[t]] += 1
        for j in range(len(tokens) - 1):
            bg = tokens[j] + ' ' + tokens[j+1]
            X_ngrams[i, ngram_vocab[bg]] += 1
    print("N-grams Vectorization shape:", X_ngrams.shape)

    # ------------------------
    # 3) TF-IDF Vectorization
    # ------------------------
    N = len(cleaned_texts)
    df = {term: 0 for term in count_vocab}
    for term in count_vocab:
        for doc in cleaned_texts:
            if term in doc.split():
                df[term] += 1
    idf = {term: np.log((N + 1) / (df_count + 1)) + 1 for term, df_count in df.items()}

    X_tfidf = np.zeros_like(X_count, dtype=float)
    for i, doc in enumerate(cleaned_texts):
        tokens = doc.split()
        len_tokens = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for term, idx in count_vocab.items():
            tf_val = tf.get(term, 0) / len_tokens if len_tokens > 0 else 0.0
            X_tfidf[i, idx] = tf_val * idf[term]
    print("TF-IDF Vectorization shape:", X_tfidf.shape)

    # -------------------------------------------
    # 4) Word2Vec Embeddings (рандомні вектори слів)
    # -------------------------------------------
    np.random.seed(random_state)
    w2v_vocab = {term: np.random.uniform(-0.5, 0.5, size=w2v_size)
                 for term in count_vocab}
    X_w2v = np.zeros((len(cleaned_texts), w2v_size), dtype=float)
    for i, doc in enumerate(cleaned_texts):
        tokens = doc.split()
        vectors = [w2v_vocab[t] for t in tokens if t in w2v_vocab]
        if vectors:
            X_w2v[i] = np.mean(vectors, axis=0)
    print("Word2Vec Embedding shape:", X_w2v.shape)

    return {
        'count':  (count_vocab, X_count),
        'ngrams': (ngram_vocab, X_ngrams),
        'tfidf':  (count_vocab, X_tfidf),
        'word2vec': (w2v_vocab, X_w2v),
    }

