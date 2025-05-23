import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize_texts(cleaned_texts, w2v_size=100, random_state=42):

    boW_vocab = {}
    for doc in cleaned_texts:
        for token in doc.split():
            if token not in boW_vocab:
                boW_vocab[token] = len(boW_vocab)

    data = []
    rows = []
    cols = []
    for i, doc in enumerate(cleaned_texts):
        for token in doc.split():
            if token in boW_vocab:
                rows.append(i)
                cols.append(boW_vocab[token])
                data.append(1)
    X_BoW = csr_matrix((data, (rows, cols)), shape=(len(cleaned_texts), len(boW_vocab)))
    print("BoW Vectorization shape:", X_BoW.shape)


    ngram_vocab = {}
    for doc in cleaned_texts:
        tokens = doc.split()
        for t in tokens:
            if t not in ngram_vocab:
                ngram_vocab[t] = len(ngram_vocab)
        for j in range(len(tokens) - 1):
            bg = tokens[j] + ' ' + tokens[j + 1]
            if bg not in ngram_vocab:
                ngram_vocab[bg] = len(ngram_vocab)

    data = []
    rows = []
    cols = []
    for i, doc in enumerate(cleaned_texts):
        tokens = doc.split()
        for t in tokens:
            if t in ngram_vocab:
                rows.append(i)
                cols.append(ngram_vocab[t])
                data.append(1)
        for j in range(len(tokens) - 1):
            bg = tokens[j] + ' ' + tokens[j + 1]
            if bg in ngram_vocab:
                rows.append(i)
                cols.append(ngram_vocab[bg])
                data.append(1)
    X_ngrams = csr_matrix((data, (rows, cols)), shape=(len(cleaned_texts), len(ngram_vocab)))
    print("N-grams Vectorization shape:", X_ngrams.shape)



    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)
    print("TF-IDF Vectorization shape:", X_tfidf.shape)


    np.random.seed(random_state)
    w2v_vocab = {term: np.random.uniform(-0.5, 0.5, size=w2v_size) for term in boW_vocab}
    X_w2v = np.zeros((len(cleaned_texts), w2v_size), dtype=float)
    for i, doc in enumerate(cleaned_texts):
        tokens = doc.split()
        vectors = [w2v_vocab[t] for t in tokens if t in w2v_vocab]
        if vectors:
            X_w2v[i] = np.mean(vectors, axis=0)
    print("Word2Vec Embedding shape:", X_w2v.shape)

    return {
        'BoW': (boW_vocab, X_BoW),
        'ngrams': (ngram_vocab, X_ngrams),
        'tfidf': (tfidf_vectorizer, X_tfidf),
        'word2vec': (w2v_vocab, X_w2v),
    }
