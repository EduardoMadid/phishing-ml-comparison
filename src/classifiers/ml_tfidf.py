import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

if __name__ == "__main__":
    print("Carregando dados...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train_text = df_train["URL"]
    y_train = df_train["Label"]
    X_test_text = df_test["URL"]
    y_test = df_test["Label"]

    # Vetorizador: quebra URLs em n-grams de caracteres (3 a 5 chars)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=10000,
    )

    print("Vetorizando (fit no treino)...")
    start = time.perf_counter()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    vec_time = time.perf_counter() - start
    print(f"Vetorização: {vec_time:.2f}s | Vocabulário: {len(vectorizer.vocabulary_)} n-grams")

        # Treinar
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    print("Treinando...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # Prever
    start = time.perf_counter()
    predictions = model.predict(X_test)
    pred_time = time.perf_counter() - start

    print(f"\nTreino: {train_time:.2f}s | Predição: {pred_time:.2f}s")
    print("\n" + classification_report(y_test, predictions, digits=4))

    # --- Interpretabilidade: top n-grams por classe ---
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]  # pesos (classe positiva = 'good')

    # Junta n-gram com seu peso e ordena
    ranked = sorted(zip(feature_names, weights), key=lambda x: x[1])

    print("\nTop 15 n-grams que indicam PHISHING (peso mais negativo):")
    for ngram, w in ranked[:15]:
        print(f"  {repr(ngram):<12} {w:.3f}")

    print("\nTop 15 n-grams que indicam LEGÍTIMO (peso mais positivo):")
    for ngram, w in ranked[-15:][::-1]:
        print(f"  {repr(ngram):<12} {w:.3f}")