import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

SAMPLE_SIZE = 30000

if __name__ == "__main__":
    print("Carregando dados...")
    df_train = pd.read_csv(TRAIN_PATH).sample(SAMPLE_SIZE, random_state=42)
    df_test = pd.read_csv(TEST_PATH).sample(10000, random_state=42)

    print(f"Treino (amostra): {len(df_train)} | Teste (amostra): {len(df_test)}")

    print("Carregando modelo de embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Gerando embeddings (treino)...")
    start = time.perf_counter()
    X_train = model.encode(df_train["URL"].tolist(), show_progress_bar=True)
    X_test = model.encode(df_test["URL"].tolist(), show_progress_bar=True)
    encode_time = time.perf_counter() - start

    y_train = df_train["Label"]
    y_test = df_test["Label"]

    print(f"\nEncoding: {encode_time:.1f}s | Shape: {X_train.shape}")

    model_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    print("Treinando classificador...")
    model_clf.fit(X_train, y_train)

    predictions = model_clf.predict(X_test)
    print("\n" + classification_report(y_test, predictions, digits=4))