import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.features import extract_features

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

def load_xy(path):
    df = pd.read_csv(path)
    X = [extract_features(url) for url in df['URL']]
    y = df['Label']
    return X, y

if __name__ == "__main__":
    print("Carregando dados...")
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)
    print(f"Treinamento: {len(X_train)} URLs, Teste: {len(X_test)} URLs")

    # Normalizar features (fit só no treino — evita data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 1. Cria o modelo
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # 2. Treina o modelo(aprende os pesos a partir de X_train e y_train)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # 3. Prever no test set
    start = time.perf_counter()
    predictions = model.predict(X_test)
    pred_time = time.perf_counter() - start

    feature_names = ["length", "dots", "hyphens", "digits", "slashes", "at", "query", "has_ip"]
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nImportância das features:")
    for name, imp in importances:
        print(f"  {name:<10} {imp:.4f}")

    print(f"\nTreino: {train_time:.2f}s | Predição: {pred_time:.2f}s")
    print("\n" + classification_report(y_test, predictions, digits=4))