import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_PATH = "data/train.csv"
MODEL_PATH = "models/phishing_model.joblib"

if __name__ == "__main__":
    print("Carregando treino...")
    df = pd.read_csv(TRAIN_PATH)

    print("Treinando TF-IDF + LogReg (modelo vencedor)...")
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5, max_features=10000)
    X = tfidf.fit_transform(df["URL"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, df["Label"])

    print(f"Salvando em {MODEL_PATH}...")
    joblib.dump({"vectorizer": tfidf, "model": model}, MODEL_PATH)
    print("Pronto!")