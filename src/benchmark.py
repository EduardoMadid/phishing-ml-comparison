import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from src.features import extract_features
from src.classifiers.regex_baseline import is_phishing
from src.classifiers.ml_llm import classify_url

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SAMPLE = 500

def carregar_dados():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH).sample(SAMPLE, random_state=42)
    return df_train, df_test


def treinar_modelos(df_train):
    print("Treinando TF-IDF + LogReg...")
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5, max_features=10000)
    X_tfidf = tfidf.fit_transform(df_train["URL"])
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_tfidf, df_train["Label"])

    print("Treinando Random Forest (features estruturais)...")
    X_feat = [extract_features(u) for u in df_train["URL"]]
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_feat, df_train["Label"])

    return tfidf, logreg, rf

def montar_preditores(tfidf, logreg, rf):
    # Cada abordagem embrulhada numa função uniforme: recebe URL, devolve "bad"/"good"
    def pred_regex(url):
        return is_phishing(url)

    def pred_tfidf(url):
        X = tfidf.transform([url])
        return logreg.predict(X)[0]

    def pred_rf(url):
        X = [extract_features(url)]
        return rf.predict(X)[0]

    def pred_llm(url):
        return classify_url(url)

    return {
        "Regex": pred_regex,
        "TF-IDF + LogReg": pred_tfidf,
        "Random Forest": pred_rf,
        "LLM (zero-shot)": pred_llm,
    }

def avaliar(preditores, df_test):
    y_true = df_test["Label"]
    resultados = []

    for nome, predizer in preditores.items():
        print(f"Avaliando {nome}...")
        start = time.perf_counter()
        y_pred = [predizer(url) for url in df_test["URL"]]
        elapsed = time.perf_counter() - start

        # precision/recall/f1 da classe "bad"
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["bad"], average="binary", pos_label="bad", zero_division=0
        )
        resultados.append({
            "nome": nome,
            "precision": p,
            "recall": r,
            "f1": f1,
            "ms_por_url": 1000 * elapsed / len(df_test),
        })

    return resultados


def imprimir_tabela(resultados):
    resultados = sorted(resultados, key=lambda x: x["f1"], reverse=True)
    print(f"\n{'='*70}")
    print(f"{'Abordagem':<20} {'F1':>7} {'Precision':>10} {'Recall':>8} {'ms/URL':>10}")
    print("-" * 70)
    for r in resultados:
        print(f"{r['nome']:<20} {r['f1']:>7.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['ms_por_url']:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    df_train, df_test = carregar_dados()
    print(f"Amostra de teste: {len(df_test)} URLs\n")

    tfidf, logreg, rf = treinar_modelos(df_train)
    preditores = montar_preditores(tfidf, logreg, rf)
    resultados = avaliar(preditores, df_test)
    imprimir_tabela(resultados)