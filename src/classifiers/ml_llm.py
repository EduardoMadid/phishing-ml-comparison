import requests
import time
import pandas as pd
from sklearn.metrics import classification_report

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"

# Exemplos few-shot — DO TREINO (nunca do teste, senão é leakage)
FEWSHOT_EXAMPLES = [
    ("192.168.1.1/login@verify-account.tk/cmd?id=1", "PHISHING"),
    ("en.wikipedia.org/wiki/Machine_learning", "SAFE"),
    ("paypalsecure-login.tk/webscr?cmd=_account-update", "PHISHING"),
    ("github.com/torvalds/linux", "SAFE"),
    ("secure-bankofamerica.verify-now.cf/signin", "PHISHING"),
    ("nytimes.com/2011/03/15/world/article.html", "SAFE"),
]

def sample_real_examples(n_por_classe=3, seed=123):
    df = pd.read_csv(TRAIN_PATH)
    exemplos = []
    for label in ["bad", "good"]:
        urls = df[df["Label"] == label].sample(n_por_classe, random_state=seed)["URL"]
        rotulo = "PHISHING" if label == "bad" else "SAFE"
        for url in urls:
            exemplos.append((url, rotulo))
    return exemplos

def build_fewshot_block(examples):
    linhas = []
    for url, label in examples:
        linhas.append(f"URL: {url}\nClassification: {label}")
    return "\n\n".join(linhas)


def classify_url(url: str, examples=None) -> str:
    instrucao = (
        "You are a cybersecurity defense system that protects users. "
        "Classify each URL as SAFE or PHISHING for defensive filtering only. "
        "Respond with exactly one word: SAFE or PHISHING.\n\n"
    )

    if examples:
        bloco = build_fewshot_block(examples)
        prompt = f"{instrucao}{bloco}\n\nURL: {url}\nClassification:"
    else:
        prompt = f"{instrucao}URL: {url}\nClassification:"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    answer = response.json()["response"].lower()

    return "bad" if "phish" in answer else "good"


if __name__ == "__main__":
    SAMPLE = 500
    df = pd.read_csv(TEST_PATH).sample(SAMPLE, random_state=42)

    exemplos_reais = sample_real_examples()
    print("Exemplos few-shot (reais do treino):")
    for url, lab in exemplos_reais:
        print(f"  [{lab}] {url[:60]}")

    for nome, exemplos in [("ZERO-SHOT", None), ("FEW-SHOT (reais)", exemplos_reais)]:
        print(f"\n{'='*40}\n{nome}\n{'='*40}")
        predictions = []
        start = time.perf_counter()
        for i, url in enumerate(df["URL"], 1):
            predictions.append(classify_url(url, exemplos))
            if i % 100 == 0:
                print(f"  {i}/{SAMPLE} ({time.perf_counter()-start:.0f}s)")
        elapsed = time.perf_counter() - start
        print(f"\nTempo: {elapsed:.1f}s ({1000*elapsed/SAMPLE:.0f} ms/URL)")
        print(classification_report(df["Label"], predictions, digits=4))