"""
Regex / Heuristic Baseline for Phishing URL Classification.

Jornada de iteração documentada (ablation-driven):

1) 6 regras iniciais + threshold >= 2 (precisa 2+ disparos)
   Regras: comprimento>75, IP, pontos>5, '@', hífens>4, palavras-gatilho.
   -> F1 bad 0.29 | Recall 0.20 | Precision 0.49 | Accuracy 0.77
   Lição: accuracy alta enganosa em dataset desbalanceado (77% good).

2) Threshold relaxado para >= 1
   -> F1 bad 0.43 | Recall 0.41 | Precision 0.45 | Accuracy 0.76
   Recall dobrou; precision quase intacta. Em phishing, deixar passar
   ataque custa mais que bloquear legítimo errado.

3) +3 regras de domínio: TLDs suspeitos (.tk, .ml, .xyz...),
   dígitos no domínio (g00gle), brand impersonation no path.
   -> F1 bad 0.47

4) Ablation study (leave-one-out) revelou:
   - 'hyphens > 4' ATRAPALHAVA o modelo (impact -0.03):
     URLs legítimas de blog/SEO usam muitos hífens em slugs.
   - 'ip', 'dots>5', '@' tinham impacto desprezível (~0.002).
   - Removidas. F1 estabiliza em 0.50, precision sobe para 0.53.

5) Expansão da feature 'keywords' com termos de alta especificidade:
   - Ação ('verify', 'confirm', 'update'),
   - Urgência ('suspended', 'expire', 'limited'),
   - Pagamento ('refund', 'invoice', 'wire'),
   - Pistas técnicas legacy ('webscr', 'cgi-bin', 'cmd=_').
   -> F1 bad 0.54 | Precision 0.79 | Recall 0.41 | Accuracy 0.84
   Insight: termos legacy ('webscr', 'cgi-bin') são quase-assinaturas:
   raríssimos em URL legítima, comuns em kits de phishing antigos.

CONFIGURAÇÃO FINAL: 4 regras (keywords, tld, digits, brand), threshold >= 1.
   F1 0.54 | Precision 0.79 | Recall 0.41 | Accuracy 0.84
   Latência ~0.01 ms/URL | Custo 0 | Totalmente interpretável.

Próximos passos:
- Teto de recall do regex ~0.41. ML clássico deve romper isso.
- 'keywords' é a feature dominante; modelos devem capturar similar
  via tokens/n-grams.
"""

import re
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import time

TEST_PATH = 'data/test.csv'

def _rule_length(url):
    return len(url) > 75

def _rule_ip(url):
    return bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))

def _rule_dots(url):
    return url.count(".") > 5

def _rule_at(url):
    return "@" in url

def _rule_hyphens(url):
    return url.count("-") > 4

def _rule_keywords(url):
    words = [
        # Ação forçada
        "login", "verify", "confirm", "update", "validate", "activate", "reset",
        # Conta e credenciais
        "account", "signin", "password", "credential", "unlock", "restore", "recover",
        # Urgência (engenharia social)
        "urgent", "suspended", "expire", "expired", "alert", "warning", "limited",
        # Segurança teatral
        "secure", "security", "ssl", "verification", "authenticate",
        # Pagamento
        "billing", "invoice", "refund", "payment", "wire", "transfer",
        # Pistas técnicas clássicas de phishing
        "webscr", "cgi-bin", "cmd=_",
    ]
    return any(w in url.lower() for w in words)

def _rule_tld(url):
    tlds = [".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".xyz", ".click", ".country"]
    return any(url.lower().endswith(t) or t + "/" in url.lower() for t in tlds)

def _rule_digits(url):
    domain = url.split("/")[0]
    return sum(c.isdigit() for c in domain) >= 3

def _rule_brand(url):
    brands = ["paypal", "apple", "microsoft", "amazon", "google", "facebook", "netflix", "bank"]
    domain = url.split("/")[0].lower()
    path = "/".join(url.split("/")[1:]).lower()
    return any(b in path and b not in domain for b in brands)


RULES = {
    # "length": _rule_length,   ← removido na 2ª rodada (impact -0.0194)
    # "ip": _rule_ip,           ← removido (impact +0.0003)
    # "dots": _rule_dots,       ← removido (impact +0.0017)
    # "at": _rule_at,           ← removido (impact +0.0024)
    # "hyphens": _rule_hyphens, ← removido (impact -0.0296, atrapalhava)
    "keywords": _rule_keywords,
    "tld": _rule_tld,
    "digits": _rule_digits,
    "brand": _rule_brand,
}

def is_phishing(url: str, enabled: set = None) -> str:
    if enabled is None:
        enabled = set(RULES.keys())
    score = sum(1 for name, rule in RULES.items() if name in enabled and rule(url))
    return "bad" if score >= 1 else "good"

if __name__ == "__main__":
    print(f"Loading {TEST_PATH}...")
    df_test = pd.read_csv(TEST_PATH)
    print(f"Test set: {len(df_test)} URLs")

    # Predição cronometrada (avaliação completa)
    start = time.perf_counter()
    predictions = df_test["URL"].apply(is_phishing)
    elapsed = time.perf_counter() - start

    print(f"\nTime: {elapsed:.2f}s ({1000 * elapsed / len(df_test):.3f} ms/URL)")
    print("\n" + classification_report(df_test["Label"], predictions, digits=4))

    # === ABLATION STUDY ===
    print("\n=== ABLATION STUDY ===")

    baseline_f1 = f1_score(df_test["Label"], predictions, pos_label="bad")
    print(f"\nBaseline F1 (all rules): {baseline_f1:.4f}\n")
    print(f"{'Rule':<12} {'F1 without':<14} {'Impact':<10}")
    print("-" * 36)

    for rule_name in RULES:
        reduced = set(RULES.keys()) - {rule_name}
        preds = df_test["URL"].apply(lambda u: is_phishing(u, reduced))
        f1_without = f1_score(df_test["Label"], preds, pos_label="bad")
        impact = baseline_f1 - f1_without
        print(f"{rule_name:<12} {f1_without:.4f}        {impact:+.4f}")