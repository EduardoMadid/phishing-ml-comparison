# Phishing ML Comparison

**🌐 Idioma / Language:** **Português** · [English](README.md)

> Detecção de phishing apenas pela URL — comparando **4 abordagens** (regex → ML clássico → embeddings → LLM local) sobre os **mesmos dados**, medindo F1, latência e custo. O objetivo não é o modelo — é a medição honesta de *quando cada abordagem compensa*.

![Comparação do benchmark](benchmark_comparison.png)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-multi--stage-blue)

---

## 🎯 Resumo — a tabela que diz tudo

As quatro abordagens avaliadas na **mesma amostra de 500 URLs** (`random_state=42`), para a comparação ser rigorosamente justa. O tamanho da amostra é limitado pela abordagem mais lenta (o LLM).

| Abordagem | F1 (phishing) | Precision | Recall | ms / URL | Custo |
|-----------|:-------------:|:---------:|:------:|:--------:|:-----:|
| **TF-IDF + LogReg** 🥇 | **0.884** | 0.839 | 0.934 | 0.34 | $0 |
| Random Forest (8 features) | 0.670 | 0.621 | 0.726 | 17.34 | $0 |
| LLM — Llama 3.2 3B (zero-shot) | 0.647 | 0.595 | 0.708 | 275.50 | GPU |
| Regex baseline | 0.566 | **0.783** | 0.443 | **0.01** | $0 |

**Como ler:** o modelo *aprendido* mais barato (TF-IDF + Regressão Logística) vence em F1 *e* é ~800× mais rápido que o LLM. O LLM — a tecnologia mais badalada — ficou em **3º**, abaixo de um Random Forest com 8 features contadas à mão. O regex tem a **maior precisão** de todos (quando ele grita "phishing", costuma acertar), mas o pior recall.

> **O objetivo deste projeto não é o modelo. É a medição honesta dos trade-offs** — quando cada abordagem compensa, com números reais.

---

## ❓ O Problema

URLs de phishing imitam sites legítimos para roubar credenciais. Detectá-las **apenas pela string da URL** (sem conteúdo da página, sem chamadas de rede) é uma primeira linha de defesa barata, usada em navegadores, filtros de e-mail e SOCs. A pergunta que este projeto responde: **qual técnica realmente compensa para essa tarefa — e por quê?**

---

## 🗂️ O Dataset

[Phishing Site URLs — Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls): ~549 mil URLs rotuladas (`good` / `bad`), cruas — **sem features pré-extraídas** (a engenharia de features faz parte do exercício).

O que a EDA revelou (e por que importou):

| Achado | Número | Consequência |
|--------|--------|--------------|
| Duplicatas removidas | 42.150 (→ 507.196 linhas) | duplicatas vazam entre treino/teste e inflam métricas |
| Balanço de classes | 77,5% good / 22,5% bad | **accuracy engana** — um modelo "sempre good" tira 0.775 |
| Comprimento URL (phishing) | média **71** / mediana 47 | URLs de phishing são mais longas → feature útil |
| Comprimento URL (legítima) | média **46** / mediana 40 | — |

O desbalanceamento é o motivo de todo modelo usar `class_weight="balanced"` e split treino/teste **estratificado**, e de a métrica-rei em todo o projeto ser o **F1 da classe phishing**, nunca a accuracy.

---

## 🔬 As 4 Abordagens (a jornada)

### 1. Regex baseline — "o chão"

Regras heurísticas escritas à mão com um limiar de pontuação. Melhoradas **empiricamente**, não por chute:

| Iteração | F1 (phishing) | O que mudou |
|----------|:-------------:|-------------|
| 6 regras, limiar ≥ 2 | 0.29 | início ingênuo |
| limiar ≥ 1 | 0.43 | recall dobrou |
| + 3 regras de domínio | 0.47 | TLDs suspeitos, dígitos no domínio, brand impersonation |
| **limpeza por ablation** | 0.50 | removi a regra `hyphens` — ela **atrapalhava** (-0.03): blogs legítimos usam muitos hífens em slugs |
| keywords expandidas | **0.54** | adicionei tokens legados de phishing (`webscr`, `cgi-bin`, `cmd=_`) → precision saltou pra 0.79 |

Um **estudo de ablation leave-one-out** ranqueou a contribuição de cada regra. Insight-chave: a regra que *parecia* sensata (`hífens demais`) estava prejudicando o modelo — provado por dados, não por intuição.

### 2. ML clássico — features manuais

8 features estruturais (comprimento, nº de pontos, nº de dígitos, etc.) entregues ao scikit-learn:

- **Regressão Logística, sem normalização → F1 0.44** (pior que o regex!). Causa: features em escalas muito diferentes (comprimento ~2000 vs `has_ip` ∈ {0,1}) — as grandes dominam.
- **+ StandardScaler →** mesmo F1, mas **treino 5× mais rápido** (normalizar ajuda a convergência, não este problema de recall).
- **+ `class_weight="balanced"` → F1 0.49.**
- **Random Forest → F1 0.66.** Não-linear: aprende limiares e interações entre features automaticamente — basicamente o regex, mas descoberto pelos dados.

O `feature_importances_` confirmou: `length` + `digits` responderam por 52% das decisões; `has_ip` e `@` foram quase inúteis — **a mesma conclusão a que o ablation do regex chegou de forma independente.** Dois métodos diferentes concordando = conclusão confiável.

### 3. TF-IDF — deixar a máquina ler o texto

N-grams de caractere (3–5) + Regressão Logística. Sem features manuais:

- **F1 0.89** (test set completo, 101k) — salto grande. O modelo aprendeu sozinho que sequências como `webscr`, `.exe`, `login`, `php`, `.edu` são sinal — **redescobrindo as keywords do regex sem ser instruído.**
- **Teste de honestidade:** suspeitei de "shortcut learning" — que tokens de ano (`2011`, `/09`) fossem artefatos temporais. **Testei** em vez de assumir: os anos 2009/2010/2011 são todos ~98,6% `good`, *consistentemente*. Isso refutou a hipótese de viés (sites de conteúdo datam URLs — sinal legítimo) e confirmou que os tokens fortes (`login` aparece em 99,5% dos phishings) são genuínos.

### 4. Embeddings & LLM — testando o hype

- **Embeddings de sentença** (`all-MiniLM-L6-v2`, 384 dim): **F1 0.79** numa amostra de 30k. Na *mesma* amostra, o TF-IDF tirou **0.86** — e foi **~80× mais rápido**. Embeddings capturam *significado*; uma URL não é linguagem natural, então o modelo semântico perde.
- **LLM local** (Llama 3.2 3B, em Docker, acelerado por GPU):
  - Primeiro **recusou** a tarefa (um guardrail de segurança confundiu "classificar phishing" com "conduzir phishing"). Resolvido com prompt engineering (papel defensivo explícito).
  - Zero-shot: **F1 0.65**, **275 ms/URL**.
  - **Few-shot piorou** — duas vezes. Tanto exemplos caricatos quanto realistas empurraram o modelo pequeno a marcar quase tudo como phishing (F1 ~0.40). Few-shot ajuda modelos *grandes*; pode prejudicar os pequenos.

---

## 📊 Resultados em dois contextos

A tabela de Resumo é a comparação **justa** (todos nas mesmas 500 URLs, limitado pelo LLM). Mas as abordagens baratas conseguem rodar o **test set completo de 101k** — estatisticamente mais robusto para elas. As duas visões, de propósito:

**A) Benchmark unificado — 500 URLs, amostra idêntica (justo, LLM incluído)**
→ ver [tabela de Resumo](#-resumo--a-tabela-que-diz-tudo).

**B) Testes individuais em escala — abordagens baratas no test set completo de 101k (robusto, LLM excluído — lento demais)**

| Abordagem | F1 (phishing) | Precision | Recall | Tamanho do teste |
|-----------|:-------------:|:---------:|:------:|:----------------:|
| TF-IDF + LogReg | 0.89 | 0.85 | 0.93 | 101.440 |
| Random Forest (8 feat) | 0.66 | 0.61 | 0.72 | 101.440 |
| Regex baseline | 0.54 | 0.79 | 0.41 | 101.440 |
| Embeddings + LogReg | 0.79 | 0.71 | 0.89 | 10k (custo de encoding) |
| LLM (zero-shot) | 0.65 | 0.60 | 0.71 | 500 (custo de velocidade) |

Os números são próximos entre os dois contextos — o TF-IDF vence nos dois. O contexto B mostra o teto real dos modelos baratos em escala; o contexto A é o único que consegue incluir o LLM de forma justa.

### Ressalvas honestas (a parte que dá credibilidade)
- O resultado do LLM é de um modelo **pequeno (3B), zero-shot, só-URL**. Um modelo de fronteira ou few-shot bem curado iria melhor — isto compara *abordagens sob restrições iguais*, não dá a palavra final sobre cada tecnologia.
- O dataset é **estático**; phishing real é adversarial e muda com o tempo. ~0.89 é realista para um dataset acadêmico, otimista frente à produção.

---

## 🏗️ Arquitetura & Deploy

```
                    URL (texto)
                        │
   ┌──────────┬─────────┼──────────┬──────────┐
   ▼          ▼         ▼          ▼          ▼
 Regex   RandomForest TF-IDF   Embeddings   LLM
   │          │         │          │          │
   └──────────┴─────────┴──────────┴──────────┘
                        │  benchmark.py (mesma amostra, mesma métrica)
                        ▼
              Vencedor: TF-IDF + LogReg
                        │
                        ▼
        joblib  →  FastAPI  →  Docker (multi-stage)
```

- **Serialização:** o `{vectorizer, model}` vencedor é salvo com `joblib`.
- **API:** FastAPI com validação automática (Pydantic), endpoint `/predict` retornando rótulo + confiança, health check e docs auto-geradas em `/docs`.
- **Docker:** **build multi-stage → imagem de 586 MB** (~75% menor que um build single-stage ingênuo com PyTorch embutido). Roda como **usuário não-root**, com `HEALTHCHECK`.
- **Compose:** um comando para subir o serviço.
- **Decisão de engenharia:** o **LLM foi deliberadamente deixado fora do deploy.** A API serve o *vencedor* (TF-IDF), não a opção mais impressionante. Coerência acima do hype.

---

## 📁 Estrutura do projeto

```
phishing-ml-comparison/
├── data/                         # dataset (gitignored — baixar do Kaggle)
│   └── .gitkeep
├── models/
│   └── phishing_model.joblib     # vencedor serializado (versionado p/ o Docker funcionar)
├── src/
│   ├── features.py               # extração de features estruturais
│   ├── prepare_data.py           # split estratificado treino/teste
│   ├── train_model.py            # treina + serializa o modelo vencedor
│   ├── benchmark.py              # comparação unificada (mesma amostra, mesma métrica)
│   ├── api.py                    # serviço FastAPI
│   └── classifiers/
│       ├── regex_baseline.py     # 1. regras heurísticas + estudo de ablation
│       ├── ml_classic.py         # 2. features estruturais + LogReg / RandomForest
│       ├── ml_tfidf.py           # 3. n-grams de caractere + LogReg (vencedor)
│       ├── ml_embeddings.py      # 4a. embeddings de sentença + LogReg
│       └── ml_llm.py             # 4b. LLM local (Ollama) zero/few-shot
├── Dockerfile                    # build multi-stage (586 MB)
├── docker-compose.yml
├── requirements.txt              # dev (completo, com torch)
├── requirements-api.txt          # prod (enxuto, só a API)
├── README.md                     # English
└── README.pt-br.md               # Português
```

## 🚀 Como rodar

```bash
git clone <repo-url> && cd phishing-ml-comparison

# Opção A — a API (vencedor), containerizada
docker compose up -d --build
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "secure-paypal.tk/webscr?cmd=login"}'
# → {"url":"...","label":"bad","confidence":1.0}

# Docs interativa:  http://localhost:8000/docs
```

```bash
# Opção B — reproduzir os experimentos
python -m venv .venv && source .venv/Scripts/activate   # (Windows Git Bash)
pip install -r requirements.txt
python -m src.prepare_data          # split estratificado treino/teste
python -m src.classifiers.regex_baseline
python -m src.classifiers.ml_tfidf
python -m src.benchmark             # a tabela comparativa unificada
```

---

## 🧠 O que aprendi

- **Accuracy mente em dados desbalanceados.** O F1 da classe minoritária é a métrica honesta (precision / recall / matriz de confusão acima de um número só).
- **Olhe os dados *antes* de criar uma feature.** Uma regra "é HTTPS?" marcou *tudo* como phishing — as URLs do dataset não têm prefixo de protocolo. Só foi pego ao rodar de verdade.
- **Evite vazamento de dados (data leakage):** ajuste o scaler/vetorizador **só no treino**, nunca no conjunto inteiro — senão informação do teste vaza pro treino e as métricas mentem.
- **Feature engineering > escolha do modelo.** Os mesmos dados foram de F1 0.49 (linear, features estruturais) a 0.89 (TF-IDF) — o salto veio de *não esconder o sinal mais forte* (o texto) do modelo.
- **Quando dois métodos independentes concordam, confie no resultado.** O ablation do regex e o `feature_importances_` do Random Forest chegaram ao *mesmo* ranking (`@` e IP são inúteis).
- **A ferramenta certa depende do problema, não do hype.** Uma técnica de 1972 venceu um LLM de 2026 aqui — e foi ~800× mais rápida.
- **Teste suas próprias hipóteses.** Minha preocupação com "vazamento temporal" e minha aposta de que "few-shot ajudaria" foram ambas refutadas pelos dados. Método acima de intuição.
- **Guardrails de IA podem bloquear trabalho defensivo legítimo.** O LLM recusou classificar phishing até o prompt deixar a intenção defensiva explícita — prompt engineering como habilidade de segurança.
- **Nota de segurança:** `joblib`/`pickle` executam código ao carregar — nunca dê `load()` num arquivo de modelo não confiável (vetor real de RCE). O `safetensors` existe para resolver isso em redes neurais.

---

## 🔮 Trabalho futuro

- **Split temporal** (treinar em URLs antigas, testar em novas) para medir generalização real contra um adversário.
- **LLM maior / few-shot curado** para um teto mais justo do LLM.
- **Mais sinais:** idade do domínio (WHOIS), certificado TLS, feeds de reputação — o que sistemas de produção realmente usam.

---

## ⚙️ Stack

`Python 3.13` · `pandas` · `scikit-learn` · `sentence-transformers` · `Ollama (Llama 3.2)` · `FastAPI` · `Docker`
