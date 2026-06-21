import matplotlib.pyplot as plt
from matplotlib import font_manager

# Resultados do benchmark unificado (mesma amostra de 500 URLs)
RESULTS = [
    ("Regex",           0.566,   0.01),
    ("Random Forest",   0.670,  17.34),
    ("LLM (Llama 3.2)", 0.647, 275.50),
    ("TF-IDF + LogReg", 0.884,  0.34),
]

# Paleta
VERDE   = "#27AE60"   # vencedor
AZUL    = "#5B7FA6"   # neutro
VERMELHO = "#E07A5F"  # destaque negativo (LLM lento)
CINZA   = "#6B7280"

RESULTS.sort(key=lambda x: x[1])
nomes = [r[0] for r in RESULTS]
f1s = [r[1] for r in RESULTS]
latencias = [r[2] for r in RESULTS]

melhor_f1 = max(f1s)
pior_lat = max(latencias)
cores_f1 = [VERDE if f1 == melhor_f1 else AZUL for f1 in f1s]
cores_lat = [VERMELHO if lt == pior_lat else (VERDE if lt == min(latencias) else AZUL) for lt in latencias]

plt.rcParams["font.family"] = "DejaVu Sans"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("white")

fig.suptitle("Phishing Detection — 4 Approaches, Same 500-URL Test",
             fontsize=16, fontweight="bold", color="#1F2937", y=0.98)

def estilo(ax):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

# --- F1 ---
bars1 = ax1.barh(nomes, f1s, color=cores_f1, zorder=3, height=0.65)
ax1.set_title("Accuracy  ·  F1 score (higher is better)", fontsize=12, color="#374151", pad=12)
ax1.set_xlim(0, 1)
for i, v in enumerate(f1s):
    ax1.text(v + 0.015, i, f"{v:.2f}", va="center", fontweight="bold", color="#1F2937")
estilo(ax1)

# --- Latência (log) ---
bars2 = ax2.barh(nomes, latencias, color=cores_lat, zorder=3, height=0.65)
ax2.set_title("Speed  ·  ms per URL, log scale (lower is better)", fontsize=12, color="#374151", pad=12)
ax2.set_xscale("log")
ax2.set_xlim(0.005, 1000)
for i, v in enumerate(latencias):
    ax2.text(v * 1.35, i, f"{v:.2f} ms", va="center", fontweight="bold", color="#1F2937")
estilo(ax2)

# Rodapé
fig.text(0.99, 0.01, "Eduardo Madid · phishing-ml-comparison",
         ha="right", fontsize=9, color=CINZA, style="italic")

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig("benchmark_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Salvo: benchmark_comparison.png")
plt.show()