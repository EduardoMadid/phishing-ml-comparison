import re

def extract_features(url:str) -> list:
    features = []
    # 1 . Comprimento total
    features.append(len(url))
    # 2. Numero de pontos
    features.append(url.count('.'))
    # 3. Numero de hifens
    features.append(url.count('-'))
    # 4. Numero de digitos
    features.append(sum(char.isdigit() for char in url))
    # 5. Numero de barras
    features.append(url.count('/'))
    # 6. Numero de @
    features.append(url.count('@'))
    # 7. Número de parâmetros de query (? e &)
    features.append(url.count("?") + url.count("&"))
    # 8. Tem endereço IP? (1 = sim, 0 = não)
    features.append(1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0)
    return features