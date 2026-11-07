import os
import re
import html
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

os.makedirs('results', exist_ok=True)

# 1. Levanto la data
train_df = pd.read_csv('data/ag_news_train.csv')
test_df = pd.read_csv('data/ag_news_test.csv')

# 2. Limpieza básica y lematización
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def limpiar_texto(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"\b(quot|reuters|ap|afp)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    
    doc = nlp(text)
    lemas = [t.lemma_ for t in doc if t.is_alpha and len(t) > 1]
    return " ".join(lemas)

def procesar_or_load(df, cache_path):
    if os.path.exists(cache_path):
        csv_guardado = pd.read_csv(cache_path)
        if len(csv_guardado) == len(df):
            print(f"Ya estaba procesado: {cache_path}")
            return csv_guardado
            
    print(f"Procesando {cache_path}...")
    df = df.copy()
    df['text_limpio'] = df['text'].apply(limpiar_texto)
    df.to_csv(cache_path, index=False)
    return df

train_df = procesar_or_load(train_df, 'results/train_preprocesado.csv')
test_df = procesar_or_load(test_df, 'results/test_preprocesado.csv')

X_train_txt = train_df['text_limpio']
y_train = train_df['label']
X_test_txt = test_df['text_limpio']
y_test = test_df['label']

# 3. Grid manual usando una parte del train (sin tocar test)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_txt, y_train, test_size=0.2, random_state=42, stratify=y_train
)

max_features_list = [1000, 5000, 20000]
ngram_list = [(1, 1), (1, 2)]

modelos = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVC': LinearSVC()
}

res = []
for max_f in max_features_list:
    for ngram in ngram_list:
        # Fit SOLO al sub-train para no filtrar vocabulario
        vec = TfidfVectorizer(max_features=max_f, ngram_range=ngram, stop_words='english')
        X_tr_vec = vec.fit_transform(X_tr)
        X_val_vec = vec.transform(X_val)

        for nombre, mod in modelos.items():
            mod.fit(X_tr_vec, y_tr)
            preds = mod.predict(X_val_vec)
            res.append({
                'modelo': nombre,
                'max_features': max_f,
                'ngram_range': str(ngram),
                'accuracy': round(accuracy_score(y_val, preds), 4),
                'f1_macro': round(f1_score(y_val, preds, average='macro'), 4)
            })

df_res = pd.DataFrame(res).sort_values('f1_macro', ascending=False).reset_index(drop=True)
df_res.to_csv('results/experimentos.csv', index=False)
print(df_res.to_string())

mejor_config = df_res.iloc[0]
print(f"Mejor combo en val: {mejor_config['modelo']} | max_f: {mejor_config['max_features']} | ngrams: {mejor_config['ngram_range']}")

# 4. Entreno final con todo el train y evalúo en test
ngram_eval = eval(mejor_config['ngram_range'])
vec_final = TfidfVectorizer(
    max_features=int(mejor_config['max_features']), 
    ngram_range=ngram_eval, 
    stop_words='english'
)

# Fit solo en train_txt completo, transform en test
X_train_vec = vec_final.fit_transform(X_train_txt)
X_test_vec = vec_final.transform(X_test_txt)

m_final = modelos[mejor_config['modelo']]
m_final.fit(X_train_vec, y_train)
preds_test = m_final.predict(X_test_vec)

# 5. Métricas y matriz
rep = classification_report(y_test, preds_test, digits=3)
print("\n--- TEST SET METRICS ---")
print(rep)

with open('results/classification_report.txt', 'w') as f:
    f.write(rep)

clases = sorted(y_train.unique())
cm = confusion_matrix(y_test, preds_test, labels=clases)

plt.style.use('dark_background')
plt.figure(figsize=(6, 5))
cmap = LinearSegmentedColormap.from_list('custom_blue', ['#111111', '#1f77b4'])
plt.imshow(cm, cmap=cmap)

for i in range(len(clases)):
    for j in range(len(clases)):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white', fontweight='bold')

plt.xticks(range(len(clases)), clases)
plt.yticks(range(len(clases)), clases)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title(f"Matriz de confusión ({mejor_config['modelo']})")
plt.tight_layout()
plt.savefig('results/matriz_confusion.png', dpi=150)

print("Listo, guardado en /results")
