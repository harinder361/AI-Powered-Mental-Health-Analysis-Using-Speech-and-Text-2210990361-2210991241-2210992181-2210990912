from datasets import load_dataset
from transformers import pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import numpy as np

print("Loading DAIR-AI dataset...")
train_data = load_dataset("dair-ai/emotion", "split", split="train")
test_data  = load_dataset("dair-ai/emotion", "split", split="test")

dair_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_map   = {"sadness":"sadness","joy":"joy","love":"joy",
               "anger":"anger","fear":"fear","surprise":"surprise"}

# Prepare train and test sets
X_train = [r["text"] for r in train_data]
y_train = [label_map[dair_labels[r["label"]]] for r in train_data]
X_test  = [r["text"] for r in test_data]
y_test  = [label_map[dair_labels[r["label"]]] for r in test_data]

def metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {pre*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1:.4f}")
    return acc, pre, rec, f1

results = {}

# ── BASELINE 1: Naive Bayes (TF-IDF) ──────────────────────────
print("\nRunning Naive Bayes...")
nb_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf',   MultinomialNB())
])
nb_pipe.fit(X_train, y_train)
nb_pred = nb_pipe.predict(X_test)
results['Naive Bayes (TF-IDF)'] = metrics("Naive Bayes + TF-IDF", y_test, nb_pred)

# ── BASELINE 2: SVM (TF-IDF) ──────────────────────────────────
print("\nRunning SVM...")
svm_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf',   SVC(kernel='linear', probability=True))
])
svm_pipe.fit(X_train, y_train)
svm_pred = svm_pipe.predict(X_test)
results['SVM (TF-IDF)'] = metrics("SVM + TF-IDF", y_test, svm_pred)

# ── BASELINE 3: BERT (base) ────────────────────────────────────
print("\nRunning BERT base (zero-shot, not fine-tuned)...")
bert_pipe = pipeline("text-classification",
                     model="bhadresh-savani/bert-base-uncased-emotion",
                     return_all_scores=False)

bert_label_map = {
    "sadness":"sadness","joy":"joy","love":"joy",
    "anger":"anger","fear":"fear","surprise":"surprise"
}

bert_pred = []
for i, text in enumerate(X_test):
    try:
        res = bert_pipe(text[:512])[0]
        label = res['label'].lower()
        bert_pred.append(bert_label_map.get(label, 'neutral'))
    except:
        bert_pred.append('neutral')
    if (i+1) % 200 == 0:
        print(f"  BERT: {i+1}/2000")

results['BERT-base Emotion'] = metrics("BERT-base Emotion (Bhadresh-Savani)", y_test, bert_pred)

# ── OUR MODEL: DistilRoBERTa ───────────────────────────────────
print("\nRunning our DistilRoBERTa model...")
our_pipe = pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=False)

our_pred = []
for i, text in enumerate(X_test):
    try:
        res = our_pipe(text[:512])[0]
        our_pred.append(res['label'].lower())
    except:
        our_pred.append('neutral')
    if (i+1) % 200 == 0:
        print(f"  Ours: {i+1}/2000")

results['Ours: DistilRoBERTa'] = metrics("OUR MODEL: DistilRoBERTa", y_test, our_pred)

# ── FINAL COMPARISON TABLE ─────────────────────────────────────
print("\n\n" + "="*70)
print("  FINAL COMPARISON TABLE")
print("="*70)
print(f"  {'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-"*70)
for model, (acc, pre, rec, f1) in results.items():
    marker = " <-- OURS" if "DistilRoBERTa" in model else ""
    print(f"  {model:<35} {acc*100:>6.2f}% {pre*100:>6.2f}% {rec*100:>6.2f}% {f1:>7.4f}{marker}")
print("="*70)