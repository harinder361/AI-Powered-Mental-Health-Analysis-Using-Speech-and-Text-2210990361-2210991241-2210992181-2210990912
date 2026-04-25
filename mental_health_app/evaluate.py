from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

print("Loading emotion model...")
model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)
print("Model ready!\n")

# DAIR-AI Emotion dataset — labels match your model closely
print("Downloading DAIR-AI Emotion dataset...")
dataset = load_dataset("dair-ai/emotion", "split", split="test")
print(f"Dataset loaded — {len(dataset)} test samples\n")

# DAIR-AI uses these 6 labels (as integers)
dair_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Map to your model's 7 emotions
label_map = {
    "sadness":  "sadness",
    "joy":      "joy",
    "love":     "joy",       # love is closest to joy
    "anger":    "anger",
    "fear":     "fear",
    "surprise": "surprise",
}

# ── Prepare samples ────────────────────────────────────────────
MAX_SAMPLES = 2000
texts       = []
true_labels = []

print(f"Preparing {MAX_SAMPLES} test samples...")
for row in dataset:
    if len(texts) >= MAX_SAMPLES:
        break
    dair_emotion = dair_labels[row["label"]]
    mapped = label_map.get(dair_emotion)
    if mapped is None:
        continue
    texts.append(row["text"])
    true_labels.append(mapped)

print(f"Ready — {len(texts)} samples prepared\n")

# ── Run predictions ────────────────────────────────────────────
print("Running predictions...")
predicted_labels = []
for i, text in enumerate(texts):
    result = model(text)[0]
    predicted_labels.append(result["label"].lower())
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(texts)}...")

print("\nDone! Calculating metrics...\n")

# ── Metrics ────────────────────────────────────────────────────
accuracy  = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
recall    = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
f1        = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

print("=" * 50)
print("         EVALUATION RESULTS")
print("=" * 50)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1:.4f}")
print("=" * 50)

print("\nPer-Emotion Breakdown:")
print(classification_report(true_labels, predicted_labels, zero_division=0))

# ── Save ───────────────────────────────────────────────────────
results_df = pd.DataFrame({
    "text":      texts,
    "true":      true_labels,
    "predicted": predicted_labels,
    "correct":   [t == p for t, p in zip(true_labels, predicted_labels)]
})
results_df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")
