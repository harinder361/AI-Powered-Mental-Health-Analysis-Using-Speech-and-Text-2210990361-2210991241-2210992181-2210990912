from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the emotion detection model (downloads once, then cached)
print("Loading emotion model... please wait")
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)
print("Model ready!")

# Suggestions based on detected emotion
suggestions = {
    "joy":      "Great to see you feeling positive! Keep journaling what made you happy today.",
    "sadness":  "It's okay to feel sad. Try a short walk outside or call someone you trust.",
    "anger":    "Take 5 slow deep breaths. A short walk or exercise can help release tension.",
    "fear":     "Try the 5-4-3-2-1 grounding technique: name 5 things you can see right now.",
    "disgust":  "Step away from what's bothering you. A short break and fresh air can help.",
    "surprise": "Take a moment to process. Write down your thoughts to make sense of things.",
    "neutral":  "You seem balanced. A short mindfulness session can help maintain that calm.",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text"}), 400

    result = emotion_model(text)[0]
    emotion = result["label"].lower()
    score   = round(result["score"] * 100, 1)
    suggestion = suggestions.get(emotion, "Take care of yourself today.")

    return jsonify({
        "emotion":    emotion,
        "confidence": score,
        "suggestion": suggestion
    })

if __name__ == "__main__":
    app.run(debug=True)