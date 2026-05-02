from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import random
import os

app = Flask(__name__)

# Enable CORS (important if frontend/backend mismatch happens)
from flask_cors import CORS
CORS(app)

# Lazy load model
emotion_model = None

def load_model():
    global emotion_model
    if emotion_model is None:
        print("Loading emotion model...")
        emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=False
        )
        print("Model loaded successfully!")
    return emotion_model


# Suggestions
suggestions = {
    "joy": [
        "Great to see you feeling positive! Keep journaling what made you happy today.",
        "This is wonderful! Share your joy with someone close to you.",
        "Celebrate this moment! What made you happiest today?",
        "Ride this wave of happiness—it's contagious! Spread it around."
    ],
    "sadness": [
        "It's okay to feel sad. Try a short walk outside or call someone you trust.",
        "Consider talking to someone about what's troubling you.",
        "Sadness is temporary. Give yourself permission to feel it.",
        "A warm cup of tea, some journaling, or a favorite song might help right now."
    ],
    "anger": [
        "Take 5 slow deep breaths. A short walk or exercise can help release tension.",
        "Channel this energy into something productive—physical activity works wonders.",
        "Step back for a moment. What's really bothering you beneath the surface?",
        "Try cold water on your face or intense exercise to calm the nervous system."
    ],
    "fear": [
        "Try the 5-4-3-2-1 grounding technique: name 5 things you can see right now.",
        "Remember: fear is just your mind trying to protect you. You're safe.",
        "Talk through your fears with someone you trust or write them down.",
        "Take deep breaths and remind yourself of times you've overcome challenges before."
    ],
    "neutral": [
        "You seem balanced. A short mindfulness session can help maintain that calm.",
        "This equilibrium is valuable—keep nurturing what brings you peace.",
        "Consider doing something you enjoy to enhance this positive state.",
        "Reflect on what's helping you stay grounded right now."
    ],
}

@app.route("/")
def home():
    return render_template("index.html")


# Optional health check (helps Render detect app is alive)
@app.route("/health")
def health():
    return "OK", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Please enter some text"}), 400

        model = load_model()
        result = model(text)[0]

        emotion = result["label"].lower()

        if emotion not in suggestions:
            emotion = "neutral"

        score = round(result["score"] * 100, 1)

        # Map unwanted emotions
        emotion_mapping = {
            "disgust": "anger",
            "surprise": "neutral"
        }
        emotion = emotion_mapping.get(emotion, emotion)

        suggestion = random.choice(
            suggestions.get(emotion, ["Take care of yourself today."])
        )

        return jsonify({
            "emotion": emotion,
            "confidence": score,
            "suggestion": suggestion
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong"}), 500


# Only for local run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)