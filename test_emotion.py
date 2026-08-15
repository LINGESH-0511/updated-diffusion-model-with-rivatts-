from transformers import pipeline

# Load the emotion classification pipeline
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # returns scores for all emotion labels, not just the top one
)

def check_emotion(text):
    results = classifier(text)[0]  # list of {'label': ..., 'score': ...}
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    print(f"\nInput text: {text}")
    print("Emotion scores:")
    for r in results_sorted:
        print(f"  {r['label']:10s} -> {r['score']:.4f}")

    top_emotion = results_sorted[0]
    print(f"\n>>> Detected emotion: {top_emotion['label']} ({top_emotion['score']:.2%})")
    return top_emotion['label']

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'q' to quit): ")
        if text.lower() == 'q':
            break
        check_emotion(text)