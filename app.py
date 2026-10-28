from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "RandomForestModel.joblib"

app = Flask(__name__)

def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Download RandomForestModel.joblib from your Experiment 11 Colab notebook "
            "and place it in the same folder as app.py."
        )

    artifact = joblib.load(MODEL_PATH)

    # Supports either the plain pipeline created by the notebook, or a
    # dictionary bundle if you later choose to save extra metadata.
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        feature_names = artifact.get("feature_names")
        label_encoder = artifact.get("label_encoder")
        class_labels = artifact.get("class_labels")
    else:
        model = artifact
        feature_names = getattr(model, "feature_names_in_", None)
        label_encoder = None
        class_labels = None

    if feature_names is None:
        raise ValueError(
            "Could not read feature names from the saved model. "
            "Please save the model after fitting it with a pandas DataFrame."
        )

    return model, list(feature_names), label_encoder, class_labels

try:
    model, feature_names, label_encoder, class_labels = load_artifact()
    load_error = None
except Exception as exc:
    model, feature_names, label_encoder, class_labels = None, [], None, None
    load_error = str(exc)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = load_error
    submitted_values = {}

    if request.method == "POST" and model is not None:
        try:
            row = {}
            for feature in feature_names:
                raw_value = request.form.get(feature, "").strip()
                submitted_values[feature] = raw_value
                if raw_value == "":
                    raise ValueError(f"Please enter a value for '{feature}'.")
                row[feature] = float(raw_value)

            input_df = pd.DataFrame([row], columns=feature_names)
            predicted_value = model.predict(input_df)[0]

            if label_encoder is not None:
                try:
                    prediction = str(label_encoder.inverse_transform([int(predicted_value)])[0])
                except Exception:
                    prediction = str(predicted_value)
            elif class_labels is not None:
                try:
                    prediction = str(class_labels[int(predicted_value)])
                except Exception:
                    prediction = str(predicted_value)
            else:
                prediction = str(predicted_value)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction,
        error=error,
        submitted_values=submitted_values,
        model_ready=(model is not None),
    )

if __name__ == "__main__":
    app.run(debug=True)
