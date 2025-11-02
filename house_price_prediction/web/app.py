from flask import Flask, render_template, request, jsonify
from FileUtil import FileUtil
import os

app = Flask(__name__)

# Load the trained model at startup
model_path = "housingmodel.zip"
if os.path.exists(model_path):
    trainedModel = FileUtil.loadmodel(model_path)
    print(f"Model loaded successfully from {model_path}")
else:
    trainedModel = None
    print(f"WARNING: Model file '{model_path}' not found!")
    print("Please train and save a model first using the Tkinter or PyQt application.")


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/doprediction", methods=["POST"])
def doPrediction():
    # Check if model is loaded
    if trainedModel is None:
        return jsonify({
            "error": "Model not loaded",
            "message": "Please train and save a model first using the desktop application."
        }), 500

    try:
        # Get JSON data from request
        data = request.get_json()

        area_income_value = float(data["area_income"])
        area_house_age_value = float(data["area_house_age"])
        area_number_of_rooms_value = float(data["area_number_of_rooms"])
        area_number_of_bedrooms_value = float(data["area_number_of_bedrooms"])
        area_population_value = float(data["area_population"])

        result = trainedModel.predict([[area_income_value,
                                        area_house_age_value,
                                        area_number_of_rooms_value,
                                        area_number_of_bedrooms_value,
                                        area_population_value]])

        return jsonify({"prediction": float(result[0])})

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
