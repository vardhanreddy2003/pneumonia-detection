# import os
# import numpy as np
# from flask import Flask, render_template, request, jsonify
# from PIL import Image
# from tensorflow.keras.models import load_model

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

# model = load_model("pneumonia_detection_model.h5",compile=False)

# app = Flask(__name__)


# @app.route('/')
# def index():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def detect_result():
#     pneumonia_image = request.files.get("image")

#     if not pneumonia_image:
#         return jsonify({"error": "No image uploaded"}), 400

#     processed_image = process_uploaded_image(pneumonia_image)
#     prediction = model.predict(processed_image,verbose=0)
    
#     result=" "
    
#     if(prediction[0]>0.5):
#         result="positive"
#     else:
#         result="negative"
#     return result        


# def process_uploaded_image(file_obj):
#     img = Image.open(file_obj).convert('RGB')
#     img = img.resize((256, 256))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)
#     return img_array


# # if __name__ == "__main__":
# #     app.run(host='0.0.0.0',port=8080)
# if __name__ == "__main__":
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from tensorflow.keras.preprocessing import image
from transformers import AutoModelForImageClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

# Disable optimizations to avoid issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

# Download model from Hugging Face if needed, or load it directly
MODEL_NAME = "vardhan2003/pneumonia-detection-model"

# Load the model (This assumes your model is a TensorFlow model)
model_path = hf_hub_download(repo_id=MODEL_NAME, filename="pneumonia_detection_model.h5")
model = load_model(model_path, compile=False)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def detect_result():
    pneumonia_image = request.files.get("image")

    if not pneumonia_image:
        return jsonify({"error": "No image uploaded"}), 400

    processed_image = process_uploaded_image(pneumonia_image)
    prediction = model.predict(processed_image, verbose=0)
    
    result = "positive" if prediction[0] > 0.5 else "negative"
    
    return jsonify({"result": result, "prediction": prediction[0]})


def process_uploaded_image(file_obj):
    img = Image.open(file_obj).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)
    return img_array


# Use Waitress for production
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
