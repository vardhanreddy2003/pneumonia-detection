# Pneumonia Detection Model Deployment 🚑

A deep learning-based pneumonia detection system deployed using Flask.
🌐 Live Demo
This project contains a deep learning model trained to detect pneumonia from chest X-ray images. The model is deployed using a Flask web application, allowing users to upload an image and receive a prediction.

🚀 Features
Upload chest X-ray images via a web interface.

Get real-time predictions (Normal or Pneumonia).

Simple and lightweight Flask server for deployment.

🧠 Model Details
•	Designed and implemented a CNN model to accurately detect pneumonia from chest X-ray images.
•	Focused on optimizing recall to minimize false negatives, ensuring that no pneumonia cases are overlooked.
•	Built  a custom  architecture incorporating L2 regularization, EarlyStopping, and Dropout layers to enhance recall performance.

🔧 Setup Instructions
Clone the Repository

bash
Copy code
git clone https://github.com/your-username/pneumonia-detection-flask.git
cd pneumonia-detection-flask
Create a Virtual Environment (Optional but Recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Requirements

bash
Copy code
pip install -r requirements.txt
Run the Flask App

bash
Copy code
python app.py
Open in Browser

Visit http://127.0.0.1:5000/ to access the application.

📸 Screenshots
![Pneumonia Detection and 1 more page - Personal - Microsoft​ Edge 26-04-2025 15_49_24](https://github.com/user-attachments/assets/73786e88-30dc-40e4-bf91-717bb1ad250f)
![Pneumonia Detection and 1 more page - Personal - Microsoft​ Edge 26-04-2025 15_50_48](https://github.com/user-attachments/assets/5c645a8b-2aef-4fa7-b03d-c9ecbdf72fff)


📂 Requirements
Flask

TensorFlow / Keras

NumPy

Pillow



✨ Future Improvements
Can improve model performance

Improve UI/UX.


🧑‍💻 Author
Sai vardhan

