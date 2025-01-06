# Pneumonia Detection using Deep Learning

## Description
This project involves the development and implementation of a deep learning model to detect pneumonia from x-ray images. The primary goal is to classify images into categories such as "positive" (pneumonia detected) and "negative" (healthy).

## Key Features

### Data Preparation:
- The dataset is organized into training and testing directories. The training set is used to teach the model to differentiate between positive and negative images, while the testing set is used to evaluate the model's performance.
- A CSV file is generated for both the training and testing datasets, containing image paths and corresponding labels.

### Model Architecture:
- The model is built using TensorFlow and Keras, incorporating layers such as Dense, Flatten, Dropout, and L2 regularizations.

### Training:
- The model is trained with various callbacks for monitoring the training process using Weights and Biases.
- Adam optimizer is utilized to optimize the model during training.

### Visualization:
- Visualization tools like Seaborn and Matplotlib are used for plotting and analyzing the training process and results.
- Visualizations of model architecture and performance metrics are generated.

## Tools and Libraries:
- **TensorFlow** and **Keras** for deep learning model development.
- **OpenCV** for image processing.
- **Pandas** for data handling.
- **Seaborn** and **Matplotlib** for data visualization.

## Note:
In pneumonia detection, I primarily focus on **recall** as it plays a crucial role in identifying true positive cases. Ensuring a high recall rate helps minimize the risk of missing any pneumonia cases, which is critical for timely diagnosis and treatment. By optimizing recall, we can enhance the ability of the detection system to recognize as many affected patients as possible, especially in medical applications where early intervention can significantly impact patient outcomes.
