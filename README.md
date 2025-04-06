
# Handwritten Digit Recognition using MNIST and Deep Learning 🧠🖊️

This project demonstrates a deep learning-based solution to recognize handwritten digits (0–9) using the MNIST dataset. A Convolutional Neural Network (CNN) model was built with Keras and TensorFlow to classify grayscale images of handwritten digits with high accuracy.

The model is trained, evaluated, and integrated with a simple GUI using Tkinter that allows users to draw digits and receive instant predictions.

## 🚀 Features

- MNIST dataset pre-processing and normalization
- CNN model with Conv2D, MaxPooling, and Dense layers
- Achieved high accuracy on test data
- Tkinter-based GUI for drawing and real-time prediction
- Model saved and loaded using `.h5` file format

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow (PIL)
- Tkinter

## 📂 Project Structure

- mnist_model_training.py # Model training and saving
- gui_predictor.py # GUI for live digit drawing and prediction
- mnist.h5 # Trained model weights
- README.md # Project description

  
## 🧠 How it Works

1. The CNN model is trained on 60,000 images from the MNIST dataset.
2. The trained model achieves >98% accuracy on the 10,000 test samples.
3. A GUI app built with Tkinter allows users to draw a digit.
4. The drawing is processed (grayscale + resized), then passed to the model.
5. The model predicts and displays the digit in real-time.

## 📈 Result

- Final Test Accuracy: ~98%
- Prediction Time: Instant (within milliseconds)

## 📚 Dataset

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/): 28x28 grayscale images of handwritten digits.

## 🙋‍♂️ Author

Ashish Deep Sen 
GitHub: [github.com/Ashishdeepsen](https://github.com/Ashishdeepsen)

---

Feel free to fork, contribute or leave feedback!


### 📘 Project Summary – Handwritten Digit Recognition using MNIST

In this project, I explored deep learning techniques for image classification using the MNIST dataset. I designed a Convolutional Neural Network (CNN) that can classify handwritten digits (0 to 9) with high accuracy. 

The model was trained using Keras and TensorFlow, and later integrated into a GUI built with Tkinter. The application enables users to draw a digit using a mouse, which is then predicted in real-time by the trained model. 

**Key Learning Outcomes:**

- Understanding CNN architecture and image feature extraction
- Data preprocessing and normalization
- Model evaluation and optimization
- Saving/loading models using `.h5` format
- GUI development in Python using Tkinter
- Using OpenCV and Pillow for image manipulation

