# IMAGE-CLASSIFICATION-MODEL-python-program

COMPANY : CODTECH IT SOLUTIONS

NAME : Thangaraj P

INTERN ID : CT08DY1102

DOMAIN : MACHINE LEARNING

DURATION : 8 WEEKS

MENTOR : NEELA SANTHOSH

TASK DESCRIOTON : 

The main objective of this task was to build an Image Classification Model using a Convolutional Neural Network (CNN) in Python.
The goal was to train a neural network that can automatically recognize and classify images into different categories.
This project demonstrates how deep learning techniques can process and learn patterns from image data.
CNNs are a powerful class of neural networks that have revolutionized image processing by automatically detecting edges, shapes, textures, and complex patterns without manual feature extraction.

Tools and Technologies Used
•	Programming Language: Python
•	Libraries Used:
o	TensorFlow – to build and train the CNN model
o	Keras (part of TensorFlow) – for high-level neural network API
o	NumPy – for numerical computations
o	Matplotlib – for visualizing training accuracy and loss
o	Seaborn – for displaying confusion matrix
o	scikit-learn – for splitting datasets and evaluating model performance
•	Dataset Used: The CIFAR-10 dataset, which contains 60,000 images of 10 categories such as airplanes, cars, birds, cats, and ships. Each image is 32×32 pixels with 3 color channels (RGB).
•	Editor/Platform: Visual Studio Code (VS Code) with Python extension or Jupyter Notebook.

Implementation Process
1.	Importing Required Libraries
The first step was importing all the essential libraries such as TensorFlow, Keras, NumPy, Matplotlib, and scikit-learn.
2.	Loading the Dataset
I used the CIFAR-10 dataset available directly through Keras (keras.datasets.cifar10). It automatically splits data into training (50,000 images) and testing (10,000 images) sets.
3.	Data Preprocessing
Images were normalized by dividing pixel values by 255.0 so that all values fall between 0 and 1. This normalization helps the neural network train faster and more efficiently.
The labels (numbers from 0–9) were converted into categorical format using to_categorical() for multi-class classification.
4.	Building the CNN Architecture
The CNN model was built using Keras’ Sequential API. The architecture consisted of:
o	Convolutional Layers – to extract features using 3×3 filters
o	Activation Function (ReLU) – to introduce non-linearity
o	MaxPooling Layers – to reduce spatial dimensions and computation
o	Dropout Layers – to prevent overfitting
o	Flatten Layer – to convert 2D features into a 1D vector
o	Dense (Fully Connected) Layers – for learning complex relationships
o	Output Layer (Softmax) – to predict the probability of each image class
5.	Compiling the Model
The model was compiled with:
o	Optimizer: Adam (for efficient gradient descent)
o	Loss Function: Categorical Crossentropy (for multi-class problems)
o	Metrics: Accuracy
6.	Training the Model
The model was trained using the fit() method for 10–20 epochs with a batch size of 64.
During training, I observed both training accuracy and validation accuracy to ensure the model wasn’t overfitting.
7.	Evaluating the Model
After training, the model was evaluated on the test dataset using the evaluate() function. The model achieved around 75–85% accuracy, depending on the number of epochs and tuning.
8.	Visualization
o	I plotted training accuracy vs. epochs and loss vs. epochs using Matplotlib.
o	I also visualized a few test images along with their predicted and actual labels.
o	Finally, I used Seaborn to draw a confusion matrix to see how well each class was predicted.
9.	Saving and Testing the Model
The trained model was saved using model.save("cnn_image_classifier.h5").
Later, it was reloaded and tested on unseen images to verify its performance.

Results and Analysis
The CNN model performed very well on the CIFAR-10 dataset.
It successfully learned to recognize patterns such as shapes, edges, and colors. The model achieved around 80% accuracy, which is a strong result for a basic CNN without data augmentation or transfer learning.
The training graphs showed a gradual increase in accuracy and a decrease in loss, indicating proper learning. The confusion matrix helped identify which classes (like cats and dogs) were harder to distinguish — a common challenge in image classification.
Overall, the model demonstrated the effectiveness of CNNs for real-world image recognition tasks.

Applications
CNN-based image classification is widely used in many industries and domains, including:
•	Facial Recognition Systems – used in phones, security, and attendance tracking
•	Self-Driving Cars – for detecting lanes, pedestrians, and traffic signs
•	Medical Imaging – detecting diseases in X-rays, CT scans, and MRIs
•	E-commerce – automatic product categorization from uploaded photos
•	Agriculture – identifying crop types, plant diseases, and soil conditions
•	Security Systems – object detection and surveillance

Conclusion
Through this task, I learned how to build, train, evaluate, and visualize a CNN model using TensorFlow and Keras.
The task helped me understand the importance of convolution, pooling, and activation functions in processing image data.
I used Visual Studio Code as my primary editor, which made running, debugging, and visualizing model performance easy and efficient.
This project strengthened my understanding of Deep Learning and Computer Vision, providing practical knowledge applicable to advanced AI projects like object detection, facial recognition, and autonomous systems.


