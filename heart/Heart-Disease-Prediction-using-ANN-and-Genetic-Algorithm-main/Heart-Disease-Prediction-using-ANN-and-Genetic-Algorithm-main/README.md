# Heart-Disease-Prediction-using-ANN-and-Genetic-Algorithm
This project aims to develop an accurate and reliable system for heart disease prediction using two popular techniques in machine learning: artificial neural network (ANN) and genetic algorithm. The code is written in Python and uses the Keras library for building and training the ANN model. The genetic algorithm is implemented using the DEAP library.

# Dataset
The dataset used in this project is the Cleveland Heart Disease dataset, which consists of 14 attributes and 303 instances. The dataset is preprocessed using techniques such as scaling, one-hot encoding, and feature selection to improve the accuracy of the model.

# ANN Architecture
The ANN architecture used in this project consists of 3 hidden layers with 32, 16, and 8 neurons, respectively. The model is trained using the Adam optimizer and binary cross-entropy loss function.

# Genetic Algorithm Optimization
The genetic algorithm is used to optimize the hyperparameters of the ANN, such as the learning rate, batch size, and number of epochs. The fitness of each individual is evaluated using the accuracy of the ANN model. The fittest individuals are selected for reproduction and crossover to generate a new population for the next generation. The process is repeated for a specified number of generations until the optimal set of hyperparameters is found.

# Usage
To use the project, clone the repository and install the required dependencies using pip. Then run the main.py file to launch the Flask application. Or open the res.ipynb file to review the code as well as the previous results. 

# Conclusion
This project demonstrates the potential of using ANN and genetic algorithm for heart disease prediction. The system achieves high accuracy and can be useful for healthcare professionals in diagnosing and preventing heart disease. Contributions and suggestions for improvement are welcome.






