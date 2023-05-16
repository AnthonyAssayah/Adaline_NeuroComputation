
# Classification of Characters
This project involves the classification of Hebrew characters using two different algorithms: Adaline and Feedforward Neural Network (NN).
The purpose of this project is to explore and compare the performance of these algorithms in the task of character classification.

## Adaline Algorithm
The Adaline algorithm is utilized for the classification task with a learning rate of 0.01 and 50 epochs. 
The dataset consists of three-letter Hebrew characters: "bet", "mem", and "lamed". The Adaline algorithm is trained on this dataset to learn the patterns 
and relationships between the input features and the corresponding labels.

The code implements the Adaline algorithm and prints the confusion matrix and classification report for each iteration. Additionally, it computes the average accuracy and standard deviation across all iterations, providing insights into the algorithm's performance.

## Feedforward Neural Network (NN)
The Feedforward NN is employed as an alternative algorithm for character classification. The architecture of the Feedforward NN consists of multiple layers, including input, hidden, and output layers. The input layer has a dimension corresponding to the input data, while the hidden layers are designed with 256, 128, and 64 neurons, respectively. The ReLU activation function is utilized in the hidden layers to introduce non-linearity and capture complex relationships in the data. The output layer consists of neurons equal to the number of classes to be classified, with a softmax activation function to produce class probabilities.

The Feedforward NN is trained using the Adam optimizer and the binary cross-entropy loss function. The model is compiled and evaluated using accuracy as the performance metric. The code implements cross-validation using StratifiedKFold with 5 folds to assess the performance of the Feedforward NN.

## Code Organization
The project code is organized into several functions:

 - ` plot_confusion_matrix(predictions, y_test) ` : This function generates a confusion matrix plot based on the predicted and actual labels.
 - ` create_feedforward_nn(input_dim, num_classes) `: This function constructs a Feedforward NN model with the specified architecture and settings.
 - ` classify(X, y, label_pair) `: This function performs the classification task using either Adaline or Feedforward NN. It filters the dataset based on the specified labels, prepares the input data and labels, normalizes the input data, and performs cross-validation. It trains and evaluates the model, prints the accuracy for each iteration, and generates confusion matrices and classification reports. Finally, it calculates the average accuracy and standard deviation.

## Getting Started
To run the code, ensure that the required libraries are installed, including Pandas, NumPy, Scikit-learn, Seaborn, and Matplotlib. Additionally, TensorFlow and Keras should be installed for the Feedforward NN implementation.

You can customize the parameters in the code to fit your specific requirements, such as the learning rate, number of epochs, number of hidden layers, and activation functions. The code is designed to handle the classification of three-letter Hebrew characters, but you can modify it to suit different datasets and classification tasks.

To execute the code, simply run the main script or call the classify function with the appropriate inputs.

## Conclusion
This project showcases the application of Adaline and Feedforward NN algorithms for the classification of Hebrew characters. By comparing the results and performance metrics of both algorithms, insights can be gained into their effectiveness in handling character classification tasks. The provided code, functions, and visualizations serve as a foundation for further exploration and analysis of these algorithms in different classification scenarios.
