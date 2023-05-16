import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Function print confusion matrix
def plot_confusion_matrix(predictions, y_test):
    cm = confusion_matrix(predictions, y_test)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("Confusion matrix")
    plt.xlabel("Actual label")
    plt.ylabel("Predicted label")

def create_feedforward_nn(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def classify(X, y, label_pair):
    label1, label2 = label_pair

    # Filter the dataframe to include only the specified labels
    dataframe = pd.DataFrame({'vector': list(X), 'label': list(y)})
    dataframe = dataframe[dataframe['label'].isin(label_pair)]

    # Prepare the input data (X) and labels (y)
    X = np.array([np.array(x) for x in dataframe['vector']])
    y = np.array(dataframe['label'])

    # Convert labels to 0 and 1
    y = np.where(y == label1, 0, 1)

    # Normalize the input data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Cross-validate the feedforward neural network classifier
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracies = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_cat, y_test_cat = to_categorical(y_train), to_categorical(y_test)
        model = create_feedforward_nn(input_dim=X.shape[1], num_classes=2)
        model.fit(X_train, y_train_cat, epochs=100, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        accuracies.append(accuracy)
        print(f"Iteration {i+1}: Test accuracy: {accuracy * 100:.2f}%")

        # Print confusion matrix and classification report
        predictions = np.argmax(model.predict(X_test), axis=-1)
        # plot_confusion_matrix(predictions, y_test)
        cr = classification_report(y_test, predictions, target_names=['Negative', 'Positive'])
        # print(f"\nClassification report for iteration {i+1}:\n{cr}\n")

    # Calculate the average accuracy and standard deviation
    avg_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    print(f"Average accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Standard deviation: {std_deviation * 100:.2f}%\n")
    return avg_accuracy, std_deviation

if __name__ == '__main__':
    # Load the saved dataframe
    dataframe = pd.read_pickle('combined_dataframe.pkl')

    # Classify 'ב' vs 'מ'
    print("************* Classify 'ב' vs 'מ' *************")
    label_pair = [1, 3]
    classify(dataframe['vector'], dataframe['label'], label_pair)

    # Classify 'ב' vs 'ל'
    print("************* Classify 'ב' vs 'ל' *************")
    label_pair = [1, 2]
    classify(dataframe['vector'], dataframe['label'], label_pair)

    # Classify 'מ' vs 'ל'
    print("************* Classify 'ל' vs 'מ' *************")
    label_pair = [2, 3]
    classify(dataframe['vector'], dataframe['label'], label_pair)