# Diabetes-Prediction-Pima-Indians

## Description
This project uses the Pima Indians Diabetes Database to predict diabetes using K-Nearest Neighbors (KNN) and Neural Networks.

## Dataset
The dataset used in this project is from Kaggle: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data). The contained information regarding the patients’ features including pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. Most importantly, it contained whether these patients got diabetes. In terms of libraries and frameworks, we used NumPy, Pandas, PyTorch, Scikit-Learn, Seaborn, MatPlotLib, TensorFlow, Keras, and Optuna, common machine learning libraries to manipulate the data and implement our models.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Diabetes-Prediction-Pima-Indians.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Diabetes-Prediction-Pima-Indians
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the K-Nearest Neighbors model:  
    ```bash python knn.py ```  
To run the Neural Network model (basic and advanced NN models):  
    ```bash python nn_basic.py```  
    ```bash python nn_advanced.py```

## Models

### K-Nearest Neighbors (KNN)
The K-Nearest Neighbors (KNN) model is a simple, instance-based learning algorithm used for classification. The basic idea is to predict the class of a given sample based on the classes of its nearest neighbors in the feature space.

#### Implementation Details
- **Training Data**: The model uses the Pima Indians Diabetes Database for training.
- **Distance Metric**: Euclidean distance is used to measure the similarity between data points.
- **Number of Neighbors (k)**: The optimal value of k is determined through cross-validation.
- **Normalization**: Feature scaling is applied to ensure that all features contribute equally to the distance calculations.

#### Visualization
- **Bar Chart**: The script generates a bar chart to visualize the performance metrics (accuracy, precision, recall, and F1-score) for different values of k. This helps in selecting the optimal number of neighbors.
- **ROC Curves**: Receiver Operating Characteristic (ROC) curves are plotted to evaluate the model's performance. The ROC curve shows the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate) for various threshold settings. The Area Under the Curve (AUC) is also calculated to summarize the model's performance.

### Neural Networks (NN)
The Neural Network (NN) model is a more complex algorithm that uses layers of neurons to make predictions. This section describes both the basic and advanced implementations of the NN model.

#### Implementation Details
- **Basic Neural Network**: A simple neural network with one or two hidden layers.
- **Advanced Neural Network**: A more sophisticated model with multiple hidden layers, dropout for regularization, and other advanced techniques.

#### Data Preprocessing and Interpolation
- **Data Preprocessing**: The dataset is preprocessed by normalizing the features to ensure that all input features contribute equally to the model. Missing values, if any, are handled appropriately.
- **Interpolation**: Interpolation techniques are used to fill in any missing data points to ensure a complete dataset for training the model.

#### Hyperparameter Optimization with Optuna
- **Optuna Study**: Optuna is used for hyperparameter optimization. The script sets up an Optuna study to find the best hyperparameters for the neural network, such as learning rate, number of layers, and number of neurons per layer. This is done by defining an objective function that trains the model and evaluates its performance using cross-validation.
- **Optimization Process**: The Optuna study runs multiple trials, each with a different set of hyperparameters, and selects the best set based on the performance metrics.