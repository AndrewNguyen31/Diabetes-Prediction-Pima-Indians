# python program that creates a neural network to predict whether a person will have diabetes or not
# the neural network is trained on the Pima Indians Diabetes dataset
# the dataset includes females at least 21 years old of Pima Indian heritage
# the dataset includes 768 observations and 8 attributes
# 614 (80%) of the observations will be used for training the neural network
# increase the number of records used for training the neural network to 80% of the observations
# 154 (20%) of the observations will be used for testing the neural network
# then we will use Optuna hyperparameter optimization to find the best hyperparameters for the neural network
# through this, we will find the best model that can predict whether a person will have diabetes or not

# import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import optuna

# load in the dataset
dataset = pd.read_csv('diabetes.csv')   

# replace missing/0 values with the median of the column
for column in dataset.columns:
    if column in ['Pregnancies', 'Outcome']: 
        continue
    
    # replace 0's with NA's and then fill the NA's with the median of the columns
    dataset[column].replace(0, pd.NA, inplace=True)
    column_median = dataset[column].median()
    dataset[column].fillna(column_median, inplace=True)
    
# turn dataset into a csv file and save it to the directory
dataset.to_csv('diabetes_cleaned.csv', index=False)

# create the feature and target datasets
x = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# create the training and testing datasets based on most impactful features
impactful_features = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = StandardScaler().fit_transform(x[impactful_features])
y = y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# scale the features using MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# define the data augmentation functions
def add_noise(x, mean = 0, std = 0.1):
    noise = np.random.normal(mean, std, size = x.shape)
    x_noisy = x + noise
    return np.clip(x_noisy, 0, 1)

def interpolate(x, y, alpha = 0.5):
    indices = np.random.permutation(len(x))
    x_permuted = x[indices]
    y_permuted = y[indices]
    x_interpolated = alpha * x + (1 - alpha) * x_permuted
    y_interpolated = alpha * y + (1 - alpha) * y_permuted
    return x_interpolated, y_interpolated.astype(int)

# make copies of the scaled training data
x_train_augmented = x_train_scaled.copy()
y_train_augmented = y_train.copy()

# add noise to the training data 
x_train_noisy = add_noise(x_train_scaled, std = 0.1)
x_train_augmented = np.concatenate((x_train_augmented, x_train_noisy))
y_train_augmented = np.concatenate((y_train_augmented, y_train))

# interpolate the training data
x_train_interpolated, y_train_interpolated = interpolate(x_train_scaled, y_train)
x_train_augmented = np.concatenate((x_train_augmented, x_train_interpolated))
y_train_augmented = np.concatenate((y_train_augmented, y_train_interpolated))

# define the objective function for Optuna
def objective(trial):
    # define the hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 1, 2)
    units = []
    for i in range(n_layers):
        units.append(trial.suggest_int(f'units_{i}', 4, 64))
        
    # create the model with the sampled hyperparameters
    model = Sequential()
    model.add(Dense(units[0], input_dim=len(impactful_features), activation='relu'))
    for i in range(1, n_layers):
        model.add(Dense(units[i], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    learning_rate = trial.suggest_uniform("lr", 1e-5, 1e-1)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    # define early stopping callback
    class AccuracyThresholdCallback(Callback):
        def __init__(self, threshold):
            super(AccuracyThresholdCallback, self).__init__()
            self.threshold = threshold

        def on_epoch_end(self, _, logs=None):
            val_accuracy = logs.get('val_accuracy')
            if val_accuracy >= self.threshold:
                self.model.stop_training = True
    accuracy_threshold = AccuracyThresholdCallback(threshold=0.95)
    
    # set the batch size and number of epochs
    batch_size = trial.suggest_int('batch_size', 16, 128)
    epochs = 100

    # train the model with the augmented training data
    model.fit(x_train_augmented, y_train_augmented, validation_data=(x_test_scaled, y_test),
              epochs=epochs, batch_size=batch_size, 
              callbacks=[accuracy_threshold], verbose = 0)

    # evaluate the model and print its accuracy
    _, accuracy = model.evaluate(x_test_scaled, y_test)
    return accuracy

# create an Optuna study
study = optuna.create_study(direction='maximize')

# optimize the hyperparameters
study.optimize(objective, n_trials = 100)

# print the best trial and its hyperparameters
best_trial = study.best_trial
print(f'\n Value: {best_trial.value}')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')