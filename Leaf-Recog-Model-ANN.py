
# Installing Keras
# pip install --upgrade keras

# Importing the libraries


import pandas as pd


# Importing the dataset
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
root= tk.Tk()
def getEXC():
    global df  
    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path)
    return df

root.mainloop()
data=getEXC()

X = data.iloc[:, :8].values



# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_7 = LabelEncoder()
X[:, 7]= labelencoder_X_7.fit_transform(X[:, 7])
onehotencoder = OneHotEncoder(categorical_features = [7])
X= onehotencoder.fit_transform(X).toarray()
y=X[:, :12]
X=X[:, 12:]

#Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

#Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Part 2 - make the ANN!

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', input_dim = 7))
#
## Adding the first hidden layer
classifier.add(Dense(output_dim = 101, init = 'uniform', activation = 'tanh'))
#
# Adding the output layer
classifier.add(Dense(output_dim = 12))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss='mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1, epochs = 5000)

# Part 3 - Making the predictions and evaluate the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
