#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

import pandas as pd

# Importing Dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#Encoding Categorical Features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_enc = LabelEncoder()
X[:,3] = label_enc.fit_transform(X[:,3])

hot_enc = OneHotEncoder(categorical_features=[3])
X = hot_enc.fit_transform(X).toarray()

#Remove first dummy feature to prevent dummy variable trap
X = X[:, 1:]

print(X)

# Splitting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler

sscaler = StandardScaler()
X_train = sscaler.fit_transform(X_train)
X_test = sscaler.transform(X_test)

#ANN
from keras.models import Sequential
from keras.layers import Dense

#https://stackoverflow.com/questions/47944463/specify-input-argument-with-kerasregressor
def baseline_model(x):
    def bm():
        # Initialising ANN
        model = Sequential()

        # Adding layer
        model.add(Dense(4, input_dim=5, activation='relu'))

        # Adding the second hidden layer
        model.add(Dense(6, kernel_initializer='normal', activation='relu'))

        # Adding the output layer
        model.add(Dense(1, kernel_initializer='normal'))

        # Compiling ANN
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        return model

    return bm



#The Keras wrapper object for use in scikit-learn as a regression estimator is called KerasRegressor.
#We create an instance and pass it both the name of the function to create the neural network model.

from keras.wrappers.scikit_learn import KerasRegressor
classifier = KerasRegressor(build_fn=baseline_model(5), epochs=100, batch_size=5, verbose=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(y_pred)

from sklearn.model_selection import cross_val_score

results = cross_val_score(classifier, X_train, y_train, scoring='r2')
print(results.mean(), results.std())