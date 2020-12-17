###################################################################################################################
###################           HYBRID DL MODEL ---->  SOM + ANN         ############################################
###################################################################################################################

###################################################################################################################
###################################   PART-1 IDENTIFY FRAUDS USING SOM (SELF ORGANIZING MAPS) MODEL   #############
###################################################################################################################

################################# I> IMPORT THE LIBRARIES  ########################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################# II> IMPORT THE DATASET   ########################################################

dataset = pd.read_csv('E:\\DESK PROJECTS\\Github\\DATASETS\\Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

################################# III> FEATURE SCALING  ###########################################################

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

################################# IV> TRAIN THE SELF ORGANIZING MAPS (SOM) MODEL  #################################

from minisom import MiniSom # pip install minisom--> try this in case of any import errors
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

################################# V> VISUALIZE THE RESULTS  #######################################################

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

################################ VI> FIND THE FRAUDS  ###############################################################

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#####################################################################################################################
###################################   PART-2 GO FROM UNSUPERVISED TO SUPERVISED DEEP LEARNING    ####################
#####################################################################################################################

################################### I> CREATE THE MATRIX OF FEATURES  ###############################################

customers = dataset.iloc[:, 1:].values

################################### II> CREATE THE DEPENDENT VARIABLE  ##############################################

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

################################## III> FEATURE SCALING  ############################################################
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#####################################################################################################################
###################################   PART-3 MAKE THE ANN MODEL   ###################################################
#####################################################################################################################

################################## I> IMPORT THE KERAS LIBRARIES AND PACKAGES  ######################################

from keras.models import Sequential
from keras.layers import Dense

################################## II> INITIALIZE THE ANN MODEL  ####################################################

classifier = Sequential()

################################## III> ADD THE INPUT LAYER AND THE 1st HIDDEN LAYER  ###############################

classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

################################## IV> ADD THE OUTPUT LAYER  ########################################################

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

################################## V> COMPILE THE ANN MODEL  ########################################################

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

################################## VI> FIT THE ANN MODEL TO THE TRAINING SET  #######################################

classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

################################## VII> PREDICT THE POSSIBILITIES OF FRAUDS  ########################################

y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
y_pred
