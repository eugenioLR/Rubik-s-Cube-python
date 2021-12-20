import sys
sys.path.append("..")

import numpy as np
import numpy.random
import torch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

cube_data = np.loadtxt("NN_input.csv", delimiter=',')
targets = np.loadtxt("NN_target.csv ", delimiter=',')

cube_data = torch.from_numpy(cube_data)

# Amount of colors in each face
colors_in_face = []
for i in range(len(cube_data)):
    colors_in_face.append(torch.tensor([len(torch.unique(i)) for i in torch.reshape(cube_data[i], [6, 9])]))
colors_in_face = torch.stack(colors_in_face)

# Corners
indices = torch.tensor([0,2,6,8])
indices = torch.cat([indices+(9*i) for i in range(6)])
corners = cube_data[:,indices]

inputs = torch.cat([cube_data, colors_in_face], dim=1)
#inputs = torch.cat([corners, colors_in_face], dim=1)
#inputs = colors_in_face
inputs = inputs.numpy()

data_amount = 25000

order = list(range(len(inputs)))
np.random.shuffle(order)
order = order[:data_amount]

inputs = inputs[order].astype(int)
targets = targets[order].astype(int)

print(inputs.shape)
print(targets.shape)

X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)

"""
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
grid = GridSearchCV(svm.SVR(), param_grid, refit=True, verbose=3)


grid.fit(X_train, y_train)
"""

regr = svm.SVR(kernel='rbf', gamma=0.005, C=0.5, verbose=True)

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test).astype(int)

print(y_test)
print(y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))

print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
