from numpy.distutils.command.config import config

from configuration import config
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


# initialize lists to hold our features and target predicted values
print("Loading Dataset...")
features = []
targets = []
# loop over the rows in our features CSV file
for row in open(config.FEATURES_PATH):
	# parse the row and extract (1) the target pixel value to predict
	# along with (2) the 5x5=25 pixels which will serve as our feature
	# vector
	row = row.strip().split(",")
	row = [float(x) for x in row]
	target = row[0]
	pixels = row[1:]
	# update our features and targets lists, respectively
	features.append(pixels)
	targets.append(target)
	# convert the features and targets to NumPy arrays
features = np.array(features, dtype="float")
target = np.array(targets, dtype="float")

(trainX, testX, trainY, testY) = train_test_split(features, target,
	test_size=0.25, random_state=42)
print("Training Model....")
model = RandomForestRegressor(n_estimators=10)
model.fit(trainX, trainY)
# compute the root mean squared error on the testing set
print("Evaluating model...")
preds = model.predict(testX)
rmse = np.sqrt(mean_squared_error(testY, preds))
print("RMSE: {}".format(rmse))
# serialize our random forest regressor to disk
f = open(config.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()



