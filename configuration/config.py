import os

BASE_PATH ="denoising-dirty-documents" # initialize the base path to the input documents dataset
TRAIN_PATH=os.path.sep.join([BASE_PATH,"train"]) # sep param is working to add /
CLEANED_PATH = os.path.sep.join([BASE_PATH, "train_cleaned"])

# define the path to our output features CSV file then initialize
# the sampling probability for a given row
FEATURES_PATH = "features.csv"
SAMPLE_PROB = 0.03
MODEL_PATH = "denoiser.pickle" # define the path to our document denoiser model