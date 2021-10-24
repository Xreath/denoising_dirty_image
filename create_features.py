from configuration import config
from helpers_functions.helpers import blur_and_th
from imutils import paths
import progressbar
import random
import cv2

# grab the paths to our training images
trainPaths = sorted(list(paths.list_images(config.TRAIN_PATH)))
cleanedPaths = sorted(list(paths.list_images(config.CLEANED_PATH)))
# initialize the progress bar
widgets = ["Creating Features: ", progressbar.Percentage(), " ",progressbar.Bar(), " ", progressbar.ETA()]
prog_bar = progressbar.ProgressBar(maxval=len(trainPaths), widgets=widgets).start()


imagePaths=zip(trainPaths,cleanedPaths)
csv=open(config.FEATURES_PATH,'w') # zip our training paths together, then open the output CSV file for writing

for (i,(trainPaths,cleanedPaths)) in enumerate(imagePaths):
    trainImage=cv2.imread(trainPaths)
    cleanImage=cv2.imread(cleanedPaths)
    trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
    cleanImage = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2GRAY)

    trainImage = cv2.copyMakeBorder(trainImage, 2, 2, 2, 2,
                                    cv2.BORDER_REPLICATE)
    cleanImage = cv2.copyMakeBorder(cleanImage, 2, 2, 2, 2,
                                    cv2.BORDER_REPLICATE)

    trainImage = blur_and_th(trainImage)
    cleanImage = cleanImage.astype("float") / 255.0 # scale the pixel intensities in the cleaned image from the range [0, 255] to [0, 1] (the noisy image is already in the range [0, 1])


    #slide a 5x5 window across the images
    for y in range(0,trainImage.shape[0]):
        for x in range(0, trainImage.shape[1]):
        #extract the window ROIs for both the train image and clean image, then grab the spatial dimensions of the ROI
            trainROI=trainImage[y:y+5,x:x+5]
            cleanROI=cleanImage[y:y + 5, x:x + 5]
            (rH,rW)=trainROI.shape[:2]
            # if the ROI is not 5x5, throw it out
            if rW != 5 or rH != 5:
                continue
            # our features will be the flattened 5x5=25 raw pixels
            # from the noisy ROI while the target prediction will
            # be the center pixel in the 5x5 window
            features = trainROI.flatten()
            target = cleanROI[2, 2]

            if random.random() <= config.SAMPLE_PROB: # thereby reducing the total number of rows in the file
                # write the target and features to our CSV file
                features = [str(x) for x in features]
                row = [str(target)] + features
                row = ",".join(row)  # I convert list type to string type as a one value
                csv.write("{}\n".format(row))

    # update the progress bar
    prog_bar.update(i)
# close the CSV file
prog_bar.finish()
csv.close()