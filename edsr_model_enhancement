import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to super resolution model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image we want to increase resolution of")
args = vars(ap.parse_args())

# initialize OpenCV's super resolution DNN object and load the super
# resolution model from disk
print("[INFO] loading super resolution model: {}".format(args["model"]))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])

# Set the model name and scale for EDSR_x4.pb model
sr.setModel("edsr", 4)

# load the input image from disk and display its spatial dimensions
image = cv2.imread(args["image"])
print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))

# use the super resolution model to upscale the image, timing how
# long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] super resolution took {:.6f} seconds".format(end - start))

# show the spatial dimensions of the super resolution image
print("[INFO] w: {}, h: {}".format(upscaled.shape[1], upscaled.shape[0]))

# show the original input image and the super resolution output
cv2.imshow("Original", image)
cv2.imshow("Super Resolution", upscaled)
cv2.waitKey(0)

