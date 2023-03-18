# USAGE
# python Generate-Images-Useing-DataAugment.py --dataset "Directory input files" --output "Directory out files"
# python Generate-Images-Useing-DataAugment.py --dataset "Directory input files" --output "Directory out files"
# python Generate-Images-Useing-DataAugment.py --dataset "Directory input files" --output "Directory out files"
# python Generate-Images-Useing-DataAugment.py --dataset "Directory input files" --output "Directory out files"
# python Generate-Images-Useing-DataAugment.py --dataset "Directory input files" --output "Directory out files"
# python Generate-Images-Useing-DataAugment.py --dataset "Directory input files" --output "Directory out files"


# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
import os
from imutils import paths

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default=100,
	help="# of training samples to generate")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the image, swap color channels, and resize it to be a fixed
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	# update the data and labels lists, respectively
	data.append(image)
for image in data:
	# load the input image, convert it to a NumPy array, and then
	# reshape it to have an extra dimension
	print("[INFO] loading example image...")
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
	aug = ImageDataGenerator(
		rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.75,1.5])
	
	total = 0
	# construct the actual Python generator
	print("[INFO] generating images...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
		save_prefix="image", save_format="jpg")

	# loop over examples from our image data augmentation generator
	for image in imageGen:
		# increment our counter
		total += 1

		# if we have reached the specified number of examples, break
		# from the loop
		if total == args["total"]:
			break