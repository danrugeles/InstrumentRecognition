Instrument Recognition
=======================

This code classifies a set of instruments given as a set of images.

-To test: python testing.py <name image>
the result of classification is the image output.jpg i in /test directory

-To train: python training.py
and manually label the instruments (only numbers accepted)

-To validate the dataset and print the confusion matrix:  python validate.py

-Directories:

--/storage   used to store all features *.npy they correspond to the contours of the instruments after being normalized by rotation

--/test  place the image that you want to tes

--/images  images used for training
