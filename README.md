# Hamsa
This is a UI to interact with 3D models created in Autodesk's Fusion 360. The user has the ability to zoom in, pan, and rotate around the object.

## Training the Model
I first started taking images. I took ~200 images of opened and closed hands through my webcam, ending up with 398 images total. Images were taken in a variety of locations, lighting patterns, etc. After this, I used a Yolov3 [labeling tool](https://github.com/Cartucho/OpenLabeling) to draw a bounding box around each opened and/or closed hand in the frame.


After this, I read the `README.MD` of the [C++ version of the Yolo models called darknet](https://github.com/AlexeyAB/darknet). I used this to create the necessary files for training and inference, based on my own labeled training images. Once all files were prepared, I downloaded the `darknet` folder as a zip to be stored in my Google Drive. I then opened this `.zip` file in a Google Colab (found in the training folder), and trained on my custom dataset. After training was completed, I saved the `.zip` file again so I could access the `yolov3-tiny0_best.weights` file, which contained the weights to run on my custom config for Yolov3, named `yolov3-tiny0-dark.cfg`

## Setting up the Weights in python
This part was simple: I cloned the [Yolov3](https://github.com/ultralytics/yolov3) repo which is essentially a Python wrapper for the original [C++ version called darknet](https://github.com/AlexeyAB/darknet). I then went into the `cfg/` portion of __yolov3/__ and added my custom config file that I used during training, as well as the `yolov3-tiny0_best.weights` file to the `weights/` directory. Once that was done, I could run the default commands found in the repository to get predictions off of my trained model.

## Actually adapting the model into a UI
This was done by editing the `detect.py` file inside of the __yolov3/__ directory, and saving it as a new file named `detect_box.py`. A seperate file named `hand_altering.py` holds multiple functions used inside of `detect_box.py`. To see what was actually edited between the files, see [here](https://www.diffchecker.com/lANKgUWu)

## Hand Gestures
