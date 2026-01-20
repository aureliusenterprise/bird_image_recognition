# Context

This repo is related to a linkedin [post](https://www.linkedin.com/posts/andreaswombacher_ai-imageprocessing-ml-activity-7416793655799078913-KedQ?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAAV3PkBGnA_9CALg58xJHHnop6AJFyJr8E) about a Christmas project with my son to detect birds in a bird house mounted to a glas door to the garden and storing the images. The project was using a simple web cam and a Raspberry PI 3.

In this repo, I share the code and explain the process on how you can replicate the setup in your garden ;-) Enjoy!

I was able to collect images, which I then turned into a video again:

[![Watch the video](photos/video_frame.png)](https://dms.licdn.com/playlist/vid/v2/D4E05AQG8Oqlj82KBCA/mp4-720p-30fp-crf28/B4EZu2.hbbKcCM-/0/1768301403876?e=1769533200&v=beta&t=9aNf6FdDRCoz75MZMCquKiuNUrOmd4Uxu3cVpnpbVfk)


# Hardware

I was using a Raspberry PI 3 with a 32GB SD card. A simple USB based Web Cam which was able to provide images in 640*480 pixels.
Further, we used a plexiglas bird house for a few EURs from Action, which is a chain in the Netherlands and other parts of the world.
I am sure you can find a similar bird house online.


# Basic Setup

The basic setup was very temporary ;-)

<img src="photos/IMG_20251231_095756483_HDR.jpg" alt="Overall setup" width="500" />

The bird house was mounted to the door on the outside and the camera was attached with strings to the curtain rails inside. The Raspberry PI was attached also to the curtain rail using a rubber band.
The disadvantage of this setup was that closing and opening the curtain moved the camera and also resulted in a change of the position relative to the bird house.


# Configuration of the Raspberry PI

The Raspberry PI was setup as a web server running a Apache HTTPD service. 
Further, a python file was running in deamon mode to read the camera images, :
 - detect movement and store the images, or later on
 - detect movement, check whether the image shows a bird and then store the image and report the timestamp in a csv file.

The stored files are stored in the directory /var/www/html/ directory, which is the default content directory of the web server.
To be able to see the files, the /var/www/html/index.html file has to be removed, such that the content of the directory is visible.

To check the quality of the detected birds in the second case I created a folder /var/www/html/non/ where all images of movements are stored, which are NOT showing a bird.

To ensure that the python program can write to the volder /var/www/html/ and /var/www/html/non/ perform the following commands:

```bash
sudo chmod o+w /var/www/html/
sudo chmod o+w /var/www/html/non/
```

To work easily with the repo on the Raspberry PI clone the repo on the Raspberry PI.
```bash
git clone https://github.com/aureliusenterprise/bird_image_recognition.git
```


# Overall approach

The overall approach is following a classicial machien learning approach:
 - collect an intitial dataset as a basis to train a model
 - annotate the data manually and create a training dataset
 - Train the model using a pre-trained model
 - deploy the trained model to the Raspberry PI and start the bird detection

Some important points:
 - Basemodel is based on a 160*160 images, thus make sure that the images in the training dataset are resized to this size or at least are quadratic. Otherwise the resizing will distort the image significantly reducing the benefit of a pre-trained model. 
 - Basemodel is pretrained with 1000 classes including birds.
 - Model is working well with different light conditions
 - Having all kinds of preprocessing (cropping & resizing) in the model resulted in problems doing the GradCAM to evaluate wrong positives and false negatives afterwards. Thus, these operations were applied before the model is applied.
 - I share the model I have used as a transformer in ```bird_classifier_int8.tflite```.


# Step by step 

In this section I will provide some more details on what the different code part do and how they are called. 

## Collect data to create a trainings dataset


## Create dataset

## Train the model

## Detect birds with a trained model

start on raspberry pi with

```
nohup ./start_video_detection.bsh > job.log 2>&1 &
```


Post process images to have the right settings for linkedin.

ffmpeg -i output_video2.mp4 -c:v libx264 -profile:v baseline -level 4.2 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k output_linkedin2.mp4
