# Context

This repo is related to a linked in [post](https://www.linkedin.com/posts/andreaswombacher_ai-imageprocessing-ml-activity-7416793655799078913-KedQ?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAAV3PkBGnA_9CALg58xJHHnop6AJFyJr8E) about a Christmas project with my son to detect birds in bird house mounted to a glas door to the garden and storing the images. The project was using a simple web cam and a Raspberry PI 3.

In this repo, I share the code and explain the process on how you can replicate the setup in your garden ;-) Enjoy!


# Hardware

I was using a Raspberry PI 3 with a 32GB SD card. A simple USB based Web Cam which was able to provide images in 640*480 pixels.
Further, we used a plexiglas bird house for a few EURs from Action, which is a chain in the Netherlands and other parts of the world.
I am sure you can find a similar bird house online.


# Basic Setup

The basic setup was very temporary ;-)

<img src="photos/IMG_20251231_095756483_HDR.jpg" alt="Overall setup" width="500" />

The bird house was mounted to the door on the outside and the camera was attached with strings to the curtain rails inside. The Raspberry PI was attached also to the curtain rail using a rubber band.
The disadvantage of this setup was that closing and opening the curtain moved the camera and also resulted in a change of the position relative to the bird bouse.


# Configuration of the Raspberry PI




start on raspberry pi with

```
nohup ./start_video_detection.bsh > job.log 2>&1 &
```


Post process images to have the right settings for linkedin.

ffmpeg -i output_video2.mp4 -c:v libx264 -profile:v baseline -level 4.2 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k output_linkedin2.mp4
