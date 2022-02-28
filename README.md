# Continuum Robotics Laboratory (CRL)
![CRL logo](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcrl.utm.utoronto.ca%2Fcrl%2Fwp-content%2Fuploads%2F2021%2F01%2FCRLab-logo_dark_1_large.png&f=1&nofb=1)

# What are Continuum Robots?
Continuum robots, which are biologically inspired and organic compliant structures, differ fundamentally from traditional robots, which rely on a rigid joint-link composition. Their appearance is evocative of animals and organs such as trunks, tongues, worms, and snakes. Composed of flexible, elastic, or soft materials, continuum robots can perform complex bending motions and appear with curvilinear shapes. However, their softness and deformability raise major challenges, including the continuous recognition of the manipulator’s actual shape.

# Research Purpose
At the Continuum Robotics Laboratory at University of Toronto, we create innovative continuum robot designs with novel features. To test our continuum robots and validate our models, algorithms, and controllers, we perform experiments on the bench-top and observe the robot's performance using different sensors. The aim of this research project is to implement calibration and computer-vision algorithms for an existing multi-camera system to capture the motion and sense the shape of tendon-driven continuum robots.

![Research Poster](https://github.com/YasPHP/Continuum-Robotics/blob/main/data/results/UofT%20Research%20Poster.png)

Undergraduate Research Symposium Presentation:

[![](https://img.youtube.com/vi/0yfp1_y0DcI/0.jpg)](https://youtu.be/0yfp1_y0DcI)

# Project Outcomes
The following functionalities have to be implemented using Python or C++ using OpenCV and Qt:
- Run, save, and load extrinsic and intrinsic camera calibration for a set of three cameras
- Capture images and movies from up to three cameras simultaneously
- Extraction a depth map from the images
- Segmentation of tendon-driven continuum robot features (e.g. spacer disks)
- Determining the pose of the continuum robot's tip and its shape
- Building a GUI encompassing the above

## Links
- [State of the Art](https://docs.google.com/presentation/d/1KUIQfTkrYJrEYH9h86QRs4NgSrWIPhfqEkEYPvq-XTY/edit?usp=sharing)
- [Intermediate Presentation](https://docs.google.com/presentation/d/1RT5F-ng_JTA60MQ3RYI30OvNvZX3M8dQOKFBXEosT2o/edit?usp=sharing)
- [Final Presentation](https://docs.google.com/presentation/d/14V83sBujteqvNrxjuqr8CkWf_xnvpeKI6Qa8R-Xt6EE/edit?usp=sharing)
- [Journal](https://docs.google.com/document/d/1CyAVqjSZbK8LlLNESrIe_CeDBRzSGpMibFqUTEYHJQY/edit?usp=sharing)
- [Undergraduate Research Poster](https://github.com/YasPHP/CRL/blob/main/data/results/SmartiGras%202021%20Yasmeen%20Hmaidan%20Poster%20.png)
- [ROP Report](https://github.com/YasPHP/CRL/blob/main/data/results/CRL%20ROP%20Report%202021%20Yasmeen%20Hmaidan.pdf)




# Get Started
Clone this repository and install the other dependencies with ```pip```:
```
git clone https://github.com/YasPHP/CRL.git
cd CRL
pip install -U -r requirements.txt
```

# Multi-Camera System (Stereovision Camera Pairs)

- Note: to escape each python script's steps, either press 'x' on the popped up window each time and then the 'esc' key twice when the detected ArUco marker screen pops up at the end (as there is no 'x'/minimize/maximize screen option for those last two outputted images).
- You know you will have reached the end of each program, when the terminal outputs the TDCR's final ArUco Marker Coordinates matrix!

## Run Camera 1 & 2 Program
```
python cam1and2.py
```

## Run Camera 1 & 3 Program
```
python cam1and3.py
```

## Run Camera 2 & 3 Program
```
python cam2and3.py
```

# Multi-n-Camera System Extension
## Run Individual methods
```
python test_functions.py
```

## Initialize n Cameras
```
python calibrate.py
```


# ArUco Markers
- Print out these [ArUco Markers](https://github.com/YasPHP/CRL/tree/main/aruco_markers) for future TDCR detection tests
- Generate your own ArUco markers [here](https://chev.me/arucogen/)

# Credited Tutorial Sources
- [OpenCV Tutorial Series and StackExchange](https://docs.opencv.org/master/d3/d81/tutorial_contrib_root.html)
- [St. Pölten University of Applied Sciences Computer Vision](https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/)
- [The Carnegie Mellon Robotics Institute](https://www.cs.cmu.edu/16385/lectures/lecture13.pdf)
- [Camera Calibration on Medium](https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7)

