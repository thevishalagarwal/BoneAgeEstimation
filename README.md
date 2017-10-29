# BoneAgeEstimation
### Contribured by Vishal Agarwal, Jayanth Reddy

A Deep Learning aproach for estimating bone age from hand x-ray images

This was done as a part of nNVIDIA problem statement at an 24 hour Hack2Innovate hackathon, conducted at Indian Institute of Technology Guwahati.

The dataset used for this problem statement was taken from a RSNA challenge for Pediatric Bone Age. Some of the images from the dataset are shown below
![alt text](http://url/to/img.png)

We used the concept of end-to-end learning using InceptionV3 architechture with custom fully connected layers in the final stage for regression. Transfer learning was used to initialize the weights of InceptionV3 trained on ImageNet. Experiments were done with other architechtures as well and InceptionV3 was found to be performing better among those.
