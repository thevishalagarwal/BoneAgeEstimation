# BoneAgeEstimation
### Contribured by Vishal Agarwal, Jayanth Reddy

A Deep Learning aproach for estimating bone age from hand x-ray images

This was done as a part of NVIDIA problem statement at an 24 hour Hack2Innovate hackathon, conducted at Indian Institute of Technology Guwahati.

The dataset used for this problem statement was taken from a RSNA challenge for Pediatric Bone Age. Some of the images from the dataset are shown below

[![1377.png](https://s19.postimg.org/4kjq1330j/1377.png)](https://postimg.org/image/6ccovzmdb/)       [![1380.png](https://s19.postimg.org/kk2dknm7n/1380.png)](https://postimg.org/image/jujl8alnz/)         [![1489.png](https://s19.postimg.org/gaxnihydv/1489.png)](https://postimg.org/image/8uydwpaof/)

We used the concept of end-to-end learning using InceptionV3 architechture with custom fully connected layers in the final stage for regression. Transfer learning from ImageNet pre-trained models was used to initialize the weights of InceptionV3. Experiments were done with other architechtures as well and InceptionV3 was found to be performing better among those.
The evaluation metric used in our model is Mean Absolute Error(MAE).

#### Evaluation Metrics

Train : MAE = 8.726

Valid : MAE = 10.958

Test  : MAE = 8.978

#### Future Work
- Advanced feature extraction techniques can be used to improve further accuracy.
- The RSNA Pediatric Bone Age Dataset also contains gender labels. So the model trained separately for male and female classes will improve the accuracy.
