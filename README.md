# BoneAgeEstimation
### Contribured by Vishal Agarwal, Jayanth Reddy

A Deep Learning aproach for estimating bone age from hand x-ray images

This was done as a part of nNVIDIA problem statement at an 24 hour Hack2Innovate hackathon, conducted at Indian Institute of Technology Guwahati.

The dataset used for this problem statement was taken from a RSNA challenge for Pediatric Bone Age. Some of the images from the dataset are shown below

![](/BoneAgeEstimation/dataset_sample/1377.png)

We used the concept of end-to-end learning using InceptionV3 architechture with custom fully connected layers in the final stage for regression. Transfer learning from ImageNet pre-trained models was used to initialize the weights of InceptionV3. Experiments were done with other architechtures as well and InceptionV3 was found to be performing better among those.
The evaluation metric used in our model is Mean Absolute Error(MAE).

#### Evaluation Metrics

Train - loss=123.082 MAE=8.726
Valid - loss=196.535 MAE=10.958
Test  - loss=125.307 MAE=8.978

#### Future Work
- Advanced feature extraction techniques can be used to improve further accuracy.
- The RSNA Pediatric Bone Age Dataset also contains gender labels. So the model trained separately for male and female classes will improve the accuracy.
