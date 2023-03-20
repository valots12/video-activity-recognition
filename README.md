# NN-Classifier for human activity videos from HMDB dataset 

***Part of the Foundations of Deep Learning project | UniMiB***

The purpose of the project is the development of different classification algorithms in order to predict and recognize the simplest human actions and compare their performance.

## Dataset

The selected dataset is named HMDB (Human Emotion DB) and is available at the following [link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). Each observation corresponds to one video, for a total of 6849 clips. Each video has been associated with one of 51 possible classes, each of which identifies a specific human behavior. Moreover, the classes of actions can be grouped into:
- general facial actions, such as smiling or laughing;
- facial actions with object manipulation, such as smoking;
- general body movements, such as running;
- body movements with object interaction, such as golfing;
- body movements for human interaction, such as kissing.

Due to computational problems, we have chosen only 19 classes (general body movements) on which to train the human activity recognition algorithm.

## LRCN approach

LRCN is a class of architectures which combines Convolutional layers and Long Short-Term Memory (LSTM).

BASIC LRCN
- Convolutional2D Layer
- LSTM Layer
- Dense Layer (fully connected)

ADVANCED LRCN
- 3 Convolutional2D Layers
- LSTM Layer
- Dense Layer (fully connected)

## MoveNet approach

MoveNet is an ultra fast and accurate model that detects 17 keypoints on a body. The model is offered in two variants, known as Lightning and Thunder. Lightning is intended for latency-critical applications, while Thunder is intended for applications that require high accuracy.

MoveNet is a bottom-up estimation model that uses heatmaps to accurately localize human keypoints. The architecture consists of two components: a feature extractor and a set of prediction heads.

The feature extractor in MoveNet is MobileNetV2 with an attached feature pyramid network (FPN), which allows for a high-resolution, semantically rich feature map output. There are four prediction heads attached to the feature extractor, responsible for densely predicting:
- person center heatmap: predicts the geometric center of person instances;
- keypoint regression field: predicts full set of keypoints for a person, used for grouping keypoints into instances;
- person keypoint heatmap: predicts the location of all keypoints, independent of person instances;
- 2D per-keypoint offset field: predicts local offsets from each output feature map pixel to the precise sub-pixel location of each keypoint.

<img width="583" alt="225679688-abdbc201-8b36-40f4-8ab9-db7262ed827d" src="https://user-images.githubusercontent.com/63108350/226449042-bbc1e10d-2ee8-49bd-b991-3bdad93c37ec.png">

## Results

| Network       | Validation Accuracy |
| ------------- | -------------- |
| Basic LRCN    |       34%      |
| Adavnced LRCN |       41%      |
| MoveNet       |       70%      |  

## References

[1] [Deep Learning Models for Human Activity Recognition](https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/)

[2] [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389?source=post_pagel)

[3] [Long-term Recurrent Convolutional Network for Video Regression](https://towardsdatascience.com/long-term-recurrent-convolutional-network-for-video-regression-12138f8b4713)

[4] [Long-term Recurrent Convolutional Networks](https://jeffdonahue.com/lrcn/)

[5] [Next-Generation Pose Detection with MoveNet](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)
