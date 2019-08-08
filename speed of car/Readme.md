Welcome to the comma.ai 2017 Programming Challenge!

The goal is to predict the speed of a car from a video. The input files can be found at http://commachallenge.s3-us-west-2.amazonaws.com/speed_challenge_2017.tar

train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
train.txt contains the speed of the car at each frame, one speed on each line.

test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
The deliverable is test.txt

The evaluation is done on test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.


Approach 1:
I first used openCV to extract the frames per second from the video and then train a 4 layer Convolutional Neural Network in Keras with Batch Normalization, 2D convolution with strides and callbacks with 3 dense layers of fully connected network at the end. Total trainable parameters approx 9 Million

Current best performance of the model: MSE = 1.7
Conclusion:  training error < 1.7 while the validation error = 0.8 #Fixed the overfitting issue. Submitted results

