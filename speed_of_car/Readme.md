Welcome to the comma.ai 2017 Programming Challenge!

Basically, the goal is to predict the speed of a car from a video.

train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
train.txt contains the speed of the car at each frame, one speed on each line.

test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
The deliverable is test.txt

The evaluation is done on test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

I first use openCV to extract the frames per second from the video and then train a 4 layer Convolutional Neural Network in Keras with Batch Normalization, 2D convolution, pooling and callbacks. 

Current best performance of the model: MSE = 9.42