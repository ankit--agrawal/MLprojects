CNN are widely used for computer vision tasks. In this project, I implemented popular CNN architectures (transfer learning with keras) on ImageNet dataset.
ImageNet dataset is a project aimed at labeling (manually) and categorizing images into almost 22,000 separate categories for computer vision research.
In this implementation, we are considering a subset of dataset i.e. 1,000 separate categories.

Models used:
1. VGG16
2. VGG19
3. ResNet50
4. InceptionV3
5. Xception
6. MobileNet
7. DenseNet121
8. NASNetMobile
9. Xception

How to run the code:
python main.py --image <path/to/image> --model <model_name>

How to interpret the results:
1. The image will be correctly preprocessed based on the model selected (including image scaling).
2. If you are running any of these models for the 1st time, you will automatically download the weights for the appropriate model during the 1st run (introduces  time overhead only for the 1st run).
3. The final result is top 5 predictions with confidence (probability) in Descending order.


The documentation of all these architectures is available at https://keras.io/applications/

Resources that explain these model architectures:
https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/

What is next?
I want to implement these models with 'GlobalAveragePooling' layer to reduce the number of trainable parameters. Recent articles that I came across claim that we do not need to add
Dropout, Flatten layers to CNN models anymore.

