import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import Xception
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import argparse


def main():
    
    '''there are other versions of these models available as well. The major difference is the size, # of parameters and depth.
    for the purpose of this implementation, we are using the versions with smaller newtworks because the performance doesn't
    change significantly between these versions'''
    #models implemented
    Models = {"vgg16":VGG16,
              "vgg19":VGG19,
              "inception":InceptionV3,
              "xception":Xception,
              "resnet":ResNet50,
              "nasnnet":NASNetMobile,
              "mobilenet":MobileNet,
              "densenet":DenseNet121}
    
    if args["model"] not in Models.keys(): #correct model name should be passed
        raise AssertionError("Incorrect model selected")
        

    if args["model"] in ("inception","xception"):
        inputShape = (299,299)
        preprocess = preprocess_input
    else:
        inputShape = (224,224)
        preprocess = imagenet_utils.preprocess_input

    #load network weights from disk for the model selected above
    print("loading weights for {}...".format(args["model"]))
    Network = Models[args["model"]]
    model = Network(weights="imagenet")
    
    #load a image to predict and preprocess it
    print("loading and preprocessing image...")
    im = load_img(args["image"], target_size = inputShape) #resize the image
    im = img_to_array(im)
    
    im = np.expand_dims(im,axis=0) #(inputShape) -> (1,inputShape)
    im = preprocess(im)
    
    #classify the image
    pred = model.predict(im)
    p = imagenet_utils.decode_predictions(pred)
    print("Image successfully classified using {}...".format(args["model"]))
    
    #top 5 predictions
    for (i, (imagenetID, label, probability)) in enumerate(p[0]):
        print("{} {}:{}%".format(i+1,label,probability*100))
        
    #original_im = cv2.imread(args["image"])
    #(imagenetID, label, probability) = p[0][0]
    plt.imshow(args["image"])
    
    
if __name__=="__main__":
    #taking command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i","--image",required=True,help="input image path")
    argparser.add_argument("-m","--model",type=str,default="resnet",help="CNN model that you wish to load")
    args = vars(argparser.parse_args())
    
    main()
    
    
    