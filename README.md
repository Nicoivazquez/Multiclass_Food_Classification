# Multi-class Food Classification

## Data

This data set has 101 categories of data and 1,000 images for each category totaling 101,000 images in total. Found here 

[https://www.kaggle.com/kmader/food41](https://www.kaggle.com/kmader/food41) 

## Exploring data and Image Processing

The data came with an images folder and inside of that it had a folder for each class with jpg images for each, each folder had 1,000 different images. 

The first step was to write some code that would move the files into a holdout/test set. And while I was at it I decided to also through in an equal-sized validation set. 

This resulted in 3 new folders with 70,700 pictures in the train, 15,150 in the validation set, and 15,150 in the holdout/test set. Great! 

All these images are going to need some power to classify so I will use an AWS instance to run my models on all the different classes. 

For testing my models on my local machine, I used np.random.choice to choose 5 food from the directory and move into a mini training set. The same process of train/test split outlined above was done here.

### The randomly chosen ones were:

['apple_pie', 'waffles', 'gnocchi', 'chocolate_mousse', 'baklava'] 

## Lets see some food

Then I graphed all the images in order to see what they all looked like:

![images/Screen_Shot_2020-11-30_at_4.49.18_PM.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Screen_Shot_2020-11-30_at_4.49.18_PM.png)

To many to see clearly so lets grab a random set of 20. 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Screen_Shot_2020-11-30_at_4.54.47_PM.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Screen_Shot_2020-11-30_at_4.54.47_PM.png)

Much better. Seems like these are pictures of different, angles, locations, resolutions, and even on different plates and in conjunction with other items too. 

Looks like they are all already square nice, but resolutions look different, so I'll have to fix that before passing it on to the model!

For future images, I will show one image to make things cleaner for now but the next processes will be done on all of these images. 

## Creating data set from directory using Keras

[https://keras.io/api/preprocessing/image/](https://keras.io/api/preprocessing/image/)

One of the most used methods to load data for your model from folders is using the  flow_from_directory method as a way to load all my data for easy handling. Many ML models are picky but flow_from_directory has all the parameters you would need. In fact, I commonly use this method for easy data augmentation as well. 

## Augmentations Performed

- Grayscaling

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled.png)

Rotating 90 degrees 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%201.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%201.png)

In the end for my model after much tuning I ended up with shear_range=0.2, zoom_range = 0.2, and horizontal_flip. 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%202.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%202.png)

So thats alot of aumgentation options we have already, rotation_range=90,height_shift_range=0.5,shear_range=0.2, zoom_range=0.2, horizontal_flip=True, grayscale. But, we also have to remember to re size the image to 256,256 for easy handling in models and then later for transfer learning I use 224,224.

I also normalized the colours using rescale=1./255 in order to optimize for tensorflow.

### Code example:

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%203.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%203.png)

# Classifying

For basic models, my go-to is the TensorFlow docs. They have good models to get you started for image classification. Let's call it our Tensor model.

### Tensor model

model architecture

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%204.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%204.png)

Trainable parameters : 8,412,965

Let's go ahead and put in our 3,500 training images and 750 validation images for our 5 classes and see our baseline.

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%205.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%205.png)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%206.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%206.png)

pretty good better than guessing which is 20% for 5 classes right. 50% is an okay starting point for our Tensor Model.

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%207.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%207.png)

Lots of over-fitting, very apparent when training scores get better and better. Depends on what you want false negatives or false positive using precision and recall, but the f1 score is better for my general project of trying to classify food. 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%208.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%208.png)

I then ran the basic model with then dropout rate of .2 and then .5 to help with the overfitting that happens with this Tensor Model.

With the dropout rate of 50% being the best with a Test accuracy: 0.5667. 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%209.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%209.png)

## More Complex Custom Model

Every multi-class classification problem is different, and so using a cookie-cutter model like the one above is not always going to grant you the best results ...or is it?

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2010.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2010.png)

I used a more complicated model with 3 hidden layers with activation function relu and max_pooling to prevent overfitting and lighten training load on GPU, and 2 dropouts at .5 to prevent overtraining, for 67,127,621 trainable parameters. 17,054,021. Loss="categorical_crossentropy" and optimizer was Adam.

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2011.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2011.png)

Ended up with .2 with this model which is worse than before

I decided to lower the Kernal size to 3,3 from 4,4 and I got a bit better accuracy at .21 training and .28 validation. 

kernel size is the size of the filter, and the fillers have random weights and the Neural network is going to try to figure out patterns with those filters. 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2012.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2012.png)

Changed the drop out rate to .1 since 50% was really high for this model, as I was definitely not overfitting.

- Dropout for all the inputs 50 % means nothing. So I dropped them to .1 and only one towards the end of the model. I started at 50% because I didn't want to overfit like before but different architecture different results.

Reduced pool_size to pool_size = 4 = 2,2 .

- The max pool size layer used gets the largest pixels from a 4,4 picture, makes the 256 into 64. 16 pixels into 1.

Changed the dense neurons to 128 from 32

- The dense neurons are the final step of CNN and are used in transitional NN too. They help put everything together.

Final product looked like this 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2013.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2013.png)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2014.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2014.png)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2015.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2015.png)

When I ran a test accuracy score using my hold out data I got:

- Test accuracy: 0.478

Not bad but still a little lower than my basic model. 

- Usually adding more hidden layers of neurons generally improves accuracy, to a certain degree in many datasets to a certain limit which can differ depending on the problem. In this case, the Tensor Model is still ahead.

At this point, I'm thinking what can I push farther to get a better model. Sounds like a problem for Transfer learning to do this I will need a more advanced model, so let's get off my local machine and ssh into some power. 

## Amazon Web Services Instance

Starting a AWS instance for deep learning and starting up a jupyter notebook in the instance,

scp my images into the AWS instance as a quick and easy way to get my files.

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2016.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2016.png)

Get my python code into aws using git clone:

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2017.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2017.png)

Since we can von deeper, lets go all the way to transfer learning.

# **VGG -16**

I used VGG-16 it is one of the CNN architectures which is considered a very good model for Image classification. This model's architecture was used in the Image(ILSVR) in 2014.

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2018.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2018.png)

www.geeksforgeeks.com

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2019.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2019.png)

[https://neurohive.io/en/popular-networks/vgg16/](https://neurohive.io/en/popular-networks/vgg16/)

For transfer learning, I get everything but the softmax layer and the dense layers, so just after the last pooling, also called the head. This is what I will train to fit my model.

## Setting up the Augmented images for transfer learning

VGG16 takes an image size of 224 so let's do that. And augment the image using the same as before but adding brightness.

rescale=1./255,rotation_range=10,
width_shift_range=0.2, # horizontal shift
height_shift_range=0.2, # vertical shift
zoom_range=0.2, # zoom
horizontal_flip=True, # horizontal flip
brightness_range=[0.2,1.2]) # brightness)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2020.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2020.png)

## Big data set

So far we used three very models to solve our food classifying problem. But we also have only been giving our models small snacks and not the whole meal. Now that we have some power lets load all the 101 classes and put the computer to work. 

Loading the 101 classes onto the VGG16

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2021.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2021.png)

With Transfer learning, it is more important than ever to use a learning rate, as you don't want to quickly untrain the model you can do this by setting a learning rate and a decay.

optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])

So how did that model do? 

First run with VGG16 with top layer off

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2022.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2022.png)

Not good at around .0126 which is barely better than guessing in this situation. 

### Pushing limits of Tensor Model with 101 classes

While I was setting up the VGG16 model, I had in the background for 10 hours the Tensor model on the full data set and here are the results.

with 101 classes running this basic model was around 20 min per epochs and I trained it for 20 epochs and it gave me a test accuracy of 26.86

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2023.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2023.png)

**Top 1 accuracy**
 —Top 1 accuracy means that it guessed the image right. In an image classification problem, you extract the maximum value out of your final softmax outputs.

**Top 5 accuracy** — Top 5 accuracy is when you measure how often your predicted class falls in the top 5 values of your softmax distribution. In our case:

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2024.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2024.png)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2025.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2025.png)

## Xception model

### Set up

Used the same processed image set as before with 70,700 training images.

For the Xception model I decided to go for an more aggressive learning rate of .2 at the start to quicken things up for this model. 

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])

- Stochastic gradient descent is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset then update the weights of the 
model using the back-propagation of errors algorithm, which is called backpropagation.
- The amount that the weights are updated during training is referred to as the step size or the *learning rate*.

For Momentum I went with .9

- Momentum is another argument in SGD optimizer which we could tweak to obtain faster convergence. Unlike regular SGD, the momentum method helps the parameter vector to build up velocity in any direction with constant gradient descent so as to prevent oscillations. A typical choice of momentum is between 0.5 to 0.9. I went with .9.
- 

This resulted in a model with 21,068,429 total parameters. with only 206,949 trainable parameters. 

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2026.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2026.png)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2027.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2027.png)

So now that we trained the head lets train the rest of the model. To do this we unfreeze all the layers and run the model again. This time with a much lower learning rate since we don't want to mess up the model previous weights. 

I went with a 0.1 lr. and 0.001 decay. 

optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
metrics=["accuracy"])

This resulted in 21,068,429 total parameters just like before but the trainable parameters went up to 21,013,901 so much more to train on here. 

An the results are shocking. 

Xception Model Results:

Top1 Accuracy of .7936

Top5 Accuracy of .93969

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2028.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2028.png)

Very apparent overfitting in this model visualized perfectly from the growing difference between training and validation accuracy lines. Obvious next step is a drop out layer. 

# Improving?

### First Model Summary

Tensor Model

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2029.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2029.png)

Top1 Accuracy = .264

Top5 Accuracy = .55

### Last Model Summary

Xception Model

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2030.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2030.png)

Top 1 Test Accuracy = .794

Top 5 Test Accuracy = .94 

# Predicting

Now we are in a place to predict.

[https://images1.miaminewtimes.com/imager/u/745xauto/11655890/sushi-erika-miami-credit-fujifilmgirl-24.jpg](https://images1.miaminewtimes.com/imager/u/745xauto/11655890/sushi-erika-miami-credit-fujifilmgirl-24.jpg)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2031.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2031.png)

This image most likely belongs to cup cakes with a 2.65 percent confidence.

[https://thekitchengirl.com/wp-content/uploads/Homemade-Hummus-2.jpg](https://thekitchengirl.com/wp-content/uploads/Homemade-Hummus-2.jpg)

![Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2032.png](Multi-class%20Food%20Classification%2017e7015928574119a5b13de8cc92fc85/Untitled%2032.png)

This image most likely belongs to hummus with a 76.55 percent confidence.

## Final thoughts

So at this point, you might be asking why did my data do best with the basic plug and play model at the beginning with the 5 classes. Well, there are some things to note. First is that some foods might just be very easy for the computer to differentiate others more difficult. The plates of food must have lots of features that are very different. This could have helped the simple model more than the custom model. For many problems, you can begin with a low hidden layer and get decent results.  Just one hidden layer can theoretically model even the most complex functions if it has enough neurons. But for complex problems, deep networks have a much higher parameter efficiency than shallow ones since they can model complex functions using exponentially fewer neurons than shallow simple models, so they are more efficient in a way. They offer more performance.

Secondly, the Data augmentation also resulted in a less is better for the models. The less the data is augmented the fewer images the model has to train with and the less generalizable the model, but also if you go too far with data augmentation you can end up with images that are nothing like your test data. 

Thirdly, The pre-trained model with weights from VGG16 did fairly bad and I'm not happy with the results from the Model. The Xception model did Xceptional though and I would definitely work some more on trying to prevent the overfitting to my training and in conjunction with training for more epochs with slowly unfreezing layers, I think it's possible to get to 90% test accuracy. 

## Next steps:

Simple next steps would be to use a grid search to find optimal hyper perimeters. Doing grid takes a long time but will give you the optimal hyperparameters. The ones I am most interested in is batch size and data augmentation. 

With more time, taking this project a step further would include getting deeper with transfer learning and re-training the model layers to fit my data set more. You can do this by freezing layers slowly rather than all at once as I did. 

## Potential use cases:

This project could potentially be useful for food quality control, or even as part of a restaurant application, like yelp, where users post their pictures of food, and if a certain restaurant does not have pictures for items on their menu when users submit pictures the pictures get classified to the right menu item.
