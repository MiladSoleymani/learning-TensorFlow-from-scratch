# learning TensorFlow from scratch

## Course 1 : Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
---
### A New Programming Paradigm (Week_1)

In week 1 you'll get a soft introduction to what Machine Learning and Deep Learning are, and how they offer you a new programming paradigm, giving you a new set of tools to open previously unexplored scenarios. All you need to know is some very basic programming skills, and you'll pick the rest up as you go along. You'll be working with code that works well across both TensorFlow 1.x and the TensorFlow 2.0 alpha. To get started, check out the first video, a conversation between Andrew and Laurence that sets the theme for what you'll study...

### Introduction to Computer Vision (Week_2)

In week 1 you learned all about how Machine Learning and Deep Learning is a new programming paradigm. This week you’re going to take that to the next level by beginning to solve problems of computer vision with just a few lines of code! Check out this conversation between Laurence and Andrew where they discuss it and introduce you to Computer Vision!

### Enhancing Vision with Convolutional Neural Networks (Week_3)

In week 2 you saw a basic Neural Network for Computer Vision. It did the job nicely, but it was a little naive in its approach. This week we’ll see how to make it better, as discussed by Laurence and Andrew here.

### Using Real-World Images (Week_4)

Last week you saw how to improve the results from your deep neural network using convolutions. It was a good start, but the data you used was very basic. What happens when your images are larger, or if the features aren’t always in the same place? Andrew and Laurence discuss this to prepare you for what you’ll learn this week: handling complex images!

## Course 2 : Convolutional Neural Networks in TensorFlow
---
### Exploring a Larger Dataset (Week_1)

In this course you'll go deeper into using ConvNets will real-world data, and learn about techniques that you can use to improve your ConvNet performance, particularly when doing image classification! In Week 1, this week, you'll get started by looking at a much larger dataset than you've been using thus far: The Cats and Dogs dataset which had been a Kaggle Challenge in image classification!

### Augmentation : a Technique to Avoid Overfitting (Week_2)

You've heard the term overfitting a number of times to this point. Overfitting is simply the concept of being over specialized in training -- namely that your model is very good at classifying what it is trained for, but not so good at classifying things that it hasn't seen. In order to generalize your model more effectively, you will of course need a greater breadth of samples to train it on. That's not always possible, but a nice potential shortcut to this is Image Augmentation, where you tweak the training set to potentially increase the diversity of subjects it covers. You'll learn all about that this week!

### Transfer Learning (Week_3)

Building models for yourself is great, and can be very powerful. But, as you've seen, you can be limited by the data you have on hand. Not everybody has access to massive datasets or the compute power that's needed to train them effectively. Transfer learning can help solve this -- where people with models trained on large datasets train them, so that you can either use them directly, or, you can use the features that they have learned and apply them to your scenario. This is Transfer learning, and you'll look into that this week!

### Multiclass Classifications (Week_4)

One more thing to do before we move off of ConvNets to the next module, and that's to go beyond binary classification. Each of the examples you've done so far involved classifying one thing or another -- horse or human, cat or dog. When moving beyond binary into Categorical classification there are some coding considerations you need to take into account. You'll look at them this week!


## Course 3: Natural Language Processing in TensorFlow
---
### Sentiment in Text (Week_1)

The first step in understanding sentiment in text, and in particular when training a neural network to do so is the tokenization of that text. This is the process of converting the text into numeric values, with a number representing a word or a character. This week you'll learn about the Tokenizer and pad_sequences APIs in TensorFlow and how they can be used to prepare and encode text and sentences to get them ready for training neural networks!

### Word Embeddings (Week_2)

Last week you saw how to use the Tokenizer to prepare your text to be used by a neural network by converting words into numeric tokens, and sequencing sentences from these tokens. This week you'll learn about Embeddings, where these tokens are mapped as vectors in a high dimension space. With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space. This will begin the process of training a neural network to udnerstand sentiment in text -- and you'll begin by looking at movie reviews, training a neural network on texts that are labelled 'positive' or 'negative' and determining which words in a sentence drive those meanings.

### Sequence Models (Week_3)

In the last couple of weeks you looked first at Tokenizing words to get numeric values from them, and then using Embeddings to group words of similar meaning depending on how they were labelled. This gave you a good, but rough, sentiment analysis -- words such as 'fun' and 'entertaining' might show up in a positive movie review, and 'boring' and 'dull' might show up in a negative one. But sentiment can also be determined by the sequence in which words appear. For example, you could have 'not fun', which of course is the opposite of 'fun'. This week you'll start digging into a variety of model formats that are used in training models to understand context in sequence!

### Sequence Models and Literature (Week_4)

Taking everything that you've learned in training a neural network based on NLP, we thought it might be a bit of fun to turn the tables away from classification and use your knowledge for prediction. Given a body of words, you could conceivably predict the word most likely to follow a given word or phrase, and once you've done that, to do it again, and again. With that in mind, this week you'll build a poetry generator. It's trained with the lyrics from traditional Irish songs, and can be used to produce beautiful-sounding verse of it's own!

## Course 4: Sequences, Time Series, and Prediction
---
### Sequences and Prediction (Week_1)

In this course we'll take a look at some of the unique considerations involved when handling sequential time series data -- where values change over time, like the temperature on a particular day, or the number of visitors to your web site. We'll discuss various methodologies for predicting future values in these time series, building on what you've learned in previous courses!

### Deep Neural Networks for Time Series (Week_2)

Having explored time series and some of the common attributes of time series such as trend and seasonality, and then having used statistical methods for projection, let's now begin to teach neural networks to recognize and predict on time series!

### Recurrent Neural Networks for Time Series (Week_3)

Recurrent Neural networks and Long Short Term Memory networks are really useful to classify and predict on sequential data. This week we'll explore using them with time series...

### Real-world Time Series Data (Week_4)

On top of DNNs and RNNs, let's also add convolutions, and then put it all together using a real-world data series -- one which measures sunspot activity over hundreds of years, and see if we can predict using it.



