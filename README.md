# Grammatical-error-detection
This model is built to find whether a given sentence is grammatically correct or not. This model is based on LSTMs and deep neural network layers. The model predicts
the probability of correctness of the sentence and, depending on the threshold limit set gives the answer accordingly. The model gave an accuracy of 80% on the test set.
This work was done is association with the Keral Infrastructure for Technology and Education (KITE). The next step is to build a model that not only classifies models but also corrects the incorrect sentences.

## Libraries used:
- Tensorflow
- Keras
- Numpy

## Datasets:

● NAIST Lang-8 Learner Corpora:
This corpus contains English learners’ texts extracted from the Lang-8 website (a
language exchange social networking website geared towards language learners). It has
100,051 English entries written by 29,012 active users. We used around 30,000 of these
sentences in our dataset.

Link: https://sites.google.com/site/naistlang8corpora/

● GLUE/CoLA:
The Corpus of Linguistic Acceptability (CoLA) in its full form consists of 10657
sentences from 23 linguistics publications, expertly annotated for acceptability
(grammaticality) by their original authors.
We combined the two datasets in order to bring some diversity into our dataset, as
grammatical error detection requires a vast number of sentences with varying levels of
incorrectness. The dataset was then divided into train, development, and test sets in the
ratio of 80:10:10.

Link: https://nyu-mll.github.io/CoLA/

## The model is built using six layers as follows:-

1. Embedding : The first layer of the model consists of the embedding layer with
pre-trained weights. This is known as Transfer Learning. This significantly reduced the
number of parameters that the model had to train and hence the model required less time
and processing power to learn. Here we chose the GloVe (Global Vector) word
embeddings.

2. RNN (LSTM Layer): As mentioned earlier, LSTMs are the best fit for our application.
To decide the number of layers of LSTM, we trained different models that had 32, 64,
and 128 layers of LSTM. Out of these three, 64 layer model performed the best and hence
it was chosen.

3. Dropout: Dropout is a regularization method that approximates training a large number
of neural networks with different architectures in parallel. Simply put, the dropout layer
chooses to ignore some layers at random so that they will not be considered for the final
output. This decreases the chances of the neural network relying on a few nodes for the
final output.

4. Batch Normalization: The batch normalization layer is used to normalize the input layer
by re-centering and re-scaling it. It improves the speed, performance, and stability of the
neural network.

5. Dense layer (ReLU activation): It is the regular deeply connected neural network layer.
Here we chose the ReLU activation function as it is the most common choice when it
comes to the activation function.

6. Dense layer (softmax activation): This is another layer of regular neural network with
the softmax activation layer. The softmax activation function forces the values of output
neurons to take values between zero and one, so they can represent probability scores.

Description:The model goes to learn and remember from previous data nodes as we are using LSTM
and then detects whether the sentence contains grammatical error or not
