# Natural-Language-Processing-in-TensorFlow
Course 3 of Coursera Professional Certificate: DeepLearning.AI-Tensorflow-Developer-Professional-Certificate

# Links -

> <a href="https://www.tensorflow.org/datasets" target="_blank">TensorFlow Datasets (TFDS)</a>
> 
> <a href="https://github.com/tensorflow/datasets/tree/master/docs/catalog">TFDS Catalog</a>

<b>Kaggle links :</b>
> Dataset: <a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection">News Headlines Dataset for Sarcasm Detection</a>
> 
> Notebooks on Sarcasm Detection dataset:
> <a href="https://www.kaggle.com/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy/notebook#TRAINING-WORD2VEC-MODEL">Sarcasm Detection with GloVe/Word2Vec(83%Accuracy)</a>, <a href="https://www.kaggle.com/gauravduttakiit/sarcasm-detection-using-transformers">Sarcasm Detection using Transformers</a>

# Content of the Course -

<a href="#week-1">Week 1</a> &nbsp;&nbsp;&nbsp; <a href="#week-2">Week 2</a>

## Week-1

### File: c3_week1_lab_1_2_tokenizer_basics.ipynb ---
> Tokenizing natural language text into sequence of numbers/values/tokens per word so that the computer can work with them.

### File: c3_week1_lab_3_sarcasm_detection.ipynb ---
> Kaggle Dataset: <a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection">News Headlines Dataset for Sarcasm Detection</a>

> Only upto tokenizing the kaggle-Sarcasm-Detection dataset into sequence of numbers/tokens and creating the word_index->{key=word: value=token}.

## Week-2

### File: c3_week2_lab_1_imdb_reviews*.ipynb ---
> 1) tokenizing the <b><i>imdb_reviews(plain_text) from TensorFlow datasets</i></b> to create the word_index
> 2) creating an NLP classifier whose first layer is non-trainable Embedding layer that converts the words/tokens into vectors of fixed dimensions
> 3) then these vectors are flattened and inserted into the trainable layers
> 4) the vectors point to either pole (as binary classification) based on the labels

### File: c3_week2_lab_2_sarcasm_classifier*.ipynb ---
This is the exact COURSERA version of code - so no attempt of optimzation has been made.
> 1) importing the kaggle-Sarcasm-Detection dataset(news_headlines as the data and is_sarcastic(0/1) as the label) json and tokenizing the text to create the word_index
> 2) creating an NLP classifier whose first layer is non-trainable Embedding layer that converts the words/tokens into vectors of fixed dimensions
> 3) then these vectors are flattened and inserted into the trainable layers
> 4) the vectors point to either pole (as binary classification whether sarcastic or not) based on the labels
> 5) model.predict on custom text(caption) will output a probability towards the caption being Sarcastic as the model's last layer had a single neuron with "sigmoid" activation function which is S-shaped and thus only used to determine binary classification results

### File: c3_week2_lab_3_imdb_subwords.ipynb ---
This is the exact COURSERA version of code.
> "Taking one step back to move two steps forward"

> Subwords are fragments of whole words; for eg: the three subwords (Ten, sor, Flow) are actually parts of the word "TensorFlow"; and subwords are case,punctuation,etc.-sensitive.
>
> Here, tokenization has already been done using the subwords and the word_index is already available; so we can use TensorFlow>Tokenizer>encode to convert a text into sequence of tokens, based on the word_index->{key=subword: value=token} AND use TensorFlow>Tokenizer>decode to convert a sequence of tokens into text, based on the word_index.
> 
> Now, only training and testing the model to predict the label for a particular text, based on what subwords are present in the text doesn't really work well because only the presence of "Ten" might be confused with a completely different word, the presence of "sor" won't indicate anything because it is meaningless.
> 
> <b><i>These subwords will work well, only if we can track the sequence of the subwords to form meaningful words like "Ten-sor-Flow", and for that sequence tracking - we need Recurrent Neural Network (RNN) which we'll be using in future (not in this code).</i></b>

### File: c3_week2_optional_assignment_bbc_news_archive*.ipynb ---
This is the exact COURSERA version of code.
> Kaggle Dataset: <a href="https://www.kaggle.com/hgultekin/bbcnewsarchive">BBC News Archive</a>

> This is a Multi-Class Classification problem as there are 6 different types of labels / 6 categories of news.

> "Content of the News Article" is being used as the text data AND "Manually Labeled Categories of News" is being used as the labels/target values.

## Week-3

Till now, we were training and predicting texts/sentences based on what words are present in the sentence - like if a positive word(inferred from training using the labelled data) is encountered then the probability of the sentence being predicted as the positive class increases and same for the negative class and same in case of multi-class also.

But "My dog sat on the hat." and "The hat sat on my dog." have completely different meanings though they contain the same words - that's why the sequence of the words in the texts/sentences has a huge impact on the overall meaning of the sentence, which can only be Tracked/Implemented using special <b>SEQUENCE NEURAL NETWORK MODELS</b> like:- Recurrent Neural Network (RNN), Long Short Term Memory (LSTM), Gated Recurrent Unit (GRU).

<b>A simple Neural Network</b>
> ![image](https://user-images.githubusercontent.com/86871718/138137557-04ed311a-aca6-4e24-817c-673929f40c4d.png)
> ![image](https://user-images.githubusercontent.com/86871718/138137796-16508805-dc8f-402d-bf97-339796325ed3.png)

> Fibonacci Series Function (where the sequence of the numbers matters) :
> ![image](https://user-images.githubusercontent.com/86871718/138137825-a5539469-4945-4967-ba86-4e57fcfd21b0.png)
> ![image](https://user-images.githubusercontent.com/86871718/138137844-f4e6d51c-8ff2-42a2-9783-e9f8010614d9.png)

<b>Recurrent Neural Network (RNN)</b>

> So, to track the sequence of words/numbers, we have to store or feed in the previous output continuosly (just like <b>Sequential Circuit in Digital Electronics</b>) and this can be impelemented using RNN (they carry the meaning from one cell to the next) :
> ![image](https://user-images.githubusercontent.com/86871718/138137867-6ed97f73-7997-4850-b61b-2c4dfb77d729.png)
> ![image](https://user-images.githubusercontent.com/86871718/138137875-afc532bb-a043-408e-a94c-dee167bdb40d.png)

> Here, the predicted value in the blank depends on a keyword present in the sentence way before. As RNN only propagates each cell's output i.e. the current meaning, so the keyword 'Ireland' will be lost in case of RNN.
> ![image](https://user-images.githubusercontent.com/86871718/138138204-2125e5f4-c913-40b6-b917-e25463fb4a35.png)

<b>Long Short Term Memory (LSTM)</b>

> Along with the context/meaning being propagated like in RNN, "CELL STATE" is also passed/propagated in case of LSTM. CELL STATE keeps context from earlier tokens relevant in later ones so that "Gaelic" can be predicted from the context of the individual token of "Ireland" (earlier example) that would have been lost in case of just RNN.
> 
> Vanilla RNNs do not have a cell state. They only have hidden states and those hidden states serve as the memory for RNNs. Meanwhile, LSTM has both cell states and a hidden states. The cell state has the ability to remove or add information to the cell, regulated by "gates".

> <b>CELL STATE</b> can be either unidirectional or bidirectional.
> ![image](https://user-images.githubusercontent.com/86871718/138149705-24cd32d3-65cc-47bb-b84e-f95d983ad120.png)
> ![image](https://user-images.githubusercontent.com/86871718/138149735-198ef47f-c1f5-43c5-94e1-eb7c0b1162dd.png)

> <b>Using LSTM in NN model code :</b>
> 
> ![image](https://user-images.githubusercontent.com/86871718/138155511-019334ff-0e25-46eb-a9f4-5d9c3d1d6b01.png)
> Due to the keras layer being "bidirectional" the Cell State of LSTM can propagate in both forward and backward directions. Thus LSTM(64) produces a model with Output Shape (None, 128).
> ![image](https://user-images.githubusercontent.com/86871718/138155841-8546ade6-c588-4946-9140-0aaff11aa710.png)
>
>
> <b>LSTMs can be stacked</b> but then we have to put <b>return_sequences=True</b> for the previous LSTM layer whose output will feed into the input of the next LSTM layer, to ensure that the output of the previous LSTM layer match the input of the next LSTM layer.
> ![image](https://user-images.githubusercontent.com/86871718/138155978-8f5a70cd-616a-482f-9775-712e10032a97.png)

### File: c3_week3_lab_1_imdb_reviews_single_layer_LSTM*.ipynb
### & File: c3_week3_lab_2_imdb_reviews_multiple_layer_LSTM*.ipynb ---
<b>10 Epochs Training</b>
> The nosedive at the end of validation_accuracy for 1 layer LSTM represents overfitting. We can see that 2 layer LSTM accuracy curve over 10 epochs is much more smoother than 1 layer LSTM accuracy curve - and we have seen that jagginess can be a sign that the model needs improvement/optimization.
> ![image](https://user-images.githubusercontent.com/86871718/138558301-211b070d-a9ed-4e1a-a787-70388467d3f0.png)
> Similar results for loss curve.
> ![image](https://user-images.githubusercontent.com/86871718/138558459-6886bb47-5aad-477c-bf3f-5699d38fea53.png)

<b>50 Epochs Training</b>
> Even if the final accuracies of both 1 and 2 layer LSTMs are similar, the frequent dips(jagginess) in the 1 layer LSTM curve refer that the model is not optimized, whereas the 2 layer LSTM curve is much smoother. To consider - we have only used 8k subwords from the training set, so there might be a lot of OOV(out of vocabulary) words/tokens in the test set and thus not very high validation accuracy.
> ![image](https://user-images.githubusercontent.com/86871718/138558504-81dcc27f-071a-4a1d-aedc-d205c9337b04.png)
> Loss curve is smoothening epoch by epoch.
> ![image](https://user-images.githubusercontent.com/86871718/138558651-7eece94b-019b-459d-8e5a-e048f04e26c2.png)





