# Natural-Language-Processing-in-TensorFlow
Course 3 of the Coursera Professional Certificate : DeepLearning.AI-Tensorflow-Developer-Professional-Certificate

TensorFlow Datasets (TFDS) : https://www.tensorflow.org/datasets

TFDS Catalog : https://github.com/tensorflow/datasets/tree/master/docs/catalog

## Content of the notebook files :- 
### File: c3_week1_lab_1_2_tokenizer_basics.ipynb ---
Tokenizing natural language text into sequence of numbers/values/tokens per word so that the computer can work with them.

### File: c3_week1_lab_3_sarcasm_detection.ipynb ---
Only upto tokenizing the kaggle-Sarcasm-Detection dataset into sequence of numbers/tokens and creating the word_index->{key=word: value=token}.

### File: c3_week2_lab_1_imdb_reviews*.ipynb ---
> 1) tokenizing the imdb_reviews (plain_text) data from TensorFlow datasets to create the word_index
2) creating an NLP classifier whose first layer is non-trainable Embedding layer that converts the words/tokens into vectors of fixed dimensions
3) then these vectors are flattened and inserted into the trainable layers
4) the vectors point to either pole (as binary classification) based on the labels

### File: c3_week2_lab_2_sarcasm_classifier*.ipynb ---
This is the exact COURSERA version of code - so no attempt of optimzation has been made.
1) importing the kaggle-Sarcasm-Detection dataset(contains captions and is_sarcastic(0/1) as the label) json and tokenizing the text to create the word_index
2) creating an NLP classifier whose first layer is non-trainable Embedding layer that converts the words/tokens into vectors of fixed dimensions
3) then these vectors are flattened and inserted into the trainable layers
4) the vectors point to either pole (as binary classification whether sarcastic or not) based on the labels
5) model.predict on custom text(caption) will output a probability towards the caption being Sarcastic as the model's last layer had a single neuron with "sigmoid" activation function which is S-shaped and thus only used to determine binary classification results

### File: c3_week2_lab_3_imdb_subwords.ipynb ---
This is the exact COURSERA version of code.
> Subwords are fragments of whole words; for eg: the three subwords (Ten, sor, Flow) are actually parts of the word "TensorFlow"; and subwords are case,punctuation,etc.-sensitive.
> Here, tokenization has already been done using the subwords and the word_index is already available; so we can use TensorFlow>Tokenizer>encode to convert a text into sequence of tokens, based on the word_index->{key=subword: value=token} AND use TensorFlow>Tokenizer>decode to convert a sequence of tokens into text, based on the word_index.
> Now, only training and testing the model to predict the label for a particular text, based on what subwords are present in the text doesn't really work well because only the presence of "Ten" might be confused with a completely different word, the presence of "sor" won't indicate anything because it is meaningless.
> These subwords will work well, only if we can track the sequence of the subwords to form meaningful words like "Ten-sor-Flow", and for that sequence tracking - we need Recurrent Neural Network (RNN) which we'll be using in future (not in this code).
