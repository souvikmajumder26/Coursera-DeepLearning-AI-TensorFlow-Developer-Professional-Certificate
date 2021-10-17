# Natural-Language-Processing-in-TensorFlow
Course 3 of the Coursera Professional Certificate : DeepLearning.AI-Tensorflow-Developer-Professional-Certificate

TensorFlow Datasets (TFDS) : https://www.tensorflow.org/datasets

TFDS Catalog : https://github.com/tensorflow/datasets/tree/master/docs/catalog


File: c3_week1_lab_1_2_tokenizer_basics.ipynb --- tokenizing natural language data into sequence of numbers/values/tokens per word so that the computer can work with them.

File: c3_week1_lab_3_sarcasm_detection.ipynb --- only upto tokenizing the kaggle-Sarcasm-Detection dataset into sequence of numbers/tokens and creating the word index-{key=word: value=token}

File: c3_week2_lab_1_imdb_reviews*.ipynb --- 1) tokenizing the imdb_reviews (plain_text) data from TensorFlow datasets  2) creating an NLP classifier whose first layer is non-trainable Embedding layer that converts the words/tokens into vectors of fixed dimensions   3) then these vectors are flattened and inserted into the trainable layers   4) the vectors point to either pole (as binary classification) based on the labels
