import numpy as np
import math
import re
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import model
import torch
from bs4 import BeautifulSoup


def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # PREPROCESSING --------------------------------------------------------------------------------------------------
    # LOAD FILES
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    train_data = pd.read_csv(
        "data/training.1600000.processed.noemoticon.csv",
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )
    test_data = pd.read_csv(
        "data/testdata.manual.2009.06.14.csv",
        header=None,
        names=cols,
        engine="python",
        encoding="latin1"
    )

    # CLEANING DATA
    train_data.drop(["id", "date", "query", "user"], axis=1, inplace=True)
    data_clean = [clean_tweet(tweet) for tweet in train_data.text]
    data_labels = train_data.sentiment.values
    data_labels[data_labels == 4] = 1
    set(data_labels)

    # TOKENIZATION
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(data_clean, target_vocab_size=2 ** 16)
    data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

    # PADDING: add 0s to end of sentence if shorter than MAX_LEN
    MAX_LEN = max([len(sentence) for sentence in data_inputs])
    data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs, value=0, padding="post", maxlen=MAX_LEN)

    # SPLIT DATA INTO TRAINING/TEST SET
    test_idx = np.random.randint(0, 800000, 8000)
    test_idx = np.concatenate((test_idx, test_idx + 800000))
    test_inputs = data_inputs[test_idx]
    test_labels = data_labels[test_idx]
    train_inputs = np.delete(data_inputs, test_idx, axis=0)
    train_labels = np.delete(data_labels, test_idx)

    # TRAIN AND EVALUATE MODEL----------------------------------------------------------------------------------------
    # CONFIG PARAMETERS
    VOCAB_SIZE = tokenizer.vocab_size
    EMB_DIM = 200
    NB_FILTERS = 100
    FFN_UNITS = 256
    NB_CLASSES = len(set(train_labels))
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    NB_EPOCHS = 5

    # TRAINING
    Dcnn = model.DCNN(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, nb_filters=NB_FILTERS, FFN_units=FFN_UNITS,
                      nb_classes=NB_CLASSES, dropout_rate=DROPOUT_RATE)
    Dcnn = Dcnn.to(device)
    if NB_CLASSES == 2:
        # binary classification
        Dcnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    else:
        # multiclass classification
        Dcnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
    # training checkpoint loader
    checkpoint_path = "."
    ckpt = tf.train.Checkpoint(Dcnn=Dcnn)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    # fit and save model
    Dcnn.fit(train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=NB_EPOCHS)
    ckpt_manager.save()

    # EVALUATION
    results = Dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
    print(results)




















