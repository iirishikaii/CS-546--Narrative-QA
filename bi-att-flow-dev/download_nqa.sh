#!/usr/bin/env bash

DATA_DIR=$HOME/Documents/cs546

# Download GloVe
#GLOVE_DIR=$DATA_DIR/glove
#mkdir $GLOVE_DIR
#wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
#unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download NLTK (for tokenizer)
# Make sure that nltk is installed!
python3 -m nltk.downloader -d $HOME/Documents/cs546/nltk_data punkt
