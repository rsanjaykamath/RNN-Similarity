#!/usr/bin/env bash

cd data/embeddings
echo 'Processing GloVe'
echo '==========================================='

if [ -f glove.840B.300d.txt ]; then
    echo 'glove.840B.300d.txt already exists.. skipping glove.840B.300d.zip extraction..'
else
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
fi