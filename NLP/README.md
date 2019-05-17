## Word-level LSTM Lyrics Generator

### The second mini-project is an NLP project. The idea is to build an LSTM lyrics generator. 

### We use a dataset which collects lyrics for 57650 songs in English from LyricsFreak. The dataset can be found at https://www.kaggle.com/mousehead/songlyrics. It’s also available in our GitHub Repo and is named as ‘songdata.csv’. The complete code can be found in the file final2040_NLP.py. The notebook final2040_NLP.ipynb shows all outputs. We run the script on GPU. 

### The idea of making a lyrics generator is motivated by Francois Chollet’s keras text generation example: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py.

### Our project is based on another character-level LSTM Rap Lyric Generator, which is done by Karan Jaisingh. That project can be found on Karan Jaisingh’s GitHub repository: https://github.com/kjaisingh/rap-lyrics-generator. He used a Kaggle dataset with over 380,000 lyrics. Here is the link of his data set: https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics.

### Our project uses the character-level Rap Lyric Generator as a base model and tries to make some improvements. The main change is to implement a word-level LSTM text generator instead of a character-level one. When vectorizing words, we use one-hot encoding and word-embeddings. When doing data preprocessing and setting up data generators, we refer to https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb. We change the architecture of the model with dropout layers, and also try to improve the model by using early stopping callbacks, changing the optimizer and adding train-test split.

### The word-level LSTM Lyrics Generator requires the following to be installed on the system:
- Python 3
- Tensorflow 2.0.0
- Keras 2.2.4
- Numpy
- Pandas
