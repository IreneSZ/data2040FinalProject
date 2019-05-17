
# coding: utf-8

# In[ ]:


from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np
import pandas as pd
import random
import sys


# Read the entire file containing song lyrics
df = pd.read_csv('songdata.csv')
df.head()

df = df.dropna(subset = ['text'])
df.drop_duplicates(subset = ['text'], keep='first', inplace=True)

corpus = sys.argv[1] # first command line arg
text = df['text'].str.lower().str.replace('  \n  \n','  \n')
text = text.str.replace('  \n', ' \n ')
text = text.str.replace('\n\n', ' \n ')
text = text.str.replace('?','')
text = text.str.replace('!','')
text = text.str.replace(',', ' ,')
text = text.str.replace('  ', ' ')
text = text.str.replace('(', '')
text = text.str.replace(')', '')
print('Corpus length in characters:', len(text))


text_in_words = [word for word in text.str.split(' ') ]
print('Corpus length in words:', len(text_in_words))


text_all = ""
for i in range(0, len(text_in_words)):
    if(i % 10 == 0):
        for j in range(0,len(text_in_words[i])):
            text_all += str(text_in_words[i][j]) + " "
            
    
corpus = sys.argv[1] # first command line arg

print('Corpus length in characters:', len(text_all))

text_all_in_words = [w for w in text_all.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(text_all_in_words))


# Calculate word frequency
word_freq = {}
for word in text_all_in_words:
    word_freq[word] = word_freq.get(word, 0) + 1
    
    
MIN_WORD_FREQUENCY=450
ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_WORD_FREQUENCY:
        ignored_words.add(k)  
    
words = set(text_all_in_words)
print('Unique words before ignoring:', len(words))
print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
words = sorted(set(words) - ignored_words)
print('Unique words after ignoring:', len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))



# cut the text in semi-redundant sequences of SEQUENCE_LEN words
SEQUENCE_LEN = 10
STEP = 1
sentences = []
next_words = []
ignored = 0
for i in range(0, len(text_all_in_words) - SEQUENCE_LEN, STEP):
    # Only add sequences where no word is in ignored_words
    if len(set(text_all_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
        sentences.append(text_all_in_words[i: i + SEQUENCE_LEN])
        next_words.append(text_all_in_words[i + SEQUENCE_LEN])
    else:
        ignored = ignored+1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))



def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=10):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)
    
(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words, percentage_test=10)  
  
    
    
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)    
    
    
    
def on_epoch_end_orig(epoch, logs):
    print()
    if(epoch > 45):
        print('----- Generating text after Epoch: %d\n' % epoch)
        for diversity in [0.2, 0.5, 1.0]:
            
            print('----- Diversity:', diversity, ' -----')
            generated = ['life','is']
            sentence = generated
            for i in range(100):
                
                x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, word_indices[word]] = 1.
        
                preds = model_orig.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = indices_word[next_index]
                generated.append(next_word)
                sentence = sentence[1:] + [next_word]
                sys.stdout.write(next_word)
                sys.stdout.flush()
                
            print()

print_callback_orig = LambdaCallback(on_epoch_end=on_epoch_end_orig)    
    
    
    
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1

            index = index + 1
            if index == len(sentence_list):
                index = 0
        yield x, y    

        
batch_size = 128


print('Building the LSTM Model')

model_orig = Sequential()
model_orig.add(LSTM(128, input_shape=(SEQUENCE_LEN, len(words))))
model_orig.add(Dense(len(words)))
model_orig.add(Activation('softmax'))
model_orig.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
model_orig.summary()


history_orig = model_orig.fit(generator(sentences, next_words, batch_size=128),
                              steps_per_epoch=int(len(sentences)/batch_size) + 1,
                              epochs=50,
                              callbacks=[print_callback_orig],
                              validation_data=generator(sentences_test, next_words_test, batch_size), 
                              validation_steps=int(len(sentences_test)/batch_size) + 1)

print("LSTM Network Trained")



import matplotlib.pyplot as plt

loss = history_orig.history['loss']
val_loss = history_orig.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# diversity is used by the sample function, that will randomly select
# the next most probable character from the softmax output 

print('----- Generating text -----')
for diversity in [0.2, 0.5, 1.0]:
    
    print()
    sentence = ['life','is']
    original = " ".join(sentence)
    generated = sentence
    window = sentence
    finalText = ''
    print('----- Diversity:', diversity, ' -----\n')

    for i in range(50):
        x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
        for t, word in enumerate(window):
            x_pred[0, t, word_indices[word]] = 1.0
        
        preds = model_orig.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]
        finalText += " "+next_word
        window = window[1:] + [next_word]
  
    print(original + finalText)

print('----- Text generation complete! -----')



from tensorflow.keras import layers

model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LEN, len(words))))
model.add(Dropout(0.2))
model.add(layers.Flatten())

model.add(Dense(len(words)+50, activation = 'softmax'))
model.add(Dropout(0.2))

model.add(Dense(len(words), activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()


def on_epoch_end(epoch, logs):
    print()
    if(epoch > 95):
        print('----- Generating text after Epoch: %d\n' % epoch)
        for diversity in [0.2, 0.5, 1.0]:
            
            print('----- Diversity:', diversity, ' -----')
            generated = ['life','is']
            sentence = generated
            for i in range(100):
                
                x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, word_indices[word]] = 1.
        
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = indices_word[next_index]
                generated.append(next_word)
                sentence = sentence[1:] + [next_word]
                sys.stdout.write(next_word)
                sys.stdout.flush()
                
            print()
            
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5)
    
    
history = model.fit_generator(generator(sentences, next_words, batch_size=128),
                              steps_per_epoch=int(len(sentences)/batch_size) + 1,
                              epochs=100,
                              callbacks=[print_callback,early_stopping],
                              validation_data=generator(sentences_test, next_words_test, batch_size),              
                              validation_steps=int(len(sentences_test)/batch_size) + 1)    
    
    
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()    



# diversity is used by the sample function, that will randomly select
# the next most probable character from the softmax output 

print('----- Generating text -----')
for diversity in [0.2, 0.5, 1.0]:
    
    print()
    sentence = ['life','is']
    original = " ".join(sentence)
    generated = sentence
    window = sentence
    finalText = ''
    print('----- Diversity:', diversity, ' -----\n')

    for i in range(50):
        x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
        for t, word in enumerate(window):
            x_pred[0, t, word_indices[word]] = 1.0
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]
        finalText += " "+next_word
        window = window[1:] + [next_word]
  
    print(original + finalText)

print('----- Text generation complete! -----')



from tensorflow.keras import layers

model1 = Sequential()
model1.add(GRU(32, input_shape=(SEQUENCE_LEN, len(words))))
model1.add(Dropout(0.2))
model1.add(layers.Flatten())
model1.add(Dense(len(words)+50, activation = 'softmax'))
model1.add(Dropout(0.2))
model1.add(Dense(len(words), activation = 'softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model1.summary()



def on_epoch_end1(epoch, logs):
    print()
    if(epoch > 95):
        print('----- Generating text after Epoch: %d\n' % epoch)
        for diversity in [0.2, 0.5, 1.0]:
            
            print('----- Diversity:', diversity, ' -----')
            generated = ['life','is']
            sentence = generated
            for i in range(100):
                
                x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, word_indices[word]] = 1.
        
                preds = model1.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = indices_word[next_index]
                generated.append(next_word)
                sentence = sentence[1:] + [next_word]
                sys.stdout.write(next_word)
                sys.stdout.flush()
                
            print()
            
print_callback1 = LambdaCallback(on_epoch_end=on_epoch_end1)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5)



history1 = model1.fit_generator(generator(sentences, next_words, batch_size=128),
                              steps_per_epoch=int(len(sentences)/batch_size) + 1,
                              epochs=100,
                              callbacks=[print_callback1,early_stopping],
                              validation_data=generator(sentences_test, next_words_test, batch_size),              
                              validation_steps=int(len(sentences_test)/batch_size) + 1)



loss = history1.history['loss']
val_loss = history1.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




# diversity is used by the sample function, that will randomly select
# the next most probable character from the softmax output 

print('----- Generating text -----')
for diversity in [0.2, 0.5, 0.8, 1.0]:
    
    print()
    sentence = ['life','is']
    original = " ".join(sentence)
    generated = sentence
    window = sentence
    finalText = ''
    print('----- Diversity:', diversity, ' -----\n')

    for i in range(50):
        x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
        for t, word in enumerate(window):
            x_pred[0, t, word_indices[word]] = 1.0
        
        preds = model1.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]
        finalText += " "+next_word
        window = window[1:] + [next_word]
  
    print(original + finalText)

print('----- Text generation complete! -----')




# Data generator for fit and evaluate
def generator_embedding(sentence_list, next_words_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.bool)
        y = np.zeros((batch_size), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = next_words_list[index % len(sentence_list)]
            index = index + 1
        yield x, y
        
        
model2 = Sequential()
model2.add(Embedding(len(words), 64))
model2.add(Dropout(0.2))
model2.add(GRU(64))
model2.add(Dropout(0.2))
model2.add(Dense(len(words)))
model2.add(Activation('softmax'))
model2.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model2.summary()



def on_epoch_end2(epoch, logs):
    print()
    if(epoch > 45):
        print('----- Generating text after Epoch: %d\n' % epoch)
        for diversity in [0.2, 0.5, 0.8, 1.0]:
            
            print('----- Diversity:', diversity, ' -----')
            generated = ['life','is']
            sentence = generated
            for i in range(100):
                
                x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, word_indices[word]] = 1.
        
                preds = model2.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = indices_word[next_index]
                generated.append(next_word)
                sentence = sentence[1:] + [next_word]
                sys.stdout.write(next_word)
                sys.stdout.flush()
                
            print()
            
print_callback2 = LambdaCallback(on_epoch_end=on_epoch_end2)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)



history2 = model2.fit_generator(generator_embedding(sentences, next_words, batch_size=128),
                              steps_per_epoch=int(len(sentences)/batch_size) + 1,
                              epochs=100,
                              callbacks=[print_callback2,early_stopping],
                              validation_data=generator_embedding(sentences_test, next_words_test, batch_size),              
                              validation_steps=int(len(sentences_test)/batch_size) + 1)




loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# diversity is used by the sample function, that will randomly select
# the next most probable character from the softmax output 

print('----- Generating text -----')
for diversity in [0.2, 0.4,0.5,0.6,0.7, 0.8,0.9, 1.0]:
    
    print()
    sentence = ['i','love']
    original = " ".join(sentence)
    generated = sentence
    window = sentence
    finalText = ''
    print('----- Diversity:', diversity, ' -----\n')

    for i in range(50):
        x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
        for t, word in enumerate(window):
            x_pred[0, t, word_indices[word]] = 1.0
        
        preds = model_orig.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]
        finalText += " "+next_word
        window = window[1:] + [next_word]
  
    print(original + finalText)

print('----- Text generation complete! -----')




    

