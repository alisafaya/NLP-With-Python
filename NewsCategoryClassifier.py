# predefined parameters
MAX_HEADLINE_SEQUENCE_LENGTH = 15
MAX_DESCRIPTION_SEQUENCE_LENGTH = 50
GLOVE_DIR = 'data/Glove/glove.6B'
EMBEDDING_DIM = 100
USE_DESC_INSTEAD_OF_HEADLINES = True
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.3

# loading dataset
import json
import random
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.engine.input_layer import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import LSTM, TimeDistributed, Dropout, Activation
import matplotlib.pyplot as plt

dataset = []
with open('data/NewsCategoriesDataset/News_Category_Dataset_v2.json', 'r') as datafile:
    for line in datafile:
        dataset.append(json.loads(line))


categories = [x['category'] for x in dataset]
headlines = [x['headline'] for x in dataset]
descriptions = [x['short_description'] for x in dataset]


# Vectorizing and Labeling Output
categories_dic = {}

_id = 0
for x in categories:
    if x not in categories_dic.values():
        categories_dic[_id] = x
        _id += 1
        
def get_category_id(search_category):
    for _id, category in categories_dic.items():
        if category == search_category:
            return _id
    return -1


# Preprocessing headlines
tokenizer = Tokenizer()
tokenizer.fit_on_texts(headlines) 
headlines_seqs = tokenizer.texts_to_sequences(headlines)
headlines_index = tokenizer.word_index
headlines_seqs_padded = pad_sequences(headlines_seqs, maxlen=MAX_HEADLINE_SEQUENCE_LENGTH)

# Preprocessing descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions) 
descriptions_seqs = tokenizer.texts_to_sequences(descriptions)
descriptions_index = tokenizer.word_index
descriptions_seqs_padded = pad_sequences(descriptions_seqs, maxlen=MAX_DESCRIPTION_SEQUENCE_LENGTH)

# outputs categoriess to one hot vector
categories_ids = [get_category_id(x) for x in categories]
categories_vectors = to_categorical(np.asarray(categories_ids))

# Preparing dataset
# split the data into a training set and a validation set
indices = np.arange((descriptions_seqs_padded if USE_DESC_INSTEAD_OF_HEADLINES else headlines_seqs_padded).shape[0])
np.random.shuffle(indices)
data = (descriptions_seqs_padded if USE_DESC_INSTEAD_OF_HEADLINES else headlines_seqs_padded)[indices]
labels = categories_vectors[indices]

nb_test_samples = int(TEST_SPLIT * data.shape[0])

x_train = data[:-nb_test_samples]
y_train = labels[:-nb_test_samples]

x_test = data[-nb_test_samples:]
y_test = labels[-nb_test_samples:]

nb_validation_samples = int(VALIDATION_SPLIT * x_train.shape[0])
x_val = x_train[-nb_validation_samples:]
y_val = y_train[-nb_validation_samples:]

x_train = x_train[:-nb_validation_samples]
y_train = y_train[:-nb_validation_samples]

# loading embeddings vectors
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(descriptions_index if USE_DESC_INSTEAD_OF_HEADLINES else headlines_index) + 1, EMBEDDING_DIM))
for word, i in (descriptions_index if USE_DESC_INSTEAD_OF_HEADLINES else headlines_index).items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        

# Modeling Our Network
sequence_input = Input(shape=((MAX_DESCRIPTION_SEQUENCE_LENGTH if USE_DESC_INSTEAD_OF_HEADLINES else MAX_HEADLINE_SEQUENCE_LENGTH),), dtype='int32')

embedding_layer = Embedding(len(descriptions_index if USE_DESC_INSTEAD_OF_HEADLINES else headlines_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=(MAX_DESCRIPTION_SEQUENCE_LENGTH if USE_DESC_INSTEAD_OF_HEADLINES else MAX_HEADLINE_SEQUENCE_LENGTH),
                            trainable=False)

embedded_sequences = embedding_layer(sequence_input)

x = LSTM(256, return_sequences=True, input_shape=(50,100))(embedded_sequences)
x = LSTM(256)(x)

preds = Dense(len(categories_dic), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', 'categorical_accuracy'])
print(model.summary())
history = model.fit(x_train, y_train, 
                    epochs=3,
                    batch_size=512,
                    validation_data=(x_val, y_val)
                    )

score = model.evaluate(x_test, y_test,
                       batch_size=512, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
