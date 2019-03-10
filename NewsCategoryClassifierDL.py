# predefined parameters
MAX_SEQUENCE_LENGTH = 50
GLOVE_DIR = 'data/Glove/glove.6B'
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# loading dataset
import json
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = []
with open('data/NewsCategoriesDataset/News_Category_Dataset_v2.json', 'r') as datafile:
    for line in datafile:
        dataset.append(json.loads(line))


categories = [x['category'] for x in dataset]
data = [x['short_description']  + ' ' + x['headline'] for x in dataset]

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
tokenizer.fit_on_texts(data) 
data_seqs = tokenizer.texts_to_sequences(data)
data_index = tokenizer.word_index
data_seqs_padded = pad_sequences(data_seqs, maxlen=MAX_SEQUENCE_LENGTH)

# outputs categoriess to one hot vector
categories_ids = [get_category_id(x) for x in categories]
categories_vectors = to_categorical(np.asarray(categories_ids))

# Preparing dataset
# split the data into a training set and a validation set

x_train, x_test, y_train, y_test = train_test_split(data_seqs_padded,
													 categories_vectors,
													 test_size=TEST_SPLIT,
													 random_state=10)

# loading embeddings vectors
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs


embedding_matrix = np.zeros((len(data_index) + 1, EMBEDDING_DIM))
for word, i in (data_index).items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector


# Modeling Our Network
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(data_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

embedded_sequences = embedding_layer(sequence_input)

x = LSTM(32, return_sequences=True, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM))(embedded_sequences)
x = LSTM(32)(x)

preds = Dense(len(categories_dic), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())
history = model.fit(x_train, y_train, 
                    epochs=10,
                    batch_size=64,
                    validation_split=VALIDATION_SPLIT
                    )

score = model.evaluate(x_test, y_test)

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
