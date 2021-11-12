import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"})

# Magic


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# If using pretrained word2vec, input the weights as well and makd trainable=False
# ```model.add(Embedding(max_features, 128, mask_zero=True, weights=[wgt_mat_np], trainable=False))```
model.add(Embedding(max_features, 128, mask_zero=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=15,
#           validation_data=(x_test, y_test))

model.fit(x_train, y_train,  validation_data=(x_test, y_test), batch_size=batch_size,
          epochs=15,
          callbacks=[WandbCallback()])

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

