import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#load the dataset
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
# print(X_train.shape,X_test.shape)
# plt.imshow(X_train[5])
# plt.show()

#flatten image
X_train=X_train.reshape(X_train.shape[0],784)
X_test=X_test.reshape(X_test.shape[0],784)
# print(X_train.shape,X_test.shape)
# print(X_train[0])

# print(Y_train)
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)
# print(Y_train)

#normalization
# print(X_train.max())
X_train=X_train/255
# print(X_train.max())

# print(X_test)
X_test=X_test/255
# print(X_test)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

model=Sequential()
model.add(Dense(10, input_dim=784, activation='softmax'))
model.compile(optimizer=SGD(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# callbacks=ModelCheckpoint('my model.keras', monitor='val_loss', save_best_only=True,mode='min')
res=model.fit(X_train,Y_train, epochs=30, batch_size=32,validation_split=0.2)
model.evaluate(X_test,Y_test)

plt.plot(res.history['loss'],label='training loss')
plt.plot(res.history['val_loss'],label='validation loss')
plt.show()
