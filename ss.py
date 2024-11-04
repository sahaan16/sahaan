import keras 
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
x=np.random.rand(100,1) #dataset
#print(x)
#print(len(x)) 
y = 2*x+0.05
#print(y)
#architecture
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(x,y,epochs=500)

a=model.predict(x)
print(a)
import matplotlib.pyplot as plt
plt.scatter(x,y,label='original data')
plt.plot(x,a,label='predicted data')
plt.show()