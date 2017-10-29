# Check metrics using trained weight files

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras import backend as K
import cPickle
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
epochs = 10

f = open('../data.pkl', 'rb')
x = cPickle.load(f)
f.close()
f = open('../data_age.pkl', 'rb')
y = cPickle.load(f)
f.close()
x = np.asarray(x, dtype=np.float32)
y = np.asarray(y)
x /= 255.
x_final = []
y_final = []
random_no = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
for i in random_no:
    x_final.append(x[i,:,:,:])
    y_final.append(y[i])
x_final = np.asarray(x_final)
y_final = np.asarray(y_final)

k = 1000 # Decides split count
x_test = x_final[:k,:,:,:]
y_test = y_final[:k]
x_valid = x_final[k:2*k,:,:,:]
y_valid = y_final[k:2*k]
x_train = x_final[2*k:,:,:,:]
y_train = y_final[2*k:]
print 'x_train shape:'+ str(x_train.shape)
print 'y_train shape:'+ str(y_train.shape)
print 'x_valid shape:'+ str(x_valid.shape)
print 'y_valid shape:'+ str(y_valid.shape)
print 'x_test shape:'+ str(x_test.shape)
print 'y_test shape:'+ str(y_test.shape)

base_model =  InceptionV3(weights='imagenet', include_top=False)
input = Input(shape=(224,224,3))
output_vgg16 = base_model(input)
x = Flatten()(output_vgg16)
x = Dense(512, activation='relu')(x)
predictions = Dense(1)(x)

model = Model(inputs=input, outputs=predictions)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
model.load_weights('model.h5')

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])


