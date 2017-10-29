from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras import backend as K
import cPickle
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
epochs = 30

# Load data
print('...loading training data')
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

# Shuffle images and split into train, validation and test sets
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

# Using InceptionV3 with pretrained weights from Imagenet 
base_model =  InceptionV3(weights='imagenet', include_top=False)
input = Input(shape=(224,224,3))
output_vgg16 = base_model(input)
x = Flatten()(output_vgg16)
x = Dense(512, activation='relu')(x)
predictions = Dense(1)(x)

model = Model(inputs=input, outputs=predictions)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])

# Save weights after every epoch
checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_weights_only=True,
        period=1)
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid,y_valid), callbacks = [checkpoint])

model.save_weights("model.h5")

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test MAE:', score[1])

with open('history.pkl', 'wb') as f:
	cPickle.dump(history.history, f)
f.close()


