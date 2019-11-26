import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


train_labels=[]
train_samples=[]


for i in range(1000):
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older=randint(64,100)
    train_samples.append(random_older)
    train_labels.append(1)


for i in range(50):
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older=randint(64,100)
    train_samples.append(random_older)
    train_labels.append(0)



train_labels=np.array(train_labels)
train_samples=np.array(train_samples)

print(train_samples)
print(train_labels)


#to make keras train them more easily, we use preprocessing to scale all the datas down
#to sacle them down between 0 and 1


scaler=MinMaxScaler(feature_range=(0,1))
#(-1,1) is just a technical format
scale_train_samples=scaler.fit_transform((train_samples).reshape(-1,1))

print(scale_train_samples)



# we have created the fake data for now, now we need to do some sample sequential data with keras.


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

#
# model.summary()
