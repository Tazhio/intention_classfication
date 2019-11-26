from keras.layers import SimpleRNN, Activation, Dense



if __name__=='__main__':
    X_train = X_train.reshape(-1, 28, 28) / 255.  # normalize
    X_test = X_test.reshape(-1, 28, 28) / 255.  # normalize
    