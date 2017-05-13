from keras.models import Sequential
from keras.layers import (
    Dense, Convolution2D, MaxPooling2D, Flatten, Activation, Dropout)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from constants import DESCRIPTOR_WIDTH, DESCRIPTOR_HEIGHT, OUTPUT_UNITS

'''
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# reshape data

train_x_temp = train_x.reshape(-1, 100, 100, 1)
val_x_temp = val_x.reshape(-1, 100, 100, 1)

# define vars
input_shape = (100*100,)
input_reshape = (100, 100, 1)

conv_num_filters = 5
conv_filter_size = 10

pool_size = (3, 3)

hidden_num_units = 1000
output_num_units = 3

epochs = 5
batch_size = 10


model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(10, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),
 Flatten(),
 Dense(output_dim=hidden_num_units, activation='relu'),
 Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_conv = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x_temp, val_y))
'''

class CNN:
    def __init__(self, descriptor_width=DESCRIPTOR_WIDTH,
                 descriptor_height=DESCRIPTOR_HEIGHT):
        '''
        Init a multilayer perceptron neural network

        Parameters
        ----------
        input_units (int): Number of neurons in input layer
        output_units (int): Number of neurons in output layer
        hidden_units (list or tuple): List indicating the number of neurons in
            ith hidden layer
        '''
        self.descriptor_width = descriptor_width
        self.descriptor_height = descriptor_height
        #(batch_size, rows, cols, channels)
        input_shape = (descriptor_width, descriptor_height, 1)

        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel sizeb
        nb_conv = 5

        self.model = Sequential()

        self.model.add(Convolution2D(32, 5, 5, input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Convolution2D(32, 5, 5))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1500))
        self.model.add(Activation('relu'))
        self.model.add(Dense(600))
        self.model.add(Activation('relu'))
        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

        self.compile_model()

    def compile_model(self, loss_function='categorical_crossentropy',
                      optimizer_method='adam'):

        self.loss_function = loss_function
        self.optimizer_method = optimizer_method

        self.optimizer_method = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss=loss_function, optimizer=optimizer_method,
            metrics=['accuracy', 'precision', 'recall'])

    def train(self, train_x, train_y, validation_x, validation_y,
              epochs=5, batch_size=32):
        self.trained_model = self.model.fit(
            train_x, train_y, nb_epoch=epochs, batch_size=batch_size,
            validation_data=(validation_x, validation_y))

        scores = self.model.evaluate(validation_x, validation_y)
        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1] * 100))
        #print '-------------'
        #print validation_y
        #ss = self.model.predict_proba(validation_x)
        #print '-------------'
        #import operator
        #a = []
        #for e in ss:
        #    index, value = max(enumerate(e), key=operator.itemgetter(1))
        #    m = ([1 if x == value else 0 for x in e])
        #    print m
        return scores


    def train_generator(self, method, path, train_generator, eval_generator, samples_per_epoch=80, epochs=50):
        self.trained_model = self.model.fit_generator(
            train_generator(method, path, self.descriptor_width, self.descriptor_height, 4),
            samples_per_epoch=12,
            nb_epoch=epochs,
            validation_data=eval_generator(method, path, self.descriptor_width,
                                           self.descriptor_height),
            nb_val_samples=54
        )

        scores = self.model.evaluate_generator(
            eval_generator(method, path, self.descriptor_width, self.descriptor_height),
            val_samples=54)

        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1] * 100))
        return scores



'''
    def train_generator(self, method, path, train_generator, eval_generator, steps_per_epoch=80, epochs=50):
        self.trained_model = self.model.fit_generator(
            train_generator(method, path, self.descriptor_width, self.descriptor_height),
            samples_per_epoch=steps_per_epoch,
            nb_epoch=epochs,
            validation_data=eval_generator(method, path, self.descriptor_width, self.descriptor_height),
            nb_val_samples=54)

        scores = self.model.evaluate_generator(
            eval_generator(method, path, self.descriptor_width, self.descriptor_height),
            val_samples=54)

        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1] * 100))
        return scores

'''