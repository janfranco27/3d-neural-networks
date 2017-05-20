from keras.models import Sequential
from keras.optimizers import SGD


class CNN:
    def __init__(self, descriptor_rows, descriptor_cols, channels):
        '''
        Init a multilayer perceptron neural network

        Parameters
        ----------
        input_units (int): Number of neurons in input layer
        output_units (int): Number of neurons in output layer
        hidden_units (list or tuple): List indicating the number of neurons in
            ith hidden layer
        '''
        self.descriptor_rows = descriptor_rows
        self.descriptor_cols = descriptor_cols
        #(batch_size, rows, cols, channels)
        self.input_shape = (descriptor_rows, descriptor_cols, channels)

        self.model = Sequential()

    def add_layer(self, layer):
        self.model.add(layer)

    def compile_model(self, loss_function='categorical_crossentropy',
                      optimizer_method='adam'):

        self.loss_function = loss_function
        self.optimizer_method = optimizer_method

        #self.optimizer_method = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss=loss_function, optimizer=optimizer_method,
            metrics=['accuracy', 'precision', 'recall'])

    def train_generator(self, dir, rows, cols, training_size,
                        validation_size, epochs, batch_size=32, channels=1,
                        method='hks'):

        from common.utils import train_with_generator, evaluate_with_generator

        self.history = self.model.fit_generator(
            train_with_generator(dir, rows, cols, training_size, batch_size,
                                 channels=channels, method=method),
            samples_per_epoch=training_size,
            nb_epoch=epochs,
            validation_data=evaluate_with_generator(dir, rows, cols, channels=channels,
                                                    method=method),
            nb_val_samples=validation_size
        )

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

    def predict(self, x_data):
        return self.model.predict(x_data)
'''
    def train_generator(self, method, path, train_generator, eval_generator, steps_per_epoch=80, epochs=50):
        self.trained_model = self.model.fit_generator(
            train_generator(method, path, self.descriptor_rows, self.descriptor_cols),
            samples_per_epoch=steps_per_epoch,
            nb_epoch=epochs,
            validation_data=eval_generator(method, path, self.descriptor_rows, self.descriptor_cols),
            nb_val_samples=54)

        scores = self.model.evaluate_generator(
            eval_generator(method, path, self.descriptor_rows, self.descriptor_cols),
            val_samples=54)

        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1] * 100))
        return scores

'''