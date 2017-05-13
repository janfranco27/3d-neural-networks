import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt


class MLP:
    def __init__(self,
                 input_units,
                 output_units,
                 hidden_layers=None,
                 activations=None,
                 from_file=None):
        '''
        Init a multilayer perceptron neural network

        Parameters
        ----------
        input_units (int): Number of neurons in input layer
        output_units (int): Number of neurons in output layer
        hidden_units (list or tuple): List indicating the number of neurons in
            ith hidden layer
        '''
        if not from_file:
            self.model = Sequential()

            for i, hidden in enumerate(hidden_layers):
                self.model.add(
                    Dense(
                        output_dim=hidden, input_dim=input_units,
                        activation=activations[i]
                    )
                )
                input_units = hidden

            self.model.add(
                Dense(
                    output_dim=output_units, input_dim=input_units,
                    activation=activations[-1]
                )
            )

        else:
            self.model = self.read_model(from_file)

        self.compile_model()

    def compile_model(self, loss_function='categorical_crossentropy',
                      optimizer_method='adam'):
        self.loss_function = loss_function
        self.optimizer_method = optimizer_method
        self.model.compile(
            loss=loss_function, optimizer=optimizer_method,
            metrics=['accuracy'])

    def train(self, train_x, train_y, validation_x, validation_y,
              file=None, epochs=5, batch_size=32):
        #callback = ModelCheckpoint(
        #    file, monitor='val_loss', verbose=0, save_best_only=True,
        #    save_weights_only=False, mode='auto', period=1)

        self.history = self.model.fit(
            train_x, train_y, nb_epoch=epochs, batch_size=batch_size,
            validation_data=(validation_x, validation_y))
        scores = self.model.evaluate(validation_x, validation_y)
        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1] * 100))
        return scores

    def plot(self, plot_dir, filename):
        # summarize history for loss
        plt.figure()
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(os.path.join(plot_dir, filename))

    def get_score(self, x, y):
        return self.model.evaluate(x, y)

    def save_model(self, filename):
        model_json = self.model.to_json()
        with open('{0}.json'.format(filename), 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('{0}.h5'.format(filename))
        print('Saved model to disk')

    def read_model(self, filename):
        json_file = open('{0}.json'.format(filename), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights('{0}.h5'.format(filename))
        print("Loaded model from disk")
        return loaded_model