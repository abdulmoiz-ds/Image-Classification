import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255  # Scale pixel values to between 0 and 1
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)  # Convert labels to one-hot encoding
y_test = keras.utils.to_categorical(y_test, 10)

# Build the neural network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))  # Convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))  # Another convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flatten the output from the convolutional layers
model.add(Dense(128, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(Dense(10, activation='softmax'))  # Output layer with 10 nodes for the 10 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Function to stop the training early if the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make predictions on new images
predictions = model.predict(x_test[:10])
print('Predictions:', predictions)
