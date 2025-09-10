import os
import cv2
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# Categories (adjust to your dataset)
categories = ['Irregular', 'Spiral', 'Elliptical', 'Lenticular', 'Peculiar']
lw = 150  # image width/height

# Generic dataset path (adjust to your local environment)
db_path = '/path/to/dataset'

# =======================
# CREATE TRAINING DATA
# =======================
training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(db_path, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (lw, lw))
                training_data.append([new_array, class_num])
            except Exception:
                pass

create_training_data()

# Separate features and labels
x, y = [], []
for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, lw, lw) / 255.0  # normalization
y = np.array(y)

print(x.shape, y.shape)

# Save arrays
save('x.npy', x)
save('y.npy', y)

# =======================
# MODEL DEFINITION
# =======================
x = load('x.npy')
y = load('y.npy')

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(lw, lw)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(categories))  # output layer matches number of classes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
epochs = 50
history = model.fit(x, y, epochs=epochs)
print(history)

# Save model
model.save('Galaxy_class.model')

# =======================
# VISUALIZE TRAINING RESULTS
# =======================
acc = history.history['accuracy']
loss = history.history['loss']
df = pd.DataFrame(list(zip(acc, loss)), columns=['Accuracy', 'Loss'])
df.to_csv('acc_loss_df.csv', index=False)
print(df)

# =======================
# TESTING THE MODEL
# =======================
test_data = []

def create_test_data():
    for category in categories:
        path = os.path.join(db_path, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (lw, lw))
                test_data.append([new_array, class_num])
            except Exception:
                pass

create_test_data()

x_test, y_test = [], []
for features, label in test_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, lw, lw) / 255.0
y_test = np.array(y_test)

save('x_test.npy', x_test)
save('y_test.npy', y_test)

X = load('x_test.npy')
Y = load('y_test.npy')

model = tf.keras.models.load_model('Galaxy_class.model')

test_loss, test_acc = model.evaluate(X, Y)
print('\nTest accuracy:', test_acc * 100, '%')

# =======================
# PREDICTIONS
# =======================
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(categories[predicted_label],
                                         100*np.max(predictions_array),
                                         categories[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(len(categories)))
    plt.yticks([])
    thisplot = plt.bar(range(len(categories)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows, num_cols = 5, 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

ls = 43
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i+ls, predictions[i+ls], Y, X)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i+ls, predictions[i+ls], Y)
plt.tight_layout()
plt.show()

# =======================
# CONFUSION MATRIX
# =======================
ypred = np.argmax(predictions, axis=1)
mat = confusion_matrix(Y, ypred)
plot_confusion_matrix(conf_mat=mat, class_names=categories, figsize=(8,8))
plt.show()

# =======================
# PLOT ACCURACY & LOSS
# =======================
vec = np.linspace(0, epochs, epochs)
fig = go.Figure()
fig.add_trace(go.Scatter(x=vec, y=acc, name='accuracy'))
fig.add_trace(go.Scatter(x=vec, y=loss, name='loss'))
fig.update_layout(title='Model analysis',
                  xaxis_title='Epoch',
                  yaxis_title='Loss/Accuracy',
                  font_size=14, width=1000, height=600)
fig.show()
