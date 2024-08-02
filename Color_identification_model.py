import tensorflow as tf
import numpy as np

# Define colors and labels
color_labels = ['white', 'yellow', 'blue', 'red', 'orange', 'green']

# Map for color values
color_map = {
    'white': [255, 255, 255],
    'yellow': [255, 255, 0],
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'orange': [255, 165, 0]
}

# Prepare training data
x_train = np.array([color_map[color] for color in color_labels], dtype=np.float32)
y_train = np.array([color_labels.index(color) for color in color_labels], dtype=np.int32)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(color_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100)


model.save('color_model.keras')


loaded_model = tf.keras.models.load_model('color_model.keras')


def predict_color(rgb_values):
    rgb_values = np.array(rgb_values, dtype=np.float32).reshape(1, -1)
    predictions = loaded_model.predict(rgb_values)
    predicted_index = np.argmax(predictions)
    return color_labels[predicted_index]

#testing the model 

test_colors=[
    [255,255,255],
    [255,255,0],
    [0,0,255],
    
    [255,0,0],
    [255,165,0],
    [0,255,0],
]

for color in test_colors:
     print(f"RGB values {color} are classidies as: {predict_color(color)}")
