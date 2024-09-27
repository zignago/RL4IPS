import tensorflow as tf
from tensorflow.keras import regularizers, layers

def create_q_model(state_size, num_actions, learning_rate):
    inputs = layers.Input(shape=(state_size,))
    layer1 = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    layer2 = layers.Dense(256, activation="relu")(layer1)
    layer3 = layers.Dense(128, activation="relu")(layer2)
    action = layers.Dense(num_actions, activation="linear")(layer3)
    
    model = tf.keras.Model(inputs=inputs, outputs=action)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return model
