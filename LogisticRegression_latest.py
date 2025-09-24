import tensorflow as tf
import numpy as np
learning_rate = 0.5
data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1], [14,1]]
x_data = [x_row[0] for x_row in data]
y_data = tf.constant([y_row[1] for y_row in data], dtype=tf.float64)

a= tf.Variable(tf.random.normal([1], dtype=tf.float64, seed=0))
b= tf.Variable(tf.random.normal([1], dtype=tf.float64, seed=0))




gradient_decent = tf.keras.optimizers.SGD(learning_rate=learning_rate)

for i in range(60001):
    with tf.GradientTape() as tape:
        # y= 1/(1+tf.exp(a*x_data+b))
        # loss = tf.reduce_mean(-np.array(y_data) * tf.math.log(y) + (1-np.array(y_data)) * tf.math.log(1-y))
        # y = a*x_data+b
        # loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_data, y, from_logits=True))
        y_pred = 1 / (1 + tf.exp((a * x_data + b)))
        
        # 책의 손실 함수 수식을 그대로 구현
        loss = -tf.reduce_mean((tf.constant(y_data) * tf.math.log(y_pred + 1e-10) + (1-tf.constant(y_data)) * tf.math.log(1-y_pred + 1e-10)))
    gradients = tape.gradient(loss, [a, b])
    gradient_decent.apply_gradients(zip(gradients, [a, b]))
    if i % 1000 == 0:
        print(f"Epoch: {i:3d} | RMSE: {loss.numpy():.4f} | a: {a.numpy()[0]:.4f} | b: {b.numpy()[0]:.4f}")