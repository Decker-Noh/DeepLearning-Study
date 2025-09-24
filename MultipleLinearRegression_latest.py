import tensorflow as tf;

data = [[2, 0, 81], [4,4,93], [6,2,91], [8,3, 97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]

a1 = tf.Variable(tf.random.uniform(shape=[1], minval=0, maxval=10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random.uniform(shape=[1], minval=0, maxval=10, dtype=tf.float64, seed=0))
b =  tf.Variable(tf.random.uniform(shape=[1], minval=0, maxval=100, dtype=tf.float64, seed=0))


learning_rate = 0.1

gradient_decent = tf.keras.optimizers.SGD(learning_rate=learning_rate)

for i in range(2001):
    with tf.GradientTape() as tape:
        y = a1 * x1 + a2* x2 + b

        rmse = tf.sqrt(tf.reduce_mean(tf.square(y- y_data)))
    gradients = tape.gradient(rmse, [a1, a2, b])
    gradient_decent.apply_gradients(zip(gradients, [a1, a2, b]))
    if i % 100 == 0:
        print(f"Epoch: {i:3d} | RMSE: {rmse.numpy():.4f} | a1: {a1.numpy()[0]:.4f} | a2: {a2.numpy()[0]:.4f} | b: {b.numpy()[0]:.4f}")