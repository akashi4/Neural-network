import numpy as np

# Define the input data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The desired output data
outputs = np.array([[0], [1], [1], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


# weights initialization
weights_input_hidden = np.random.uniform(size=(2, 2))
weights_output_hidden = np.random.uniform(size=(2, 1))

for epoch in range(1000):
    # for inputs, outputs in zip(inputs, outputs):
    #     inputs = inputs.reshape(1, -1)
    #     outputs = outputs.reshape(1, -1)
    # forward propagation
    hidden_layer_output = np.dot(inputs, weights_input_hidden)
    hidden_layer_activation = sigmoid(hidden_layer_output)

    final_output = sigmoid(np.dot(hidden_layer_activation, weights_output_hidden))

    # error calculation
    error = outputs - final_output

    # backpropagation
    d_predicted_output = error * derivative_sigmoid(final_output)
    error_hidden_layer = d_predicted_output.dot(weights_output_hidden.T)
    d_hidden_layer = error_hidden_layer * derivative_sigmoid(hidden_layer_activation)

    # updating weights
    weights_output_hidden += hidden_layer_activation.T.dot(d_predicted_output) * 0.1
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * 0.1

    # training the output for every input combination
    # if epoch % 100 == 0:
    print("Output after {} epochs: \n{}".format(epoch, final_output))
