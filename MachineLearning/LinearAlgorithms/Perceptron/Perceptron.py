
# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation = activation + weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0        #   the perceptron receives the inputs of a sample x and combines
    #                                                   them with the weights w to compute the net input. The net input
    #                                                   is then passed on to the activation function ( 1.0 if activation >= 0.0 else 0.0  )
    #                                                   which generates a binary output -1 or +1, the predicted class label of the sample.
    #                                                   During the learning phase, this output is used to calculate the error
    #                                                   of the prediction and update the weights.


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]  #gets the number of columns in the row and sets it all to 0
    for epoch in range(n_epoch):
        print('Epoch--------------------' + str(epoch ))
        sum_error = 0
        for row in train:

            prediction = predict(row, weights)
            print("Expected=%d, Predicted=%d" % (row[-1], prediction))
            error = row[-1] - prediction                                       # derivative or delta
            print('>error=%.3f' % (error))

            weights[0] = weights[0] + l_rate * error
            print(' Weight[0] = ' + str(weights[0]))
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                print(' Weight[' + str(i + 1) + '] = ' + str(weights[i + 1]))

            sum_error = sum_error + error ** 2

    return weights

# Calculate weights
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.1
n_epoch = 1
weights = train_weights(dataset, l_rate, n_epoch)
print(weights)
