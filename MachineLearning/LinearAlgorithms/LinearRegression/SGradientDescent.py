
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat = yhat + coefficients[i + 1] * row[i]
        print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))
    return yhat



# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]  #gets the number of columns in the row and sets it all to 0
    for epoch in range(n_epoch):
        print('Epoch--------------------' + str(epoch ))
        sum_error = 0
        for row in train:

            yhat = predict(row, coef)
            error = yhat - row[-1]                                          # derivative or delta
            print('>error=%.3f' % (error))

            coef[0] = coef[0] - l_rate * error
            print(' Coef[0] = ' + str(coef[0]))
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]         #We use the same equation with one small change. The error is altered by the input that caused it
                print(' Coef[' + str(i + 1) + '] = ' + str(coef[i + 1]))

            sum_error = sum_error + error ** 2

    return coef

dataset = [[3, 4, 7], [2, 3, 2], [4, 3, 4], [3, 2, 3], [5, 5, 4]]
l_rate = 0.0001
n_epoch = 5
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)


# In Stochastic Gradient Descent (SGD), the weight vector gets updated every time you read/process a sample,
# whereas in Gradient Descent (GD) the update is only made after all samples are processed in the iteration.
# Thus, in an iteration in SGD, the weights number of times the weights are updated is equal to the number of examples, while in GD it only happens once.
# SGD is beneficial when it is not possible to process all the data multiple times because your data is huge.


from sklearn import linear_model

y =  [7,2,4,3,4]
X =  [[3, 4], [2, 3], [4, 3], [3, 2], [5, 5]]
clf = linear_model.SGDRegressor(shuffle=False, eta0=0.0001, n_iter=5, learning_rate='constant')
clf.fit(X, y)


print (clf.coef_)
print (clf.intercept_)