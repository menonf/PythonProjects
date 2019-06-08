# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]  # gets the number of columns in the row and sets it all to 0
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error ** 2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
    return coef

# Make a prediction
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat