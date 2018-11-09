from sklearn.ensemble import RandomForestRegressor

def housePriceModel( algo, X_train, Y_train ):
    if algo == "randomForest":
        # Creates a random forest regressor model
        return randomForestModel( X_train, Y_train )
    elif algo == "linearRegression":
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        logisticRegr.fit(X_train, y_train)
    
        return logisticRegr
    else:
        return null

def randomForestModel( X_train, Y_train ):
    #Coefficients for random forest regressor
    max_depth = 10
    random_state = 0
    n_estimators = 500      

    regr = RandomForestRegressor(oob_score=True, random_state=random_state, n_estimators=n_estimators)
    regr.fit(X_train, Y_train)
    
    return regr