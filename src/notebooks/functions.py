def compileNANcols(DF):
    global NANlist
    NANlist = []
    NANlist = list(DF.columns[DF.isnull().any()])
    return NANlist

def imputeNANs(DF, col):
    X = DF[col].values.reshape(-1,1)
    imputer = KNNImputer(
            missing_values = np.nan,
            n_neighbors=3, 
            copy=False
            )
    imputer.fit(X)
    Xtrans = imputer.transform(X)
    DF[col] = Xtrans.ravel().tolist()

def cleanupDF(DF):    
    compileNANcols(DF)
    for col in NANlist:
        imputeNANs(DF, col)

def save_results(model, param_notes):
    models_summary.loc[len(models_summary)] = [model, model.score(X_test, y_test), param_notes]