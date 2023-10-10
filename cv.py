from sklearn.model_selection import KFold,GridSearchCV,train_test_split
from sklearn.metrics import make_scorer

#X_train, X_val, y_train, y_val = train_test_split(subsampled_x, subsampled_y, test_size=0.3, random_state=42)

cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
outer_results = list()

for train_ix, test_ix in cv_outer.split(subsampled_x):
    X_train, X_test = subsampled_x[train_ix, :], subsampled_x[test_ix, :]
    y_train, y_test = subsampled_y[train_ix], subsampled_y[test_ix]
    loss  = make_scorer(cost_function, greater_is_better=False, arg1=test_x_AREA[test_ix])
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    model = GaussianProcessRegressor(kernel=RBF(),random_state=35842)
    space = [0.001,0.01,0.1,0.3,0.5]
    search = GridSearchCV(model, space, scoring=loss, cv=cv_inner, refit=True)
    result = search.fit(X_train, y_train)
    best_model = result.best_estimator_
    yhat = best_model.predict(X_test)
    acc = cost_function(y_test, yhat,train_x_AREA[train_ix])
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
print('Loss: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))