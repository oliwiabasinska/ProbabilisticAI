alpha = np.linspace(0,10,num=100)
min_alpha = 0
min_score = 40000

for a in alpha:
    gp_mean, gp_std = gpr.predict(train_x_2D, return_std=True)
    predictions = gp_mean + a * gp_std
    score = cost_function(train_y, predictions, train_x_AREA)
    if score < min_score:
        print(score,a)
        min_score = score
        min_alpha = a