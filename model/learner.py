import xgboost as xgb

_default_xgb_params = {'objective': 'binary:logistic',
                       'booster': 'gbtree',
                       'eval_metric': 'auc',
                       'max_depth': 3,
                       'eta': 0.3,
                       'min_child_weight': 5,
                       'subsample': 0.8,
                       'colsample_bytree': 0.8,
                       'seed': 888,
                       'silent': True}


class XGB:
    def __init__(self, params, num_rounds=500, early_stopping_rounds=50):
        """
        XGB 'global' leaner parameters
        :param params:
        """
        self.params = params
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds

    def run(self, train_data, valid_data=None, test_data=None, shuffle=True):
        """
        run data & run specific parameters
        :param train_data:
        :param valid_data:
        :param test_data:
        :param num_rounds:
        :param early_stopping_rounds:
        :return:
        """
        train_X, train_y = train_data

        # shuffle the training set, in random order of samples
        if shuffle:
            train_X = train_X.sample(frac=1)
            train_y = train_y.loc[train_X.index]

        d_train = xgb.DMatrix(data=train_X, label=train_y)

        watchlist = [(d_train, 'train')]
        # modified by Cy: remove redundant declaration
        # d_test = None
        if test_data:
            d_test = xgb.DMatrix(data=test_data[0], label=test_data[1])
            watchlist.append((d_test, 'test'))
        # modified by Cy: remove redundant declaration
        # d_valid = None
        if valid_data:
            d_valid = xgb.DMatrix(data=valid_data[0], label=valid_data[1])
            watchlist.append((d_valid, 'valid'))

        evals_result = {}
        _model = xgb.train(params=self.params,
                           dtrain=d_train,
                           num_boost_round=self.num_rounds,
                           evals=watchlist,
                           early_stopping_rounds=self.early_stopping_rounds,
                           verbose_eval=False,
                           evals_result=evals_result)

        return {'model': _model,
                'evals_result': evals_result,
                'evals': watchlist,
                'params': self.params}

