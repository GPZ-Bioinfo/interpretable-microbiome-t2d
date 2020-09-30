from signature import shap_utils
import xgboost as xgb


class Bucket:
    """
    bundle (a unit) of dataset for predictive modeling (and more...)
    """
    def __init__(self, train_data, valid_data=None, test_data=None, meta_data={}):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.meta_data = meta_data

    def fit(self, learner, shuffle=True):
        # todo: should a bundle keep the brew result?
        return learner.run(train_data=self.train_data,
                           valid_data=self.valid_data,
                           test_data=self.test_data,
                           shuffle=shuffle)

    def transform(self, model, data='train', with_meta=False):
        if data == 'train':
            X, y = self.train_data
        elif data == 'valid':
            X, y = self.valid_data
        elif data == 'test':
            X, y = self.test_data
        else:
            X, y = data

        d_test = xgb.DMatrix(data=X, label=y)

        _model = model['model']
        # comment by Cy: ntree_limit=_model.best_ntree_limit should be considered.
        # with pred_contribs=True set to obtain SHAP values
        shap_values = _model.predict(data=d_test, ntree_limit=_model.best_iteration,
                                     pred_contribs=True, approx_contribs=False)
        # annotate teh SHAP values with sample ID and column names
        shap_values = shap_utils.wrap_feat_shap_contribs(X=X, y=y, X_shap=shap_values)

        # keep track of meta data about the models
        if with_meta:
            for meta_key, meta_value in self.meta_data.items():
                shap_values[meta_key] = meta_value

        return shap_values

    def get_best_result(self, model, with_meta=False):
        _model = model['model']
        _evals_result = model['evals_result']
        _eval_metric = model['params']['eval_metric']
        best_iteration = _model.best_iteration
        _evals_name = [it[1] for it in model['evals']]
        best_result = {'%s_%s' % (name, _eval_metric): _evals_result[name][_eval_metric][best_iteration]
                       for name in _evals_name}
        if with_meta:
            best_result.update(self.meta_data)

        return best_result
