import pandas as pd


def wrap_feat_shap_contribs(X, y, X_shap):
    """
    annotate the shap values with column and sample names
    X_shap is the return from xgb model predict (pred_contribs=True)
    the last column is the bias term
    X is the input dataframe, with sample and feature labels
    :param X:
    :param X_shap:
    :return:
    """
    # the last value is the bias term
    X_shap = pd.DataFrame(X_shap, columns=list(X.columns)+['bias'], index=X.index)
    # y should in the same order of samples, as that of X!
    X_shap['label'] = y
    return X_shap


def cal_feature_shap_scores(signatures, labels):
    # labels of the training samples, -1 for control (health), +1 for case (disease)
    assert set(labels.unique()) == {-1, 1}
    # transform shap values to shap score (contribution to the prediction) for each sample
    # negative shap value -> health (-1); positive shap value -> disease (+1)
    # the larger the score, the greater the contribution, regardless of the labels
    feature_shap_scores = signatures.mul(labels, axis="index")
    # mean contribution across samples (in a model)
    feature_shap_scores = feature_shap_scores.mean(axis=0)
    # sort the mean scores in descending order
    feature_shap_scores = feature_shap_scores.sort_values(ascending=False)
    return feature_shap_scores


def cal_feature_shap_impacts(signatures):
    # shap impacts (regardless of labels) proposed in the original SHAP paper
    # not normalized, only comparable WITHIN model
    feature_shap_impacts = signatures.abs().sum(axis=0)
    feature_shap_impacts = feature_shap_impacts.sort_values(ascending=False)
    return feature_shap_impacts
