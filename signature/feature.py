import numpy as np
import pandas as pd
import os
from signature.shap_utils import cal_feature_shap_scores


def get_feature_shap_scores(infile, drop_columns):
    shap_signatures = pd.read_csv(infile, header=0, sep='\t')
    labels = shap_signatures['label']
    # in the model, '-1' for health, '+1' for disease
    assert set(labels.unique()) == {0, 1}
    labels[labels == 0] = -1
    shap_signatures.drop(columns=drop_columns, inplace=True)
    feature_shap_scores = cal_feature_shap_scores(signatures=shap_signatures, labels=labels)
    return feature_shap_scores


def get_feature_shap_scores_null(inpath, model_name, dataset_name, drop_columns, n=100):
    feature_shap_scores_nulls = []
    for p in range(n):
        infile = os.path.join(inpath, '{}_p_{}_shap_{}.tsv'.format(model_name, p+1, dataset_name))
        feature_shap_scores = get_feature_shap_scores(infile, drop_columns=drop_columns)
        feature_shap_scores.name = str(p+1)
        feature_shap_scores_nulls.append(feature_shap_scores)
    return feature_shap_scores_nulls


def rank_shap_scores(shap_scores, rank_n):
    # concat shap score from multiple models
    shap_scores = pd.concat(shap_scores, axis=0)
    feature_scores = FeatureScores(scores=shap_scores, names=shap_scores.index)
    all_names = set(shap_scores.index)
    # in a tricky case, the lowest-scoring feature would not get a rank, fill it with zero
    ranks = pd.Series(np.zeros_like(len(all_names)), index=all_names)
    feature_ranks = feature_scores.get_ranks(r=rank_n)
    ranks.loc[feature_ranks.index] = feature_ranks
    return ranks.sort_values(ascending=False)


class FeatureScores:
    def __init__(self, scores, names):
        self.scores = scores
        self.names = names
        self.overall_count = self._assert_names()

    def _assert_names(self):
        # number of occurrence of each name should be  the same
        name_counts = self.names.value_counts()
        assert name_counts.min() == name_counts.max()
        return name_counts.min()

    def get_name_counts_above_quantile(self, q):
        # value counts above the quantile
        # return is pandas Series, index by feature names, and values are their counts
        qantile = self.scores.quantile(q=q)
        return self.names[self.scores > qantile].value_counts()

    def get_ranks(self, r=1000):
        q_counts_all = []
        for q in range(r):
            # count the number of occurrence of the features above the rank/q
            # normalized by maximum possible time of occurrence
            q_counts = self.get_name_counts_above_quantile(q=(q * 1.0) / r) / self.overall_count
            q_counts.name = q
            q_counts_all.append(q_counts)

        # concat into a q_counts_all table, columns are q_counts, rows are feature names
        q_counts_all = pd.concat(q_counts_all, axis=1)
        q_counts_all.fillna(0, inplace=True)
        q_ranks = q_counts_all.sum(axis=1) / r
        return q_ranks

