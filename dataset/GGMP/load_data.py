import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# modified by Cy: replace global variable with parsed parameter
# infolder = Path('/home/zhk/project2/shandong_t2d/data/GGMP/')
# genus_table_file = infolder / 'table-filtered-feature-rarefied5k-L6.tsv'
# metadata_file = infolder / 'sample-metadata.tsv'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def filter_taxa(taxa_annotation, prefix):
    last_rank = taxa_annotation.split(';')[-1]
    last_rank = last_rank.strip(' ')
    return last_rank.startswith(prefix)


def get_genus_table_for_xgb(genus_table_file):
    # the raw table are normalized counts of genera among samples
    genus_table = pd.read_csv(genus_table_file, index_col=0, header=0, sep='\t')
    genus_table.index.name = 'SampleID'
    # transform to relative abundance profile
    genus_table = genus_table.div(genus_table.sum(axis=1), axis=0)

    taxa_names = [it for it in genus_table.columns if filter_taxa(it, prefix='g__')]
    genus_table = genus_table[taxa_names]

    # data preparation: clean column/feature names for xgboost...
    pat = re.compile(r"\[|\]|<")
    feature_names = [str(it) for it in genus_table.columns]
    feature_names = [pat.sub(r' ', it) for it in feature_names]
    genus_table.columns = feature_names
    return genus_table


def get_sample_info(metadata_file):
    data = pd.read_csv(metadata_file, index_col=0, header=0, sep='\t')
    data.index.name = 'SampleID'
    return data


def get_disease_and_healthy(metadata_file, disease='T2DM'):
    all_samples = get_sample_info(metadata_file)
    sub_samples = all_samples[(all_samples[disease] == 1) | (all_samples['Health'] == 1)]
    disease_label = pd.Series(np.zeros_like(sub_samples[disease]), index=sub_samples.index)
    disease_label[sub_samples[disease] == 1] = 1
    sub_samples[disease] = disease_label.astype(bool)
    # modified by Cy: uniform column dtype Health (bool) to SGMP
    sub_samples['Health'] = ~disease_label.astype(bool)
    return sub_samples


if __name__ == '__main__':
    infolder = Path('/home/zhk/project2/shandong_t2d/data/GGMP/')
    genus_table_file = infolder / 'table-filtered-feature-rarefied5k-L6.tsv'
    metadata_file = infolder / 'sample-metadata.tsv'

    genus_table = get_genus_table_for_xgb(genus_table_file)
    print(genus_table.shape)
    print(genus_table.head())
    # assert that sum of relative abundances for a sample (a row) is around 1.0
    print(genus_table.sum(axis=1))

    samples = get_sample_info(genus_table_file)
    print(samples.columns)
    print(samples.head())

    samples = get_disease_and_healthy(genus_table_file, disease='T2DM')
    print(samples.head())
    print(samples.shape)
    print(samples['T2DM'].value_counts())
