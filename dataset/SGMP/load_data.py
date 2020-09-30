import os
import pandas as pd
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# modified by Cy: replace global variable with parsed parameter
# infolder = Path('/home/zhk/project2/shandong_t2d/data/SGMP/')
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


def get_disease_and_healthy(metadata_file):
    # retrieve a subset of T2D and health samples for learning
    data = get_sample_info(metadata_file)
    subset = data.query('Districts=="QD_MF1" | Districts == "QD_MF2"')
    subset['T2DM'] = (subset['host_status'] == "Type 2 diabetes")
    # modified by Cy: uniform column name (healthy -> Health) to GGMP samples
    subset['Health'] = (subset['host_status'] == "Health")
    return subset


if __name__ == '__main__':
    infolder = Path('/home/zhk/project2/shandong_t2d/data/SGMP/')
    genus_table_file = infolder / 'table-filtered-feature-rarefied5k-L6.tsv'
    metadata_file = infolder / 'sample-metadata.tsv'

    genus_table = get_genus_table_for_xgb(genus_table_file)
    print(genus_table.shape)
    print(genus_table.head())
    # assert that sum of relative abundances for a sample (a row) is around 1.0
    print(genus_table.sum(axis=1))

    samples = get_sample_info(metadata_file)
    print(samples.columns)
    print(samples.head())
    print(samples.pivot_table(index='Districts', columns='host_status',
                              values='sample_ID', aggfunc="count"))

    samples = get_disease_and_healthy(metadata_file)
    print(samples.head())
    print(samples.shape)
