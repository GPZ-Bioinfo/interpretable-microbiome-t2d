# Interpretable machine learning for T2DM microbiome meta-analysis
This is an **interpretable** machine learning tool applied on **multi-cohort** of 16s rRNA microbiome datasets to predict T2DM.


## how to integrate cohorts from different source?
our approach builds meta-models from random subset of samples (one model for each subset) from either single cohort or pooled multi-cohort datasets. By possible random subsampling from different source of microbiome variation and their disease association. This approach is designed to capture different aspects of disease-predictive information and the specific bias associated with the random subset.


## how to interpret?
To address bias and inconstancy across different cohorts or even regions within a cohort, we developed T2DM prediction models using strategy of ensemble learning. We trained an ensemble of models from random subsampling (without replacement) of an equal number of both healthy and T2DM samples. Each T2DM prediction model was trained using XGBoost with generated SHAP values, which quantify how microbiome features contributed to different models. To filter models with poor performance and verify the significance of consistently contributed features, 100 null models were developed for each within-model cross-validation. By aggregating the SHAP values across remained T2DM prediction models, microbiome signatures are extracted for each samples.

# introduction of datasets
There are two cohort based on 16s rRNA microbiome sequencing involved in this research. One is Guandong Gut Microbiome Project (GGMP) and the other is Shandong Gut Microbiome Project (SGMP).
* GGMP: 7009 community-based fecal 16s rRNA microbiome samples with clinical metadata. Only 2603 T2DM or healthy individuals were selected for model development. The clinical metadata is consist of demographic information, life style, medical history and laboratory indicators.  
* SGMP: 968 T2DM or healthy individuals were sampled for fecal 16s rRNA microbiome in hospital.

# introduction of demonstrated jupyter notebooks
![](https://github.com/GPZ-Bioinfo/interpretable-microbiome-t2d/blob/master/notebooks/images/readme_demo.png)

* ensemble of predictive models
python-based notebook:
after loading GGMP and SGMP datasets, random subsampling was applied to develop each sub-model. XGBoost with generated SHAP values were employed to represent contributions of features. Filtering out sub-models with poor performance of AUC.

* extraction of consistent signatures
python-based notebook:
demonstration of visualizing feature contributions to prediction scores.Identification and extraction of consistent microbiome features for T2DM prediction through feature ranking.

* cross-cohort validation and clinical evaluation
R-based notebook:
visualization the 3 parts of main results. Clinical indicators were involved for evaluation.










:sparkling_heart:  :sparkling_heart:  :sparkling_heart:  :sparkling_heart:  :sparkling_heart:

# Publication

# Contact Us
