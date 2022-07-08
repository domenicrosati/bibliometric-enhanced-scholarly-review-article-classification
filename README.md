# Classifying Review Articles by their Reference distribution

Review Articles might select references in a unique way compared to Non-Review Articles.
In particular, we are interested in the following questions:

**Q1:** Are Review Articles engaged in scholarly debate more than non-review articles?
**Q2:** Can Review Articles be classified based on the distribution of their referencing behaviour?

In order to explore these, Pubmed provides a comprehensive index of review versus not review articles (for these purposes we group systematic review and review together). Additionally, we can use scite.ai to gather statistics about the distribution of references made and citations made to references. Notably, scite provides data on in-text citations including classifications of whether the references were contrasted or supported.

In order to answer **Q1**, we can quantify scholarly debate with the following metrics: Was the reference **contrasted?** was the reference **supported**? What was the **ratio** of contrasting to supporting citations.

In order to answer **Q2**, classifiers can be trained on reference distributions and we can inspect the performance and feature coeffecients of the classifier.

## Data

| Number of reviews in dataset            	| 11826208 	|
|-----------------------------------------	|----------	|
| Size of sampled dataset                 	| 100000   	|
| length of train dataset                 	| 80000    	|
| Number of reviews in train dataset      	| 7496     	|
| Number of non-reviews in train dataset  	| 72504    	|
| length of test dataset                  	| 20000    	|
| Number of reviews in test dataset       	| 1948     	|
| Number of non-reviews in test dataset   	| 18052    	|

- reference_distributions_all.csv   (full dataset with reference distributions titles and abstracts)
- review_references_title_abstracts_sample_train.csv (training set)
- review_references_title_abstracts_sample_test.csv  (testing set)
- reviews_with_embeddings_sample_train.csv (training set with embeddings)
- reviews_with_embeddings_sample_test.csv (testing set with embeddings)
## Reproduction and Setup

Requires Python 3.7 or above and awscli
```
$ pip install requirements.txt
```

Copy the data from `s3://research-data-dumps/classifying_review_articles_by_references/`
```
$ mkdir ./data
$ aws cp s3://research-data-dumps/classifying_review_articles_by_references/ ./data --recursieve
```

Each of the experiments are Jupyter Notebooks.
Note that the Language Model notebooks require using a GPU which are freely available on services such as colab, kaggle, sagemaker, or paperspace.

### Experiments

**Characteriation**

- **Characterize Reference Distributions of Revew v Non Review Articles**: characterize_reference_distributions.ipynb

**Classification**

- **Deberta model that uses an MLP over embeddings and bibliometric features** DebertaV2ForSequenceClassificationWithContext.py
- **finetune DeBERTav3 with bibliometric context** finetune_DeBERTa_with_context.ipynb
- **finetune DeBERTav3 without bibliometric context** finetune_DeBERTa.ipynb
- **Test Deberta models on test set**: test_DeBERTa.ipynb
- **Generate SPECTER embeddings**: generate_specter_embeddings.ipynb
- **Train SVM with SPECTER embeddings**: specter_svm_embeddings_only.ipynb
- **Train SVM with SPECTER embeddings and bibliometric features**: specter_svm_embeddings_and_references.ipynb
- **Train Logistic Regression on bibliometric features only**: log_regression_references.ipynb
- **Train Logistic Regression on whether review is in title or abstract**: log_regression_review_in_title_abstract.ipynb
- **Train Logistic Regression on whether review is in title or abstract and bibliometric features**: log_regression_title_abstract_references
- **Train Logistic Regression on TFIDF features**: log_regression_title_abstract_tfidf.ipynb
- **Train Logistic Regression on TFIDF features and bibliometric features**: log_regression_title_abstract_tfidf_references.ipynb

