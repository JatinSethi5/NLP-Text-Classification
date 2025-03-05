# NLP Text Classification Project

Multiclass text categorization system analyzing news/article content across 18 categories (politics, sports, economy, etc.) using NLP techniques.

Dive into the sections below to discover more about our project:

- [Team](#team)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)

## Team

Our team comprises of Post Baccalaureate Data Science students at the Thomposon River University.

- [Jatin Sethi](https://github.com/JatinSethi5)
- [Soumye Kumar](https://github.com/SoumyeKumar)
- [Vishesh Khurana](https://github.com/vkay47)
- [Solomon Maccarthy](https://github.com/FiiMac)

## Project Overview

**Objective**: Develop a machine learning system to classify text documents into 18 distinct categories using:
- Advanced text preprocessing (HTML cleaning, tokenization, lemmatization)
- Stratified data sampling
- NLTK and custom tokenization pipelines
- Scikit-learn model integration

**Key Features**:
- Handles raw text containing HTML markup
- Implements balanced dataset splitting (98%/2% stratified splits)
- Prepares text for ML models through stopword removal and normalization
- Supports 18 news categories including Politics, Environment, Health, and Sports


## Installation

Follow the instructions below to run the project locally.

1. Clone the repo:

```bash
git clone https://github.com/JatinSethi5/NLP-Text-Classification.git
```

2.  and activate the required environment:

```bash
conda env create -f nlp_env.yaml
conda activate nlp
```


3. Download NLTK resources:
```bash
python -m nltk.downloader punkt_tab stopwords wordnet
```



## Usage

Follow the instructions below to run the project locally.

1. Run `nlp_model.ipynb` Jupyter notebook for:
  - Data loading and inspection
  - Text cleaning pipeline execution
  - Stratified train/test splits
  - Category distribution analysis

2. Run Flask Application
```bash
python application.py
```


