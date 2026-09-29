# UK Prime Minister Sentiment Analysis

## Project Overview

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze sentiment in text related to the UK Prime Minister.

The project classifies posts into three sentiment categories:

* **Positive**
* **Neutral**
* **Negative**

Several supervised machine learning algorithms are trained and compared to determine how effectively they classify sentiment from text.

> **Note:** The sentiment distribution in this project describes the dataset used for the analysis and should not be interpreted as a measure of the opinions of the entire UK population.

---

## Objectives

The main objectives of this project are to:

* Explore and understand the sentiment dataset
* Clean and preprocess text data
* Analyze the distribution of sentiment
* Identify frequently occurring words
* Convert text into numerical features using **TF-IDF**
* Train multiple machine learning classification models
* Compare model performance using different evaluation metrics
* Perform cross-validation
* Analyze incorrectly classified observations
* Build a function for predicting sentiment from new statements
* Save the best-performing trained model

---

## Dataset

The project uses the **SentiMP-En** dataset available through Hugging Face:

`rbnuria/SentiMP-En`

The dataset contains text data with gold-standard sentiment labels.

The sentiment labels were mapped as follows:

| Original Label | Sentiment |
| -------------- | --------- |
| -1             | Negative  |
| 0              | Neutral   |
| 1              | Positive  |

After data preparation and duplicate removal, the analysis contained **479 observations**.

---

## Tools & Technologies

The project was developed using **Python** and the following libraries:

* **Pandas** – data manipulation and analysis
* **NumPy** – numerical operations
* **Matplotlib** – data visualization
* **NLTK** – text preprocessing and lemmatization
* **WordCloud** – visualization of frequently occurring words
* **Scikit-learn** – machine learning and model evaluation
* **Hugging Face Datasets** – dataset loading
* **Joblib** – saving the trained model

---

## Project Workflow

### 1. Data Loading

The dataset was loaded using the Hugging Face `datasets` library and converted into a Pandas DataFrame.

### 2. Data Cleaning

The dataset was inspected for:

* Missing values
* Duplicate records
* Data types
* Dataset structure

The analysis focused on:

* `full_text` – input text
* `gold_label` – sentiment target

The `tie_break` variable was excluded from the sentiment modelling process.

### 3. Sentiment Mapping

The original numerical labels were converted into readable categories:

```text
-1 → Negative
 0 → Neutral
 1 → Positive
```

### 4. Exploratory Data Analysis

The project examined:

* Sentiment frequency
* Sentiment percentage distribution
* Text character counts
* Text word counts

The sentiment distribution in the analyzed dataset was:

| Sentiment | Frequency | Percentage |
| --------- | --------: | ---------: |
| Negative  |       134 |     27.97% |
| Neutral   |       101 |     21.09% |
| Positive  |       244 |     50.94% |

### 5. Text Preprocessing

The text preprocessing pipeline included:

* Converting text to lowercase
* Removing URLs
* Removing user mentions
* Removing the `#` symbol while retaining hashtag words
* Removing HTML entities
* Removing numbers
* Removing punctuation
* Removing unnecessary whitespace
* Removing stop words
* Retaining important negation words such as `not`, `no`, and `nor`
* Lemmatization using NLTK

### 6. Word Cloud Analysis

Word clouds were generated separately for:

* Negative posts
* Neutral posts
* Positive posts

This provides a visual representation of frequently occurring words within each sentiment category.

---

## Machine Learning Models

Four supervised classification algorithms were compared:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine
4. Random Forest

Text was transformed into numerical features using **TF-IDF Vectorization** with unigram and bigram features.

The dataset was divided into:

* **80% training data**
* **20% testing data**

A fixed random state of `42` and stratified splitting were used.

---

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* Weighted F1 Score
* Classification Report
* Confusion Matrix

### Hold-Out Test Results

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression     |   0.7188 |    0.7082 | 0.7188 |   0.7027 |
| Linear SVM              |   0.6667 |    0.6554 | 0.6667 |   0.6377 |
| Random Forest           |   0.5521 |    0.5146 | 0.5521 |   0.4743 |
| Multinomial Naive Bayes |   0.5208 |    0.5445 | 0.5208 |   0.3675 |

Based on the hold-out test evaluation, the Logistic Regression model achieved a weighted F1 score of **0.7027**.

---

## Cross-Validation

To obtain a more stable estimate of model performance, **5-fold stratified cross-validation** was also performed using weighted F1 score.

| Model                   | Mean CV F1 | Standard Deviation |
| ----------------------- | ---------: | -----------------: |
| Logistic Regression     |     0.6676 |             0.0501 |
| Linear SVM              |     0.6417 |             0.0346 |
| Random Forest           |     0.5405 |             0.0221 |
| Multinomial Naive Bayes |     0.3821 |             0.0287 |

The cross-validation results provide an additional view of model performance beyond the single train-test split.

---

## Confusion Matrix

A confusion matrix was generated for the selected model to examine how predictions were distributed across:

* Negative
* Neutral
* Positive

This helps identify which sentiment categories were most frequently confused by the classifier.

---

## Error Analysis

The project also examined misclassified test observations.

A total of **27 test posts** were misclassified by the selected model.

Examining these errors helps identify areas where sentiment classification can be difficult, particularly when text contains:

* Ambiguous language
* Context-dependent statements
* Political references
* Mixed sentiment
* Short or information-poor statements

---

## Sentiment Prediction

A reusable function was created to predict the sentiment of new statements:

```python
predict_sentiment("The Prime Minister is doing an excellent job for the country.")
```

The function applies the same preprocessing pipeline before passing the text to the trained model.

---

## Model Saving

The selected model was saved using Joblib:

```text
uk_pm_sentiment_best_model.pkl
```

This allows the trained model to be reused without retraining it from scratch.

---

## Project Structure

```text
UK-PM-Sentiment-Analysis/
│
├── UK_PM_Sentiment_Analysis.ipynb
├── uk_pm_sentiment_best_model.pkl
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/UK-PM-Sentiment-Analysis.git
```

Navigate into the project directory:

```bash
cd UK-PM-Sentiment-Analysis
```

Install the required libraries:

```bash
pip install datasets pandas matplotlib scikit-learn nltk wordcloud joblib numpy
```

The notebook also downloads the required NLTK resources:

```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook or upload it to Google Colab.
4. Run the notebook cells from top to bottom.
5. Review the exploratory analysis, visualizations, model results, and sentiment predictions.

---

## Key Skills Demonstrated

This project demonstrates practical experience with:

* Python Programming
* Pandas
* NumPy
* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Natural Language Processing
* Text Preprocessing
* TF-IDF Feature Extraction
* Supervised Machine Learning
* Classification
* Model Evaluation
* Cross-Validation
* Confusion Matrix
* Error Analysis
* Model Serialization with Joblib

---

## Future Improvements

Possible improvements to the project include:

* Hyperparameter tuning
* Testing additional NLP models
* Experimenting with word embeddings
* Using transformer-based models such as BERT
* Increasing the size and diversity of the training data
* Performing more detailed error analysis
* Building a simple web application for real-time sentiment prediction
* Adding an interactive dashboard for sentiment exploration

---

## Disclaimer

This project is intended for **educational and analytical purposes**. The sentiment classifications reflect the labels and patterns present in the dataset used for this analysis and should not be treated as a comprehensive representation of public opinion about the UK Prime Minister.

---

## Author

**Uzoka Esomchukwu Peace**

Aspiring Data Scientist | Python | Data Analysis | Machine Learning | NLP
