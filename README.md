# AI-Related Article Detector

A comprehensive Python project that scrapes, processes, and classifies articles as **"AI-related"** or **"Not AI-related"**. The project leverages web scraping, text cleaning, BERT embeddings, logistic regression, and interactive visualizations—all bundled into a user-friendly Streamlit web application.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Pipeline](#training-pipeline)
  - [Inference and Streamlit App](#inference-and-streamlit-app)
- [Results and Evaluation](#results-and-evaluation)
- [Notes](#notes)
- [License](#license)

## Features

- **Web Scraping:**  
  Collect articles from sources like DeepLearning.ai and Al Jazeera using Selenium.

- **Text Processing:**  
  Clean and preprocess article text using NLTK (tokenization, lemmatization, stopword removal, etc.).

- **BERT Embeddings:**  
  Use a pre-trained BERT model for generating text embeddings.

- **Dimensionality Reduction & Clustering:**  
  Reduce embedding dimensions with PCA/UMAP and visualize clusters in 2D/3D.

- **Logistic Regression Classifier:**  
  Train a classifier to distinguish between AI-related and non-AI-related content.

- **Streamlit Interface:**  
  A beautiful and interactive UI for both training and inference, complete with custom styling and caching.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Kingflow-23/AI-related-article-detector.git
   ```

2. Create and Activate a Virtual Environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install Dependencies:
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. Configure Selenium:

Ensure you have the appropriate ChromeDriver or equivalent for your browser and update its path in your project configuration if necessary.

5. Set Up Configuration:

Edit the config.py file with your desired configuration settings, such as:

- Model and tokenizer names.
- Output directories.
- Scraping parameters.

## Usage

### Training Pipeline

To run the full training pipeline that includes web scraping, text extraction, embedding, dimensionality reduction, clustering, and classifier training:

Via Command Line:
```bash 
python src/ai_related_topic_classifier.py
```


## Inference and Streamlit App
The Streamlit application supports both training and inference modes.

1. **Run the Streamlit App:**

```bash
streamlit run src/app.py --server.fileWatcherType none
```

2. **Inference Mode:**

- Switch to Inference in the sidebar.
- Enter the text you want to classify.
- Click Classify to see the prediction along with the confidence score (if available).
- If no trained model is present, a warning message will instruct you to run the training pipeline.

3. **Clear Cache:**

Use the Clear Cache button in the sidebar to refresh loaded resources (e.g., after updating the model).

## Results and Evaluation

The Logistic Regression model shows **overfitting**, with **perfect training accuracy (1.0)** but a **lower cross-validation score**, indicating poor generalization. The **classification report** highlights a severe class imbalance:  

![UMAP_classification_result_2025-03-15_18-15-04](https://github.com/user-attachments/assets/a9c0872d-a31c-4fa6-be5d-53f7a8e526a3)

![PCA_classification_result_2025-03-15_18-15-02](https://github.com/user-attachments/assets/4c5209fa-53fa-41de-8a0b-fb64fae8034b)

![learning_curve_2025-03-15_18-15-05](https://github.com/user-attachments/assets/9f7c4bb8-1206-4a31-8cc0-8a77481e4d11)

```
Classification Report:

              precision    recall  f1-score   support

           0       1.00      0.02      0.04       100
           1       0.51      1.00      0.67       100

    accuracy                           0.51       200
   macro avg       0.75      0.51      0.36       200
weighted avg       0.75      0.51      0.36       200
```

- **Non-AI-related (Class 0)**: **High precision (1.00) but very low recall (0.02)** → The model fails to detect most non-AI articles.  
- **AI-related (Class 1)**: **Full recall (1.00) but low precision (0.51)** → Many false positives.  
- **Overall accuracy: 51%**, barely better than random guessing.  

### 🛠️ Recommendations  
- **Balance the dataset** (e.g., oversampling, SMOTE).  
- **Improve feature engineering** (better embeddings, remove stopwords).  
- **Apply regularization & hyperparameter tuning** to prevent overfitting.  
- **Consider more advanced models** like Random Forest or Deep Learning.  

The current model **leans heavily toward AI-related articles and needs refinement** to improve generalization and fairness. 🚀

## Notes

- **File Watcher Issues:**
The Streamlit app disables file watching to avoid Torch-related errors. If you encounter issues, ensure you have the correct versions of libraries installed.

- **Performance:**
Web scraping and model training may be resource-intensive and require a proper runtime environment (consider headless mode for Selenium).


## License

This project is licensed under the MIT License.   
