import os
import re
import umap
import time
import nltk
import torch
import pickle
import requests
import xmltodict
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from datetime import datetime
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    learning_curve,
    cross_val_score,
    train_test_split,
    StratifiedKFold,
)

from yellowbrick.cluster import KElbowVisualizer
from transformers import BertTokenizer, BertModel

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from config import *

nltk.download("punkt")  # Required for tokenization
nltk.download("omw-1.4")  # Optional for better language support
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


def get_article_url(search_term: str, label: int, max_urls: int) -> pd.DataFrame:
    """
    Scrapes URLs of articles from either "DeepLearning.ai" or "Al Jazeera" based on the provided label.

    This function uses Selenium to search for articles on a specified website and retrieve a set number of URLs.
    The target website is determined by the `label` parameter:
    - If `label == 1`, the function will search on the "DeepLearning.ai - The Batch" page.
    - If `label == 0`, the function will search on the "Al Jazeera" news website.

    Args:
        search_term (str): The keyword or phrase to search for on the respective website.
        label (int): A flag indicating which website to scrape.
            - 1: Scrapes the "DeepLearning.ai" website.
            - 0: Scrapes the "Al Jazeera" website.
        max_urls (int): The maximum number of URLs to scrape. The function will stop scraping after this limit is reached
                        or if there are no more pages to retrieve.

    Returns:
        pandas.DataFrame: A DataFrame containing the URLs scraped from the respective website.
                          The DataFrame has two columns:
                          - "Urls": The scraped URLs.
                          - "Labels": A column indicating the source website (1 for "DeepLearning.ai", 0 for "Al Jazeera").
                          Each row in the DataFrame corresponds to a single article URL.

    Raises:
        TimeoutException: If an element is not clickable or not found within the waiting time.
        WebDriverException: If an error occurs with the WebDriver during execution.
        Exception: For any other general exceptions encountered while navigating and scraping the website.

    Behavior:
        - For "DeepLearning.ai", the function opens the search page, enters the search term, and navigates through the result pages,
          scraping URLs of articles related to the search term.
        - For "Al Jazeera", the function clicks the news section and keeps expanding the article list by clicking the "Show more"
          button, scraping URLs until the desired number is reached.
        - The function closes the browser after scraping is complete or if an error occurs.

    Example:
        # Scrape 10 articles related to 'AI' from DeepLearning.ai
        df = get_article_url('AI', 1, 10)

        # Scrape 5 articles related to any subject from Al Jazeera. Doesn't matter what you set as search_term.
        # The article retrieved can be about ai but we won't mind as it will be a minority of articles.
        # The search term will be set as index of the dataframe for reference.

        df = get_article_url('Write anything you want here', 0, 5)
    """

    driver = webdriver.Chrome()
    driver.maximize_window()

    links = []

    try:
        if label == 1:
            # Scraping from DeepLearning.ai
            driver.get("https://www.deeplearning.ai/the-batch/")

            # Wait and click the search button
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//*[@id="content"]/nav/div/div[2]/a')
                )
            )
            search_button.click()

            # Enter the search term
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="content"]/div[1]/div/div/input')
                )
            )
            search_box.send_keys(search_term)

            # Wait for the search results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="content"]/div[2]/div/div[3]/nav/div')
                )
            )

            cpt_page = 0

            # Scraping URLs across pages
            while cpt_page != MAX_PAGES:
                print("-" * 100)
                print(f"Working on page {cpt_page}.\n")
                try:
                    body_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="content"]/div[2]/div/ol')
                        )
                    )
                    all_links_in_body = body_element.find_elements(By.TAG_NAME, "a")
                    all_urls_in_body = [
                        link.get_attribute("href")
                        for link in all_links_in_body
                        if link.get_attribute("href")
                    ]

                    links.extend(all_urls_in_body)
                    links = list(set(links))

                    print(
                        f"Found {len(all_urls_in_body)} articles on this page.\nTotal non-duplicated links found for the moment: {len(links)}.\n"
                    )

                    if len(links) >= max_urls:
                        print(f"Found {max_urls} articles. Stopping the search.\n")
                        links = links[:max_urls]
                        break

                    # Go to next page.
                    try:
                        next_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//a[@aria-label="Next page"]')
                            )
                        )
                        next_button.click()
                        print("Going to next page.\n")
                        time.sleep(1)

                        cpt_page += 1

                    except TimeoutException:
                        print("No more 'next' button available.\n")
                        break

                except Exception as e:
                    print(f"Error on page {cpt_page}: {e}\n")
                    break

        else:
            # Scraping from Al Jazeera
            driver.get("https://www.aljazeera.com")

            # Handle cookie consent
            cookie_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//*[@id="onetrust-reject-all-handler"]')
                )
            )
            cookie_button.click()

            # Navigate to the news section
            news_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        '//*[@id="root"]/div/div[1]/div[1]/div/header/nav/ul/li[1]/a',
                    )
                )
            )
            news_button.click()

            # Loop to collect URLs
            while True:
                try:
                    body_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (
                                By.XPATH,
                                '//*[@id="news-feed-container"]',
                            )
                        )
                    )
                    all_links_in_body = body_element.find_elements(By.TAG_NAME, "a")
                    all_urls_in_body = [
                        link.get_attribute("href")
                        for link in all_links_in_body
                        if link.get_attribute("href")
                    ]

                    links.extend(all_urls_in_body)
                    links = list(set(links))

                    print("-" * 100)
                    print(
                        f"Found {len(all_urls_in_body)} articles on the whole page.\nTotal links non-duplicated found for the moment: {len(links)}.\n"
                    )

                    if len(links) >= max_urls:
                        print(f"Found {max_urls} articles. Stopping the search.\n")
                        links = links[:max_urls]
                        break

                    # Click 'Show more'
                    try:
                        show_more_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//*[@id="news-feed-container"]/button')
                            )
                        )
                        show_more_button.click()
                        print("Showing more articles.\n")
                        time.sleep(1)
                    except TimeoutException:
                        print("No more 'Show more' button available.\n")
                        break

                except Exception as e:
                    print("An unexpected error occurred:", e)
                    break

    except WebDriverException as e:
        print("WebDriver error:", e)

    finally:
        # Close the browser after processing
        driver.quit()

        # Return the scraped URLs in a DataFrame
        article_info = pd.DataFrame({"Urls": links}, index=[search_term] * len(links))
        article_info["Labels"] = [label] * len(links)

        return article_info


def get_wordnet_pos(word: str):
    """
    Map the part-of-speech (POS) tag of a word to a format compatible with WordNetLemmatizer.

    This function takes a word as input, determines its POS tag using NLTK's `pos_tag` function,
    and maps it to a format that the WordNetLemmatizer can interpret. WordNetLemmatizer requires
    specific tags to lemmatize words correctly. The function will map:

    - "J" to `wordnet.ADJ` (adjective)
    - "N" to `wordnet.NOUN` (noun)
    - "V" to `wordnet.VERB` (verb)
    - "R" to `wordnet.ADV` (adverb)

    If the POS tag does not match one of these categories, it defaults to `wordnet.NOUN`.

    Args:
        word (str): The word for which the POS tag needs to be determined.

    Returns:
        str: The WordNet-compatible POS tag (e.g., `wordnet.ADJ`, `wordnet.NOUN`, etc.). Defaults to `wordnet.NOUN`
             if the tag is not in the specified mapping.

    Example:
        # Get the WordNet-compatible POS tag for the word 'running'
        pos = get_wordnet_pos('running')  # returns wordnet.VERB

        # For an adjective like 'beautiful'
        pos = get_wordnet_pos('beautiful')  # returns wordnet.ADJ
    """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def remove_stopwords(words: list) -> list:
    """
    Removes stopwords from a list of words.
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return [word for word in words if word not in stop_words]


def cleaning_text(text: str) -> str:
    """
    Cleans and processes raw text extracted from article by removing unwanted characters,
    converting to lowercase, and lemmatizing the words. The cleaning includes removing punctuation,
    non-alphabetic characters, and extra spaces. Each word is lemmatized based on its part-of-speech (POS).

    Args:
        text (str): The raw text extracted from a Wikipedia page.

    Returns:
        str: The cleaned and lemmatized version of the input text.
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove newline characters and replace multiple spaces with single space
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)

    # 3. Remove punctuation and non-alpha characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove stopwords.
    words = remove_stopwords(text.split())

    # 5. Lemmatize each word based on POS tagging
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words
    ]

    # 6. Join the words back into a single string
    text = " ".join(lemmatized_sentence)

    return text


def get_text_from_links(article_info: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves and cleans the text from a list of article links provided in a DataFrame.

    This function navigates to each URL in the `article_info` DataFrame using a Selenium WebDriver, extracts the main
    content from the page, and cleans it using the `cleaning_text` function. The cleaned text is then added to the
    DataFrame in a new column named "Texts". For Al Jazeera articles, it also handles the cookie consent banner if
    encountered. The text for each article is scraped based on specific HTML element locations.

    Args:
        article_info (pd.DataFrame): A DataFrame containing article URLs in one column. It expects the URLs to be
                                     either from "DeepLearning.ai" or "Al Jazeera" websites.

    Returns:
        pd.DataFrame: The input DataFrame, extended with a new "Texts" column that contains the cleaned text from
                      each article link.
    """
    driver = webdriver.Chrome()
    driver.maximize_window()

    texts = []

    al_jazerra_clicked_cookie = 0

    for i, link in enumerate(article_info["Urls"]):
        try:
            driver.get(link)
            print("-" * 100)
            print(f"Treating article {i} of {article_info.shape[0]}.\n")

            if link.startswith("https://www.aljazeera.com"):

                # Click on the cookie button for Al Jazeera articles if it's the first visit
                if al_jazerra_clicked_cookie == 0:
                    try:
                        cookie_button = cookie_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, '//*[@id="onetrust-reject-all-handler"]')
                            )
                        )
                        cookie_button.click()

                        al_jazerra_clicked_cookie += 1

                    except Exception as e:
                        print(f"Cookie button not found: {e}")

                # Retrieve and clean the text from Al Jazeera
                text = cleaning_text(
                    WebDriverWait(driver, 10)
                    .until(
                        EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="main-content-area"]')
                        )
                    )
                    .text
                )
                texts.append(text)

            else:  # For other articles (DeepLearning.ai)
                # Retrieve and clean the text from other sources
                text = cleaning_text(
                    WebDriverWait(driver, 10)
                    .until(
                        EC.presence_of_element_located(
                            (By.XPATH, '//*[@id="content"]/article/div/div')
                        )
                    )
                    .text
                )
                texts.append(text)
                print("Article text retrieved and cleaned.\n")

        except Exception as e:
            print(f"An error occurred while navigating in {link}: {e}")
            texts.append(None)  # Append None for unsuccessful retrieval
            print("Couldn't retrieve article text !!")

    article_info["Texts"] = texts

    return article_info


def get_bert_embedding(
    model, tokenizer, corpus: list, batch_size: int = 32
) -> torch.Tensor:
    """
    Computes BERT embeddings for a given text corpus in batches to optimize memory usage and performance.

    This function processes the text corpus in smaller batches to avoid memory overload. It uses a pre-trained
    BERT model to generate embeddings. The first token of the output (usually the [CLS] token) is returned as
    the representative embedding for each input sentence.

    Args:
        model: A pre-trained BERT model.
        tokenizer: A BERT tokenizer.
        corpus (list or str): A list of sentences or a single string of text for which to compute the embeddings.
                              If a single string is provided, it will be treated as a list with one entry.
        batch_size (int): The number of sentences to process in each batch. Default is 32.

    Returns:
        torch.Tensor: A tensor containing the BERT embeddings for the corpus. The shape of the tensor will be
                      (number of sentences, embedding dimension), where the embedding dimension is typically 768
                      for BERT-base.
    """
    if isinstance(corpus, str):
        corpus = [corpus]

    all_embeddings = []

    # Process in batches
    for i in range(0, len(corpus), batch_size):
        print("-" * 100)
        print(f"treating batch {i} to {i + batch_size}.\n")
        batch = corpus[i : i + batch_size]

        # Tokenize the batch of sentences
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        # Compute embeddings without gradient calculation (for efficiency)
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the [CLS] token embedding (first token)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]

        # Append batch embeddings to the list
        all_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings


def reduce_dimensionality(
    n: int,
    embeddings: torch.Tensor,
    labels: pd.Series = None,
    topics=None,
    reduction_type: str = "PCA",
) -> pd.DataFrame:
    """
    Reduces the dimensionality of the input embeddings using either Principal Component Analysis (PCA) or UMAP.

    This function takes high-dimensional embeddings and reduces them to a specified number of components
    using PCA or UMAP. Optionally, it can associate the reduced embeddings with provided labels or topics for better interpretability.

    Args:
        n (int): The number of components to reduce the embeddings to.
        embeddings (torch.Tensor): The input embeddings to reduce. It should be a 2D tensor of shape
                                   (num_samples, num_features).
        labels  (pd.Series, optional): A pandas Series containing labels corresponding to the embeddings.
                                       If provided, it will be added to the output DataFrame.
        topics (list, optional): An optional list of topics for test articles. Used to create index labels
                                 if `labels` are not provided.
        reduction_type (str): The type of dimensionality reduction to apply. Can be 'PCA' or 'UMAP'.
                              Defaults to 'PCA'.

    Example:
        # Reduce to 2 components using PCA
        reduced_df = reduce_dimensionality(2, embeddings, labels, reduction_type='PCA')

        # Reduce to 2 components using UMAP
        reduced_df = reduce_dimensionality(2, embeddings, labels, reduction_type='UMAP')
    """
    # Convert the embeddings to a NumPy array for PCA processing
    embeddings = embeddings.cpu().numpy()

    if reduction_type == "PCA":
        # Initialize and fit PCA
        reducer = PCA(n_components=n)
    elif reduction_type == "UMAP":
        # Initialize and fit UMAP
        reducer = umap.UMAP(n_components=n, random_state=42)
    else:
        raise ValueError(
            f"Invalid reduction type: {reduction_type}. Choose 'PCA' or 'UMAP'."
        )

    # Fit the selected reduction model and transform the embeddings
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create a DataFrame for the reduced embeddings
    reduced_embeddings = pd.DataFrame(
        reduced_embeddings,
        columns=[f"{reduction_type}" + str(i + 1) for i in range(n)],
    )

    # Add index based on labels or topics
    if labels is None:
        reduced_embeddings["index"] = [f"Unknown_{topic}" for topic in topics]
    else:
        reduced_embeddings["Labels"] = labels
        reduced_embeddings["index"] = [
            "AI-related" if labels[i] == 1 else "Not-AI-related"
            for i in range(reduced_embeddings.shape[0])
        ]

    # Set the 'index' column as the DataFrame index
    reduced_embeddings.set_index("index", inplace=True)

    return reduced_embeddings


def plot_clusters_2d(reduction_type: str, reduced_embeddings: pd.DataFrame) -> None:
    """
    Plots clusters in 2D using reduced embeddings and corresponding labels.

    This function generates a scatter plot to visualize the clustering of article embeddings in a
    2D space after dimensionality reduction. Each point represents an article, colored by its
    corresponding cluster label.

    Args:
        reduced_embeddings (pd.DataFrame): A DataFrame with shape (n_samples, 2) containing the
                                            reduced embeddings. The index should represent the cluster
                                            labels, which will be used for coloring the points.

    Returns:
        None: The function saves the plot as a PNG file.
    """
    plt.figure(figsize=(14, 10))

    # Create a scatter plot
    scatter = sns.scatterplot(
        x=f"{reduction_type}1",
        y=f"{reduction_type}2",
        hue=reduced_embeddings.index,
        palette=sns.color_palette("hsv", len(set(reduced_embeddings.index))),
        data=reduced_embeddings,
        legend="full",
    )

    # Annotate points with their corresponding labels
    for i in range(len(reduced_embeddings)):
        plt.text(
            reduced_embeddings[f"{reduction_type}1"].iloc[i],
            reduced_embeddings[f"{reduction_type}2"].iloc[i],
            reduced_embeddings.index[i],
            fontsize=9,
            color="black",
            ha="right",
            va="bottom",
        )

    # Customizing the legend
    handles, _ = scatter.get_legend_handles_labels()
    custom_labels = list(set(reduced_embeddings.index))
    plt.legend(handles=handles, labels=custom_labels, title="Clusters")

    plt.title(f"Article Embeddings Clustering with {reduction_type}")
    plt.xlabel(f"{reduction_type}1")
    plt.ylabel(f"{reduction_type}2")

    os.makedirs(OUTPUT_VISUALIZATION_DIR_2D, exist_ok=True)

    # Save the plot as a PNG file
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{OUTPUT_VISUALIZATION_DIR_2D}/{reduction_type}_classification_result_{formatted_time}.png"

    plt.savefig(filename, format="png")


def plot_clusters_3d(
    reduction_type: str, reduced_embeddings: pd.DataFrame, labels: pd.Series
) -> None:
    """
    Plots clusters in 3D using reduced embeddings and corresponding labels.

    This function generates a 3D scatter plot to visualize the clustering of article embeddings in a
    3D space after dimensionality reduction. Each point represents an article, colored by its
    corresponding cluster label.

    Args:
        reduced_embeddings (pd.DataFrame): A DataFrame with shape (n_samples, 3) representing the
                                            reduced embeddings in 3D space. The columns should be labeled
                                            as 'PCA1', 'PCA2', and 'PCA3'.
        labels (pd.Series): A Series of cluster labels corresponding to each embedding. This can include
                            categorical or numerical labels.

    Returns:
        None: The function saves the plot as a PNG file.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")  # Create a 3D subplot

    # Plot the embeddings
    scatter = ax.scatter(
        reduced_embeddings[f"{reduction_type}1"],
        reduced_embeddings[f"{reduction_type}2"],
        reduced_embeddings[f"{reduction_type}3"],
        c=labels,
        cmap="viridis",
        s=50,
    )

    # Add colorbar for clusters
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Cluster")

    # Add annotations (True Class) to the points
    for i in range(len(reduced_embeddings)):
        ax.text(
            reduced_embeddings[f"{reduction_type}1"].iloc[i],
            reduced_embeddings[f"{reduction_type}2"].iloc[i],
            reduced_embeddings[f"{reduction_type}3"].iloc[i],
            reduced_embeddings.index[i],
            fontsize=9,
            color="black",
            ha="right",
            va="bottom",
        )

    # Labels and title
    ax.set_title(f"3D Article Embeddings Clustering with {reduction_type}.")

    ax.set_xlabel(f"{reduction_type} 1")
    ax.set_ylabel(f"{reduction_type} 2")
    ax.set_zlabel(f"{reduction_type} 3")

    # Legend
    handles, _ = scatter.legend_elements()
    custom_labels = list(set(reduced_embeddings.index))
    ax.legend(handles, custom_labels, title="Clusters", loc="upper right")

    os.makedirs(OUTPUT_VISUALIZATION_DIR_3D, exist_ok=True)

    # Save the figure with a timestamp
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{OUTPUT_VISUALIZATION_DIR_3D}/{reduction_type}_classification_result_{formatted_time}.png"

    plt.savefig(filename, format="png")


def plot_clusters_interactive_3d(
    reduction_type: str, reduced_embeddings: pd.DataFrame, labels: pd.Series
) -> None:
    """
    Creates an interactive 3D scatter plot for visualizing clusters using Plotly.

    This function generates an interactive scatter plot of article embeddings in 3D space, colored by cluster labels.
    Users can hover over points to see more details about each embedding.

    Args:
        reduced_embeddings (pd.DataFrame): A DataFrame with shape (n_samples, 3) representing the reduced embeddings in 3D space.
                                            The index should contain identifiers (e.g., article IDs).
        labels (pd.Series): A Series or array of cluster labels (0 or 1) corresponding to each embedding.
                            These labels are used to color the points in the scatter plot.

    Returns:
        None: The function displays the plot interactively.
    """
    # Create a unique color for each label (0 and 1)
    color_map = {0: "blue", 1: "orange"}

    # Create a scatter plot using Plotly
    fig = go.Figure()

    # Add a trace for each cluster
    for label in [0, 1]:  # Since labels are binary (0 and 1)
        label_mask = labels == label
        fig.add_trace(
            go.Scatter3d(
                x=reduced_embeddings[f"{reduction_type}1"][label_mask],
                y=reduced_embeddings[f"{reduction_type}2"][label_mask],
                z=reduced_embeddings[f"{reduction_type}3"][label_mask],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color_map[label],  # Color by cluster label
                    opacity=0.8,
                ),
                name=f"Cluster: {'AI-related' if label == 1 else 'Not-AI-related'}",  # Use the label for the legend
                text=[
                    f"True label: {idx}" for idx in reduced_embeddings.index[label_mask]
                ],  # Index as text annotation
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Interactive 3D Article Embeddings Clustering with {reduction_type}",
        scene=dict(
            xaxis_title=f"{reduction_type} 1",
            yaxis_title=f"{reduction_type} 2",
            zaxis_title=f"{reduction_type} 3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title="Clusters",
    )

    os.makedirs(OUTPUT_VISUALIZATION_DIR_3D_INTERACTIVE, exist_ok=True)

    # Save the figure with a timestamp
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    fig.write_html(
        f"{OUTPUT_VISUALIZATION_DIR_3D_INTERACTIVE}/{reduction_type}_interactive_plot_{formatted_time}.html"
    )


def analyze_and_see_clusters(
    reduction_type: str, df_reduced_embeddings: pd.DataFrame
) -> None:
    """
    This function analyzes reduced embeddings data, finds the optimal number of clusters using the Elbow method,
    and visualizes the clusters in a 3D scatter plot.

    Parameters:
    df_reduced_embeddings : pd.DataFrame
        A DataFrame containing PCA-reduced data, with the first three principal components (f"{reduction_type}1", f"{reduction_type}2", f"{reduction_type}3")
        and a "Labels" column for true labels (if available).
    """
    # Use KElbowVisualizer to find the optimal number of clusters
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 10))
    visualizer.fit(df_reduced_embeddings.drop(columns=["Labels"]))
    optimal_clusters = visualizer.elbow_value_  # Get the optimal number of clusters
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Run k-Means clustering on the combined PCA-reduced data
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_reduced_embeddings)

    df_reduced_embeddings["Clusters"] = clusters

    # Create an interactive 3D scatter plot using Plotly
    fig = go.Figure()

    # Add scatter plot for each cluster
    for cluster in range(optimal_clusters):
        cluster_data = df_reduced_embeddings[
            df_reduced_embeddings["Clusters"] == cluster
        ]
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data[f"{reduction_type}1"],
                y=cluster_data[f"{reduction_type}2"],
                z=cluster_data[f"{reduction_type}3"],
                mode="markers",
                marker=dict(size=6),
                name=f"Cluster {cluster}",
                text=list(cluster_data.index),  # Display category on hover
                hoverinfo="text",
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{reduction_type}1",
            yaxis_title=f"{reduction_type}2",
            zaxis_title=f"{reduction_type}3",
        ),
        title=f"Interactive 3D {reduction_type} Projection with Clusters",
        legend_title="Clusters",
    )

    os.makedirs(OUTPUT_VISUALIZATION_DIR_3D_INTERACTIVE, exist_ok=True)

    # Save the figure with a timestamp
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    fig.write_html(
        f"{OUTPUT_VISUALIZATION_DIR_3D_INTERACTIVE}/{reduction_type}_cluster_analysis_{formatted_time}.html"
    )


def scrape_arxiv(query="AI", max_results=100):
    """
    Args:
        query (str, optional): _description_. Defaults to 'AI'.
        max_results (int, optional): _description_. Defaults to 1000.

    Returns:
        text_articles: list of text articles scaped

    """
    url = "http://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = xmltodict.parse(response.content)

        # Parse out the individual papers
        papers = data["feed"]["entry"]

        # Extract title, summary, and other details for each paper
        text_articles = []

        for paper in papers:
            text_articles.append(paper["summary"])

        return text_articles
    else:
        return None


def evaluate_logistic_regression(df_embeddings, cv_folds=10):
    """
    Evaluates a logistic regression model using cross-validation and plots a learning curve.

    Parameters:
    df_embeddings (DataFrame): The dataset containing feature embeddings and labels.
    cv_folds (int): Number of cross-validation folds (default is 10).

    Returns:
    None
    """
    # Separate features and labels
    X = df_embeddings.drop(columns=["Labels"])
    y = df_embeddings["Labels"]

    # Initialize logistic regression model
    model = LogisticRegression()

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
    print("Cross-validation accuracy scores:", scores)
    print("Mean accuracy:", scores.mean())

    # Save cross-validation scores to a CSV file
    cv_results = pd.DataFrame({"Fold": range(1, cv_folds + 1), "Accuracy": scores})
    cv_results["Mean Accuracy"] = scores.mean()  # Adding mean accuracy to the DataFrame
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(OUTPUT_DIR_MODEL, exist_ok=True)
    cv_results.to_csv(f"{OUTPUT_DIR_MODEL}/cv_results_{timestamp}.csv", index=False)

    # Learning curve analysis
    stratified_kfold = StratifiedKFold(n_splits=cv_folds)
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=stratified_kfold,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    # Calculate mean and standard deviation of scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Save learning curve data to a CSV file
    learning_curve_data = pd.DataFrame(
        {
            "Train Size": train_sizes,
            "Train Score Mean": train_scores_mean,
            "Train Score Std": train_scores_std,
            "Test Score Mean": test_scores_mean,
            "Test Score Std": test_scores_std,
        }
    )
    learning_curve_data.to_csv(
        f"{OUTPUT_DIR_MODEL}/learning_curve_{timestamp}.csv", index=False
    )

    # Plot learning curve
    plt.figure(figsize=(14, 8))
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.title("Learning Curve for Logistic Regression")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR_MODEL}/learning_curve_{timestamp}.png")


def main():
    # Initialize model and tokenizer.
    print("0. Initializing model and tokenizer...\n")

    model = BertModel.from_pretrained(MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    # Step 1 : Get all ai-related articles.
    print("1. Getting all AI-related articles...\n")

    ai_article_info = get_article_url(search_term="AI", label=1, max_urls=MAX_URLS)

    # Step 2 : Get all non-ai-related articles.
    print("2. Getting all non-AI-related articles...\n")

    non_ai_article_info = get_article_url(
        search_term="Not-AI", label=0, max_urls=MAX_URLS
    )

    # Step 3 : Concatenate results.
    print("3. Concatenating results...\n")

    article_info = pd.concat([ai_article_info, non_ai_article_info], axis=0)

    # Step 4 : Get the text from articles.
    print("4. Getting the text from articles...\n")

    get_text_from_links(article_info)

    # Drop null articles
    print("Dropping null articles...\n")

    article_info = article_info.dropna()

    # Step 5 : Embedding of the text using BERT.
    print("5. Embedding of the text using BERT...\n")

    embeddings = get_bert_embedding(model, tokenizer, list(article_info["Texts"]))

    # Step 6 : Reduce dimensionalities.
    print("6. Reducing dimensionalities...\n")

    pca_reduced_embeddings = reduce_dimensionality(
        3, embeddings, labels=list(article_info["Labels"]), reduction_type="PCA"
    )
    umap_reduced_embeddings = reduce_dimensionality(
        3, embeddings, labels=list(article_info["Labels"]), reduction_type="UMAP"
    )

    # Step 7 : Visualizations.
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    article_info.to_excel(
        f"{OUTPUT_DIR_EXTRACTED_DATA}/extracted_content_{formatted_time}.xlsx"
    )

    print("7. Visualizations...\n")

    plot_clusters_2d(reduction_type="PCA", reduced_embeddings=pca_reduced_embeddings)
    plot_clusters_2d(reduction_type="UMAP", reduced_embeddings=umap_reduced_embeddings)

    plot_clusters_3d(
        reduction_type="PCA",
        reduced_embeddings=pca_reduced_embeddings,
        labels=pca_reduced_embeddings["Labels"],
    )
    plot_clusters_3d(
        reduction_type="UMAP",
        reduced_embeddings=umap_reduced_embeddings,
        labels=umap_reduced_embeddings["Labels"],
    )

    # Interactive 3d plot.
    plot_clusters_interactive_3d(
        reduction_type="PCA",
        reduced_embeddings=pca_reduced_embeddings,
        labels=pca_reduced_embeddings["Labels"],
    )
    plot_clusters_interactive_3d(
        reduced_embeddings=umap_reduced_embeddings,
        labels=umap_reduced_embeddings["Labels"],
        reduction_type="UMAP",
    )

    # Step 8 : Logistic Regression.
    print("8. Logistic Regression...\n")

    df_embeddings = pd.DataFrame(
        embeddings.numpy(), columns=[f"col{i}" for i in range(embeddings.shape[1])]
    )
    df_embeddings["Labels"] = list(article_info["Labels"])
    df_embeddings["index"] = [
        "AI-related" if df_embeddings["Labels"][i] == 1 else "Not-AI-related"
        for i in range(embeddings.shape[0])
    ]
    df_embeddings.set_index("index", inplace=True)

    # Step 9 : Training the Logregression model.
    print("9. Training the Logregression model...\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df_embeddings.drop(columns=["Labels"]),
        df_embeddings["Labels"],
        test_size=0.2,
        random_state=42,
    )

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Step 10 : Evaluation.
    print("10. Evaluation...\n")

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score = {accuracy}\n")

    report = classification_report(y_test, y_pred)

    report_file = os.path.join(
        OUTPUT_DIR_MODEL, f"metrics_log_reg_{formatted_time}.txt"
    )
    with open(report_file, "w") as f:
        f.write(f"Accuracy score: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Metrics saved to: {report_file}\n")

    # K Fold Cross Validation.
    print("K Fold Cross Validation...\n")

    evaluate_logistic_regression(df_embeddings)

    # Step 11 : Manuel test.
    print("11. Manuel test...\n")

    test_ai_texts = scrape_arxiv()
    test_ai_topics = ["AI"] * len(test_ai_texts)
    test_ai_labels = [1] * len(test_ai_texts)

    test_non_ai_labels = [0] * len(TEST_NON_AI_TEXTS)

    # Concatenating info
    print("Concatenating info...\n")

    test_article_texts = test_ai_texts + TEST_NON_AI_TEXTS
    test_topics = test_ai_topics + TEST_NON_AI_TOPICS
    test_labels = test_ai_labels + test_non_ai_labels

    test_article_info = pd.DataFrame(
        {"cleaned_text": [cleaning_text(text) for text in test_article_texts]},
        index=test_topics,
    )

    # Embedding
    print("Embedding...\n")

    test_embeddings = get_bert_embedding(model, tokenizer, test_article_texts)

    # Tests
    print("Tests...\n")
    y_test = test_labels
    y_pred = classifier.predict(
        pd.DataFrame(
            test_embeddings,
            columns=[f"col{i}" for i in range(test_embeddings.shape[1])],
        )
    )

    for topic, prediction in zip(test_topics, y_pred):
        print("-" * 100)
        print(f"'{topic}' is {'AI-related' if prediction == 1 else 'Not AI-related'}\n")

    test_article_info["y_test"] = y_test
    test_article_info["y_pred"] = y_pred

    test_article_info.to_excel(
        f"{OUTPUT_DIR_EXTRACTED_DATA}/test_articles_results_{formatted_time}.xlsx"
    )

    # Tests results
    print("Tests results...\n")

    report = classification_report(y_test, y_pred)

    report_file = os.path.join(
        OUTPUT_DIR_MODEL, f"test_metrics_log_reg_{formatted_time}.txt"
    )
    with open(report_file, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)

    # Step 12: Save the model
    print("Saving the model...\n")

    model_pkl_file = os.path.join(OUTPUT_DIR_MODEL, "ai_classifier_model.pkl")
    with open(model_pkl_file, "wb") as file:
        pickle.dump(classifier, file)


if __name__ == "__main__":
    main()
