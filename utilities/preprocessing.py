import re
import emoji
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from functools import reduce


class TwitterReader:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.sentiment_mapping = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }
        self.tweets = {sentiment: [] for sentiment in self.sentiment_mapping}
        self._load_tweets()

    def _load_tweets(self):
        for sentiment in self.sentiment_mapping:
            file_path = self.folder_path / f"{sentiment}"
            if file_path.exists():
                with file_path.open('r', encoding='utf-8') as file:
                    self.tweets[sentiment] = [line.strip() for line in file]
            else:
                raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    def get_all_tweets(self, return_labels=False):
        """Return the entire list of tweets. Optionally return labels."""
        all_tweets = []
        labels = []
        for sentiment, tweets in self.tweets.items():
            all_tweets.extend(tweets)
            if return_labels:
                labels.extend([self.sentiment_mapping[sentiment]] * len(tweets))
        
        if return_labels:
            return all_tweets, labels
        return all_tweets


def stratified_train_eval_test_split(X, y, feature, eval_test_size=0.2, second_split=0.5, random_state=42):
    """
    Splits data into stratified train, evaluation, and test sets based on labels and an additional feature.

    Args:
        X (list): The list of data items (e.g., tweets).
        y (list): The list of labels corresponding to the data items.
        feature (list): The list of combined features (e.g., label and length category)
        eval_test_size (float, optional): The proportion of the dataset to include in the eval+test set.
        second_split (float, optional): The proportion of the combined eval+test dataset to include in the test set
        random_state (int, optional): The seed used by the random number generator.

    Returns:
        tuple: Stratified train, evaluation, and test sets (X_train, X_eval, X_test, y_train, y_eval, y_test)
    """
    # Define the StratifiedShuffleSplit for the initial split
    split = StratifiedShuffleSplit(n_splits=1, test_size=eval_test_size, random_state=random_state)

    # Perform the first split into train and auxilliary sets
    for train_index, eval_test_index in split.split(X, feature):
        X_train = [X[i] for i in train_index]
        X_eval_test = [X[i] for i in eval_test_index]
        y_train = [y[i] for i in train_index]
        y_eval_test = [y[i] for i in eval_test_index]
        combined_eval_test = [feature[i] for i in eval_test_index]

    # Define another StratifiedShuffleSplit for the auxilliary set
    split_aux = StratifiedShuffleSplit(n_splits=1, test_size=second_split, random_state=random_state)

    # Perform the second split into evaluation and test sets
    for eval_index, test_index in split_aux.split(X_eval_test, combined_eval_test):
        X_eval = [X_eval_test[i] for i in eval_index]
        X_test = [X_eval_test[i] for i in test_index]
        y_eval = [y_eval_test[i] for i in eval_index]
        y_test = [y_eval_test[i] for i in test_index]

    return X_train, X_eval, X_test, y_train, y_eval, y_test

################CLEANING FUNCTIONS#########################
 
def remove_urls(text):
    return re.sub(r'http\S+|www\S', '', text)

def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

def find_mentions(text):
    mention_pattern = re.compile(r'@\w+')
    return mention_pattern.findall(text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def replace_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def clean_data(text):
     text=reduce(lambda x, func: func(x), [replace_multiple_spaces,remove_urls,remove_mentions],text)
     return text.strip()


