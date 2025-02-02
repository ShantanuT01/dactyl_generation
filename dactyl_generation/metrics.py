from nltk import FreqDist
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
from dactyl_generation.constants import *
from tqdm import tqdm
import spacy
import textdescriptives as td
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("textdescriptives/information_theory")
def get_twitter_word_frequencies(tweets):
    tokenizer = TweetTokenizer()
    tokens = list()
    for tweet in tqdm(tweets):
        if not (tweet is None) and len(tweet.strip()) > 0:
            tokens.extend(tokenizer.tokenize(tweet))
    # stop_words = set(stopwords.words('english'))
    dist = FreqDist([token.lower() for token in tokens])
    df_dist = pd.DataFrame.from_dict(dist, orient='index')
    df_dist = df_dist.sort_values(0, ascending=False)
    df_dist = df_dist.rename(columns={0: "count"})
    df_dist[PX] = df_dist[COUNT] / df_dist[COUNT].sum()
    df_dist[ENTROPY] = -1 * df_dist[PX] * np.log(df_dist[PX])
    return df_dist

def get_tweet_per_word_perplexity(text, token_to_entropy):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    entropy_sum = 0
    for token in tokens:
        entropy_sum += token_to_entropy.get(token.lower(), 0)
    return entropy_sum/(len(tokens))

def compute_spacy_perplexity(text):
    doc = nlp(text)
    return td.extract_df(doc)


if __name__ == "__main__":
    '''
    all_tweets = pd.read_parquet("C:/Users/shant/Documents/DACTYL/datasets/tweets/538_english_tweets.parquet")
    df_dist = get_twitter_word_frequencies(all_tweets["content"].values)
    test_df = pd.read_csv("C:/Users/shant/Documents/DACTYL/datasets/tweets/release/non_adversarial_testing.csv")
    per_word_perplexities = list()
    token_to_entropy = pd.Series(df_dist[ENTROPY], index=df_dist.index)
    for text in tqdm(test_df["text"].values):
        per_word_perplexities.append(get_tweet_per_word_perplexity(text, token_to_entropy))
    test_df["pwp"] = per_word_perplexities
    test_df.to_csv("C:/Users/shant/Documents/DACTYL/datasets/tweets/release/non_adversarial_testing.csv",index=False)
    '''
    test_df = pd.read_csv("C:/Users/shant/Documents/DACTYL/datasets/tweets/release/non_adversarial_testing.csv")
    dfs = list()
    for text in tqdm(test_df["text"].values):
        dfs.append(compute_spacy_perplexity(text))
    test_stats = pd.concat(dfs)
    pd.concat([test_df, test_stats[["entropy","perplexity","per_word_perplexity"]]], axis=1).to_csv("C:/Users/shant/Documents/DACTYL/datasets/tweets/release/twitter_pwp_stats.csv")





