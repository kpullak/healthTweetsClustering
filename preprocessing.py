# -*- coding: utf-8 -*-
"""

Created on Thursday June 18 17:38:40 2020
@author: Krishna

This script will take in the combined text file
and does pre-processing, like removes all unwanted
information and retains only the tweets

"""


import re
import string


def pre_process_tweets(url):

    f = open(url, "r", encoding="ISO-8859-1")
    tweets_list = list(f)
    list_of_tweets = []

    for i in range(len(tweets_list)):

        # remove \n from the end after every sentence
        tweets_list[i] = tweets_list[i].strip('\n')

        # Remove the tweet id and timestamp
        tweets_list[i] = tweets_list[i][50:]

        # Remove any word that starts with the symbol @
        tweets_list[i] = " ".join(filter(lambda x: x[0] != '@', tweets_list[i].split()))

        # Remove any URL
        tweets_list[i] = re.sub(r"http\S+", "", tweets_list[i])
        tweets_list[i] = re.sub(r"www\S+", "", tweets_list[i])

        # remove colons from the end of the sentences (if any) after removing url
        tweets_list[i] = tweets_list[i].strip()
        tweet_len = len(tweets_list[i])
        if tweet_len > 0:
            if tweets_list[i][len(tweets_list[i]) - 1] == ':':
                tweets_list[i] = tweets_list[i][:len(tweets_list[i]) - 1]

        # Remove any hash-tags symbols
        tweets_list[i] = tweets_list[i].replace('#', '')

        # Convert every word to lowercase
        tweets_list[i] = tweets_list[i].lower()

        # remove punctuations
        tweets_list[i] = tweets_list[i].translate(str.maketrans('', '', string.punctuation))

        # trim extra spaces
        tweets_list[i] = " ".join(tweets_list[i].split())

        # convert each tweet from string type to as list<string> using " " as a delimiter
        list_of_tweets.append(tweets_list[i].split(' '))

    f.close()

    return list_of_tweets


def write_to_file(tweets):

    # with open('bbc_processed_tweets.txt', 'w', encoding="ISO-8859-1") as f:
    with open('combined_processed_tweets.txt', 'w', encoding="ISO-8859-1") as f:
        for item in tweets:
            item = ' '.join(item)
            f.write("%s\n" % item)
    f.close()


if __name__ == '__main__':

    data_url = 'result.txt'
    # data_url = 'Health-Tweets/bbchealth.txt'
    tweets = pre_process_tweets(data_url)
    write_to_file(tweets)
