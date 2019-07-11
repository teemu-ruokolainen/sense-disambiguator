#!/usr/bin/env python
"""
Reads Finnish text (, tokenizes), POS tags and lemmatizes, and assigns
word tokens the most frequent.

Run with:
$ preprocess.py path_to_config_file
"""
from __future__ import division
import sys
import random
import logging
import os
from collections import Counter
from typing import List, Dict, Tuple
import numpy
import subprocess
import re
import argparse
import operator
from io import StringIO

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset

import pandas

# Logger: Write simultaneously to console and log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('/app/logs/', 'disambiguate_sense')),
#        logging.FileHandler("{0}/{1}.log".format('/app/logs/', 'preprocess')),
        logging.StreamHandler()
    ])
LOGGER = logging.getLogger()

"""
Map ftb-label POS tags to Wordnet POS tags
"""
posmap = {
    'ADJECTIVE' : 'a',
    'ADV' : 'r',
    'NOUN' : 'n',
    'VERB' : 'v',
}

def filter_pos(synset_list: List[Synset], pos: str) -> List[Synset]:
    res_synset_list = []
    for synset in synset_list:
        if synset.pos() == 's' and pos == 'a':
            res_synset_list.append(synset)
        elif synset.pos() == pos:
            res_synset_list.append(synset)

    return res_synset_list

def get_synset_frequency(synset: Synset):
    """
    Get synset frequency in Brown
    """
    return sum([lemma.count() for lemma in synset.lemmas()])

def get_most_frequent_sense(synset_list: List[Synset]) -> Tuple[Synset, int, str]:
    if synset_list:
        synset_frequency_list = [(synset, get_synset_frequency(synset), synset.definition())
                                    for synset in synset_list]
        synset_frequency_list.sort(key=operator.itemgetter(1))
        synset, frequency, definition = synset_frequency_list[-1]
        return synset, frequency, definition

    return '-', '-', '-'

def disambiguate_senses(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Assign the most frequent sense according to the Brown corpus.
    Add columns 'synset', 'synset_Brown_frequency', and 'synset_definition'.
    """
    nltk.download('wordnet')
    nltk.download('omw')

    synset_column = []
    synset_brown_frequency_column = []
    synset_definition_column = []
    for _, row in df.iterrows():
        synset_list = []
        synset_brown_frequency_list = []
        synset_definition_list = []
        for lemma, pos in zip(row['lemma'], row['pos']):
            pos = posmap.get(pos, pos)
            if pos in ['n', 'v', 'a', 'r']:
                synsets = wordnet.synsets(lemma, lang='fin')
                # Remove synsets with incorrect POS
                synsets = filter_pos(synsets, pos)
                # Get the most frequent sense, Brown frequency, and definition
                synset, frequency, definition = get_most_frequent_sense(synsets)
                # Append
                synset_list.append(synset)
                synset_brown_frequency_list.append(frequency)
                synset_definition_list.append(definition)
            else:
                synset_list.append('-')
                synset_brown_frequency_list.append('-')
                synset_definition_list.append('-')
        synset_column.append(synset_list)
        synset_brown_frequency_column.append(synset_brown_frequency_list)
        synset_definition_column.append(synset_definition_list)

    df['synset'] = synset_column
    df['synset_Brown_frequency'] = synset_brown_frequency_column
    df['synset_definition'] = synset_definition_column

    return df

def call_ftb_label(text_in: str):
    """
    Call ftb-label in subprocess, feed text_in, read analysis, and return decoded
    strings in a list. Each list item corresponds to one analyzed token.
    """
    p1 = subprocess.Popen(['echo', text_in], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['ftb-label'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    out, err = p2.communicate()
    return out.decode().split('\n')

def format_text_for_ftblabel(df: pandas.DataFrame) -> str:
    """
    Format column 'token' as ftb-label input.
    """
    text_in = ''
    for _, row in df.iterrows():
        token_index = 0
        for token in row['token']:
            if token_index in row['sentence_start']:
                text_in += '\n\n{}'.format(token)
            else:
                text_in += '\n{}'.format(token)
            token_index += 1

    return text_in

def ftb_label(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Assign part-of-speech and lemma to tokenized text in column 'token'.
    Add columns 'lemma' and 'pos' with the same structure as 'token'.
    """
    text_in = format_text_for_ftblabel(df)
    out = call_ftb_label(text_in)

    row_index = 0
    num_tokens = [len(tokens) for tokens in df['token']]
    token_index = 0
    lemma_column = []
    lemma_list = []
    pos_column = []
    pos_list = []
    for line in out:
        if line:
            line = line.split('\t')
            lemma_list.append(line[2])
            pos_list.append(re.search(r'\[POS=(\w+)\]', line[3]).group(1))
            token_index += 1
        elif token_index == num_tokens[row_index]:
            lemma_column.append(lemma_list)
            pos_column.append(pos_list)
            token_index = 0
            row_index += 1
            lemma_list = []
            pos_list = []

    df['lemma'] = lemma_column
    df['pos'] = pos_column

    return df

def load_tokenizers() -> Tuple[nltk.tokenize.punkt.PunktSentenceTokenizer,
                               nltk.tokenize.casual.TweetTokenizer]:
    """
    Return NLTK sentence detector and tokenizer instances.
    """
    nltk.download('punkt', quiet=True)
    sent_detector = nltk.data.load('tokenizers/punkt/finnish.pickle')
    tknzr = TweetTokenizer()
    return sent_detector, tknzr

def tokenize(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Tokenize column 'text' with NLTK sentence and word tokenizer.
    Adds columns 'token' and 'sentence_start' to df.
    Each row in column 'token' contains a list of tokens.
    Each row in column 'sentence_start' contains a list of sentence start indexes.
    """
    LOGGER.info('Tokenize...')
    sent_detector, tokenizer = load_tokenizers()

    token_column = []
    sentence_start_column = []
    for text in df['text']:
        res_text = []
        sentence_start_indexes = []
        token_index = 0
        for sentence in sent_detector.tokenize(text):
            sentence_start_indexes.append(token_index)
            tokens = tokenizer.tokenize(sentence)
            res_text.extend(tokens)
            token_index += len(tokens)
        token_column.append(res_text)
        sentence_start_column.append(sentence_start_indexes)
    df['token'] = token_column
    df['sentence_start'] = sentence_start_column

    return df

def add_text(df: pandas.DataFrame, text: str) -> pandas.DataFrame:
    """
    Add text as a column  to df.
    Each line in the original file is a row in df.
    """
    df['text'] = text.split('\n')
    return df

def print_results(df: pandas.DataFrame) -> None:
    """
    Print resulting word senses
    """
    print('token\tlemma\tpos\tsynset\tBrown_frequency\tdefinition')
    for _, row in df.iterrows():
        for token,\
            lemma,\
            pos,\
            synset,\
            frequency,\
            definition in zip(row['token'],
                              row['lemma'],
                              row['pos'],
                              row['synset'],
                              row['synset_Brown_frequency'],
                              row['synset_definition']):
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(token,
                                                  lemma,
                                                  pos,
                                                  synset,
                                                  frequency,
                                                  definition))
        print()

def parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description='Disambiguate word senses \
using the most frequent sense (Brown corpus) criterion.')
    parser.add_argument('--notokenize',
                        action='store_true',
                        default=False,
                        help='set to True to skip tokenization')
    return parser.parse_args(argv[1:])

def main(argv: List[str]) -> int:
    """
    Main function for most frequent sense disambiguation.
    """
    args = parse_args(argv)
    df = pandas.DataFrame()
    df = add_text(df, sys.stdin.read())
    if not args.notokenize:
        df = tokenize(df)
    df = ftb_label(df)
    df = disambiguate_senses(df)

    # Write to stdout
    print_results(df)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
