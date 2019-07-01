"""
Unit tests for disambiguate_sense.py
Run inside container with:
$ py.test -f /app/tests/test_disambiguate_sense.py -v
"""
import pytest

from src import disambiguate_sense
import nltk

from nltk.corpus.reader.wordnet import Synset


"""
Fixtures
"""

@pytest.fixture
def df_one_text_one_sentence():
    return pandas.DataFrame.from_dict({'text' : ['Tämä on hyvä esimerkki.']})

@pytest.fixture
def df_one_text_two_sentences():
    return pandas.DataFrame.from_dict({'text' : ['Tämä on hyvä esimerkki. Tämä on toinen.']})

@pytest.fixture
def df_two_texts_two_sentences():
    return pandas.DataFrame.from_dict({'text' : ['Tämä on hyvä esimerkki.', 'Tämä on toinen.']})



@pytest.fixture
def synset_pos_s():
    class MockSynset(object):
        def pos(self):
            return 's'
    return MockSynset()

@pytest.fixture
def synset_pos_not_s():
    class MockSynset(object):
        def pos(self):
            return 'not_s'
    return MockSynset()

@pytest.fixture
def synset_with_two_lemmas_with_counts_1_and_2():
    class MockLemma(object):
        def __init__(self, freq):
            self.freq = freq
        def count(self):
            return self.freq

    class MockSynset(object):
        def lemmas(self):
            return [MockLemma(1), MockLemma(2)]
        def definition(self):
            return ''

    mock_synset = MockSynset()

    return mock_synset

@pytest.fixture
def synset_with_two_lemmas_with_counts_3_and_4():
    class MockLemma(object):
        def __init__(self, freq):
            self.freq = freq
        def count(self):
            return self.freq

    class MockSynset(object):
        def lemmas(self):
            return [MockLemma(3), MockLemma(4)]
        def definition(self):
            return 'some_definition'

    return MockSynset()


"""
Tests
"""

def test_filter_pos(synset_pos_s, synset_pos_not_s):
    assert not disambiguate_sense.filter_pos([], 'arbitrary_pos')
    assert not disambiguate_sense.filter_pos([synset_pos_not_s], 'not_a')
    assert not disambiguate_sense.filter_pos([synset_pos_s], 'not_a')
    assert not disambiguate_sense.filter_pos([synset_pos_not_s], 'a')
    assert len(disambiguate_sense.filter_pos([synset_pos_s], 'a')) == 1
    assert len(disambiguate_sense.filter_pos([synset_pos_not_s], 'not_s')) == 1

def test_get_synset_frequency(synset_with_two_lemmas_with_counts_1_and_2,
                              synset_with_two_lemmas_with_counts_3_and_4):
    assert disambiguate_sense.get_synset_frequency(synset_with_two_lemmas_with_counts_1_and_2) == 3
    assert disambiguate_sense.get_synset_frequency(synset_with_two_lemmas_with_counts_3_and_4) == 7

def test_get_most_frequent_sense(synset_with_two_lemmas_with_counts_1_and_2,
                                 synset_with_two_lemmas_with_counts_3_and_4):
    assert disambiguate_sense.get_most_frequent_sense([]) == ('-', '-', '-')
    assert disambiguate_sense.get_most_frequent_sense([synset_with_two_lemmas_with_counts_1_and_2,
                                                       synset_with_two_lemmas_with_counts_3_and_4]) \
                                                    == (synset_with_two_lemmas_with_counts_3_and_4, 7, 'some_definition')

def test_disambiguate_senses():
    pass

def test_call_ftb_label():
    pass

def test_format_text_for_ftblabel():
    pass

def test_ftb_label():
    pass

def test_load_tokenizers():
    pass

def test_tokenize(df_one_text_one_sentence):
    df_res = preprocess.tokenize(df_one_text_one_sentence)
    assert 'token' in df_res
    assert 'sentence_index' in df_res
    assert len(df_res) == 1

    df_res = preprocess.tokenize(df_one_text_two_sentences)
    assert 'token' in df_res
    assert 'sentence_index' in df_res
    assert len(df_res) == 1

    df_res = preprocess.tokenize(df_two_texts_two_sentences)
    assert 'token' in df_res
    assert 'sentence_index' in df_res
    assert len(df_res) == 2

def test_add_text():
    pass

def test_parse_args():
    args = configparse.parse_args(['arbitrary-script.py', 'config.json'])
    assert type(args) == argparse.Namespace
    assert args.config_file == 'config.json'
