# Sense Disambiguator for Finnish

A word-sense disambiguator for Finnish using the FinnWordNet synsets.

The app reads in plain Finnish text, tokenizes it, and assigns the word tokens a synset (concept) from the FinnWordNet. For more information on the analysis pipeline, see the [Wiki](https://github.com/teemu-ruokolainen/sense-disambiguator/wiki).

The app is available as a [Docker](https://www.docker.com/) image so it can be run without installing the dependencies, see below. 

## Docker Image

The ready-made Docker image is published in the [Docker Hub](https://hub.docker.com/) where Docker can find it automatically.

To disambiguate Finnish text, run:
```
$ echo Tämä on hyvä esimerkkilause. | docker run -i teemuruokolainen/sense-disambiguator:latest
```
The output consists of word token, word lemma, part-of-speech, assigned synset, the Brown corpus frequency of the synset ([Wiki](https://github.com/teemu-ruokolainen/sense-disambiguator/wiki)), and the synset definition:
>$ echo Tämä on hyvä esimerkkilause. | docker run -i teemuruokolainen/sense-disambiguator:latest
>
>
>

To disambiguate a collection of texts (e.g. sentences or documents), run:
```
$ cat text-file.txt | docker run -i teemuruokolainen/sense-disambiguator:latest
```
where `text-file.txt` should contain one text per row. Each text is separated by an empty row in the output.






# Wiki



# Word-sense disambiguation and FinnWordNet

[Word-sense disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation) (WSD) considers the problem of identifying which sense of a word is used in a sentence. 
For example, consider the Finnish sentence  

*Tietokonetta voi ohjata hiirellä. (Computer can be controlled with a mouse)*

where *hiiri (mouse)* in this context refers to a [computer mouse](https://en.wikipedia.org/wiki/Computer_mouse) instead of the [small rodent](https://en.wikipedia.org/wiki/Mouse).

In this app, we perform the disambiguation utilizing the [FinnWordNet](http://www.ling.helsinki.fi/en/lt/research/finnwordnet/) lexical database which is a derivative of the [Princeton WordNet](https://wordnet.princeton.edu/).
FinnWordNet is a database of words (nouns, verbs, adjectives and adverbs) grouped by meaning into synonym groups representing concepts. 
For example, the synsets given in the FinnWordNet for *hiiri* are

*mouse.n.01*: any of numerous small rodents typically resembling diminutive rats having pointed snouts and small ears on elongated bodies with slender usually hairless tails

*church_mouse.n.01*: a fictional mouse created by Lewis Carroll

*mouse.n.04*: a hand-operated electronic device that controls the coordinates of a cursor on your computer screen as you move it around on a pad; on the bottom of the device is a ball that rolls on the surface of the pad

the *n* in the synset labels refers to the part-of-speech *noun*. 
(The labels and definitions are obtained using the [WordNet interface](http://www.nltk.org/howto/wordnet.html) of the [NLTK](https://www.nltk.org/) Python library.)

# Analysis pipeline

Given the availability of the FinnWordNet synsets using the NLTK toolkit, our WSD pipeline for input text is as follows:

1. Tokenize the input (optional)
2. Lemmatize and part-of-speech (POS) tag the tokenized input using the [ftb-label](https://github.com/mpsilfve/FinnPos) toolkit 
3. For each token in input:
  3.1. Obtain all possible synsets using the NLTK toolkit
  3.2. Filter out synsets which disagree with the POS predicted by ftb-label
  3.3. For remaining synsets: Choose the synset with highest frequency (count) in the [Brown corpus](http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM). (The Brown corpus counts for each synset can again be obtained using the NLTK toolkit.)

The item 3.3. means that our disambiguator implements the *most frequent sense* criterion. 
In other words, each word form is always assigned the sense most often observed in the manually annotated Brown corpus *given* the part-of-speech assigned by our POS tagger.
The reason for this choice is two-fold:

1. The most frequent sense approach is a [strong baseline](https://pdfs.semanticscholar.org/a0b3/4741522d1584d66befd2e6cacd2680e550fa.pdf) for the WSD task  
2. Outperforming the most frequent sense approach using [machine learning methodology](https://arxiv.org/pdf/1606.03568.pdf) require a large training corpus with manually assigned senses for each word token. This kind of corpus doesn't currently exist for Finnish. 

# Evaluation

Future work.







