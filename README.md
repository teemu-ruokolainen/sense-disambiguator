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

The output consists of word token, word lemma, part-of-speech, assigned synset, the Brown corpus frequency of the synset (see [Wiki](https://github.com/teemu-ruokolainen/sense-disambiguator)), and the synset definition separated with a tab (\t):

>$ echo Tämä on hyvä esimerkki. | docker run -i teemuruokolainen/sense-disambiguator:latest\
>token	lemma	pos	synset	Brown_frequency	definition\
>Tämä	tämä	PRONOUN	-	-	-\
>on	olla	VERB	Synset('be.v.01')	10742	have the quality of being; (copula, used with an adjective or a predicate noun)\
>hyvä	hyvä	ADJECTIVE	Synset('good.a.01')	190	having desirable or positive qualities especially those suitable for a thing >specified\
>esimerkki	esimerkki	NOUN	Synset('example.n.01')	50	an item of information that is typical of a class or group\
>.	.	PUNCTUATION	-	-	-\

To disambiguate a collection of texts (e.g. sentences or documents), run:
```
$ cat text-file.txt | docker run -i teemuruokolainen/sense-disambiguator:latest
```
where `text-file.txt` should contain one text per row. Each text is separated by an empty row in the output.









