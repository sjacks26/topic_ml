"""

This file creates functions to process text and extract features

"""
from . import Config
import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import map_tag
import pandas as pd
import re
from collections import Counter

tknzr = TweetTokenizer()


class Main(object):
    def get_data(platform=Config.platform):
        if platform == "BOTH":
            docs1 = pd.read_csv(Config.tw_input_data_file, quotechar='"', encoding="Latin1", keep_default_na=False)
            docs2 = pd.read_csv(Config.fb_input_data_file, quotechar='"', keep_default_na=False)
            docs1.columns = ['id', 'text', 'topics']
            docs2.columns = ['id', 'text', 'topics']
            docs = pd.concat([docs1, docs2])
            docs["text"] = docs["text"].str.replace("#", "")
            docs["text"] = docs["text"].str.replace("\n", "")
            docs.reset_index(drop=True, inplace=True) # Without this line, combined data will have replicated indices.

        elif platform == "TW":
            file = Config.tw_input_data_file
            docs = pd.read_csv(file, quotechar='"', encoding="Latin1", keep_default_na=False)
            docs.columns = ['id', 'text', 'topics']
            docs["text"] = docs["text"].str.replace("#", "")
            docs["text"] = docs["text"].str.replace("\n", "")

        elif platform == "FB":
            file = Config.fb_input_data_file
            docs = pd.read_csv(file, quotechar='"', encoding="Latin1", keep_default_na=False)
            docs.columns = ['id', 'text', 'topics']
            docs["text"] = docs["text"].str.replace("#", "")
            docs["text"] = docs["text"].str.replace("\n", "")

        return docs

    docs = get_data()

    def get_labels(docs):
        text = docs['topics']
        text = ",".join(text)
        labels = text.split(',')
        labels = list(filter(None, set(labels)))

        return labels

    labels = get_labels(docs)

    # If a stopword file is specified, use those stopwords. If not, use nltk stopwords.
    def get_stopwords():
        if Config.stopwords_file:
            path = Config.stopwords_file
            temp_stopwords = []
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                temp_stopwords.append(line.strip("\n"))
            print("stopwords: ", len(temp_stopwords))
        else:
            temp_stopwords = list(stopwords.words('english'))
            temp_stopwords.append("rt")
        return temp_stopwords

    stops = get_stopwords()

    def tokenize_docs(docs):
        texts = docs['text']
        tokens = []
        pos = []
        for text in texts:
            token_list = tknzr.tokenize(text)
            tokens.append(token_list)
            pos_tags1 = nltk.pos_tag(token_list)
            pos_tags = [map_tag('en-ptb', 'universal', tag) for word, tag in pos_tags1]
            pos.append(pos_tags)
        docs['tokens'] = tokens
        docs['pos'] = pos
        return docs

    docs = tokenize_docs(docs)

    def get_tagged_chunked(docs):
        texts = docs['text']
        chunks = []
        for text in texts:
            chunk = nltk.ne_chunk(nltk.pos_tag(tknzr.tokenize(text)))
            chunks.append(chunk)
        docs['chunks'] = chunks
        return docs

    docs = get_tagged_chunked(docs)

    def get_unigrams(docs, stops, pos=Config.pos_tags):
        def alpha_filter(w):
            pattern = re.compile('^[^a-z]+$')
            if (pattern.match(w)):
                return True
            else:
                return False

        unigrams = docs['tokens']
        unigram_list = []
        pos_tags = docs['pos']
        for u in unigrams:
            unigramz = [w.lower() for w in u]
            unigramz = [w for w in unigramz if not alpha_filter(w)]

            # This loop should filter out unigrams that aren't the POS specified in Config, if Config.pos_restriction is true
            if Config.pos_restriction:
                filt_unigrams1 = []
                for i in range(0, len(unigramz)):
                    if pos_tags[i] in pos:
                        filt_unigrams1.append(unigramz[i])
            else:
                filt_unigrams1 = unigramz
            filt_unigrams2 = [w for w in filt_unigrams1 if not w in stops]
            unigram_list.append(filt_unigrams2)
        docs['unigrams'] = unigram_list
        return docs

    docs = get_unigrams(docs, stops)

    def get_bigrams(docs, stops):
        def alpha_filter(w):
            pattern = re.compile('^[^a-z]+$')
            if (pattern.match(w)):
                return True
            else:
                return False

        tokens = docs['tokens']
        bigram_list = []
        for text in tokens:
            bigrams1 = list(nltk.bigrams(text))
            bigrams = []
            for b in bigrams1:
                if not (b[0] or b[1]) in stops:
                    if not alpha_filter(b[0]) and not alpha_filter(b[1]):
                        bigrams.append(b)
            bigram_list.append(bigrams)
        docs['bigrams'] = bigram_list
        return docs

    docs = get_bigrams(docs, stops)

    def get_pos_counts(docs):
        pos_tags = docs['pos']
        pos_count_list = []
        for tags in pos_tags:
            counts = dict(Counter(tags))
            pos_count_list.append(counts)
        docs['pos_counts'] = pos_count_list
        return docs

    docs = get_pos_counts(docs)

    # This isn't actually my feature file. It's partially processed data
    def write_partial_process(docs):
        feature_file = Config.input_data_file.replace(".csv", "_mid_process.csv")
        docs.to_csv(feature_file, index=False)

    # write_partial_process(docs)

    def get_vocab(docs, stops, u_limit=Config.num_unigrams, b_limit=Config.num_bigrams):
        def alpha_filter(w):
            pattern = re.compile('^[^a-z]+$')
            if (pattern.match(w)):
                return True
            else:
                return False

        text = docs['text']
        text = " ".join(text)
        text = text.lower()

        tokens = tknzr.tokenize(text)
        tokens1 = [w for w in tokens if not alpha_filter(w)]
        tokens2 = [w for w in tokens1 if not w in stops]
        if not Config.use_mentions:
            tokens3 = [w for w in tokens2 if not "@" in w]
        else:
            tokens3 = tokens2
        tokens4 = nltk.FreqDist(tokens3)
        tokens5 = list(tokens4.most_common(u_limit))
        unigrams = []
        for u, s in tokens5:
            unigrams.append(u)

        bigrams1 = list(nltk.bigrams(tokens))
        bigrams = []
        for b in bigrams1:
            if not (b[0] in stops or b[1] in stops):
                if not alpha_filter(b[0]) and not alpha_filter(b[1]):
                    if not Config.use_mentions:
                        if not "@" in (b[0] or b[1]):
                            b = ' '.join(b)
                            bigrams.append(str(b))
                    else:
                        b = ' '.join(b)
                        bigrams.append(str(b))
        scored_bigrams1 = nltk.FreqDist(bigrams)
        scored_bigrams2 = list(scored_bigrams1.most_common(b_limit))
        scored_bigrams = []
        for b, s in scored_bigrams2:
            scored_bigrams.append(b)

        return unigrams, scored_bigrams

    unigrams, bigrams = get_vocab(docs, stops)

    def write_feature_file(docs, unigrams=unigrams, bigrams=bigrams, labels=labels):
        try:
            os.mkdir(Config.output_dir)
        except FileExistsError:
            pass

        features = pd.DataFrame()
        features["message"] = docs["text"]
        features["topics"] = docs["topics"]
        for u in unigrams:
            u2 = "u_" + u
            features[u2] = False
            features.loc[features["message"].str.contains(u), u2] = True
        for b in bigrams:
            b2 = "b_" + b
            features[b2] = False
            features.loc[features["message"].str.contains(b), b2] = True

        if Config.pos_counts:
            features["pos_tags"] = docs["pos_counts"]
            unique_pos_tags = [x for row in features["pos_tags"] for x in row]
            unique_pos_tags = list(set(unique_pos_tags))
            for t in range(0, len(features.index)):
                for p in unique_pos_tags:
                    p2 = p + "_count"
                    try:
                        features.loc[t, p2] = int(features.loc[t, "pos_tags"][p])
                    except KeyError as e:
                        features.loc[t, p2] = 0
            features = features.drop(["pos_tags"], axis=1)

        for l in labels:
            features[l] = False
            features.loc[features["topics"].str.contains(l), l] = True
        features = features.drop(["topics"], axis=1)

        if Config.platform == "BOTH":
            feature_file = Config.comb_feature_file
        elif Config.platform == "TW":
            feature_file = Config.tw_feature_file
        elif Config.platform == "FB":
            feature_file = Config.fb_feature_file
        features.to_csv(feature_file, index=False)

    write_feature_file(docs)
