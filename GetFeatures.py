"""

This file creates functions to process text and extract features

"""
import Config # This is a local import
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import map_tag
import pandas as pd
import re
from collections import Counter
import os

tknzr = TweetTokenizer()


def main():
    def get_data(platform=Config.platform):
        """
        This function reads in data based on parameters specified in Config. It does some basic text processing (removing #, for example)
        The returned object is a Pandas dataframe containing 3 columns (message id, message text, message topics)
        """
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
            docs["text"] = docs["text"].str.replace("\n", "")

        return docs

    docs = get_data()

    def get_labels(docs):
        """
        This function gets a list of all the possible topic labels.
        The returned object is that list of unique labels
        """
        text = docs['topics']
        text = ",".join(text)
        labels = text.split(',')
        labels = list(filter(None, set(labels)))

        return labels

    labels = get_labels(docs)

    def get_stopwords():
        """
        This function sets up the list of stopwords to be used. If a stopword file is specified, use those stopwords. If not, use nltk stopwords plus "rt".
        The returned object is the list of stopwords.
        """
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
        """
        This function takes in the pandas dataframe containing 3 columns (id, text, and topic). It tokenizes the text column, creating a list of tokens for each message.
        It then tags the part of speech for each token using NLTK's universal tag set, creating a list of part of speech tags in the same order as the tokens.
        It adds a column with the lists of tokens and a column with the list of parts of speech.
        The returned object is a pandas dataframe with 5 columns (id, text, topic, tokens, and pos [part of speech tags]).
        """
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
        """
        This function takes in the pandas dataframe containing 5 columns (id, text, topic, tokens, and pos) and uses NLTK's named entity chunker to identify named entities.
        It does not use NE labels, just binary NE or not.
        For each chunk in a message, the function checks whether the chunk is a named entity.
         If it is, it appends the NE to a list, dropping the label. If not, it goes to the next chunk.
        The returned object is a pandas dataframe with 6 columns (id, text, topic, tokens, pos, and NE chunks).
        """
        texts = docs['text']
        chunks = []
        for text in texts:
            NEs = []
            chunk = nltk.ne_chunk(nltk.pos_tag(tknzr.tokenize(text)), binary=True)
            for c in chunk.subtrees():
                if c.label() == "NE":
                    NEs.append(list(c)[0][0])
            chunks.append(NEs)
        docs['ne_chunks'] = chunks
        return docs

    docs = get_tagged_chunked(docs)

    def get_unigrams(docs, stops, pos=Config.pos_tags):
        """
        This function takes in the pandas dataframe with 6 columns (id, text, topic, tokens, pos, and NE chunks) and creates a list of unigrams from the tokens column.
        It also converts all characters to lowercase.
        It also takes in the list of stopwords. It then uses the stopwords and an alphafilter to remove words that aren't meaningful and words that contain non-letter characters.
        ---> It also takes in a parameter from Config that is a list of desired POS tags. If Config.pos_restriction is True, it removes unigrams that are a POS that isn't in this list. This feature will be moved to the feature file function
        The returned object is a pandas dataframe with 7 columns (id, text, topic, tokens, pos, NE chunks, and unigrams)
        """
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
            unigramz = [w for w in unigramz if not w in stops]

            # This loop should filter out unigrams that aren't the POS specified in Config, if Config.pos_restriction is true. Move this to the feature file function
            if Config.pos_restriction:
                filt_unigrams1 = []
                for i in range(0, len(unigramz)):
                    if pos_tags[i] in pos:
                        filt_unigrams1.append(unigramz[i])
            else:
                filt_unigrams1 = unigramz
            unigram_list.append(filt_unigrams1)
        docs['unigrams'] = unigram_list
        return docs

    docs = get_unigrams(docs, stops)

    def get_bigrams(docs, stops):
        """
        This function takes in the pandas dataframe with 7 columns (id, text, topic, tokens, pos, NE chunks, and unigrams).
        It creates a list of bigrams from the tokens column.
        It takes in the list of stopwords. It then uses the stopwords and an alphafilter to remove bigrams that contain any unigrams in the list of stopword or non-letter characters.
        The returned object is a pandas dataframe with 8 columns (id, text, topic, tokens, pos, NE chunks, unigrams, and bigrams).
        """
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
        """
        This function takes in the pandas dataframe with 8 columns (id, text, topic, tokens, pos, NE chunks, unigrams, and bigrams).
        Using the pos tags column, it counts the number of times each pos appears in each message.
        For each message, it creates a dict, with {pos_tag: count_of_pos_tag_in_message}. Thus, pos_count_list is a list of dictionary items.
        The returned object is a pandas dataframe with 9 columns (id, text, topic, tokens, pos, NE chunks, unigrams, bigrams, and pos counts).
        """
        pos_tags = docs['pos']
        pos_count_list = []
        for tags in pos_tags:
            counts = dict(Counter(tags))
            pos_count_list.append(counts)
        docs['pos_counts'] = pos_count_list
        return docs

    docs = get_pos_counts(docs)

    def write_partial_process(docs):
        """
        This function takes in the pandas dataframe with 9 columns (id, text, topic, tokens, pos, NE chunks, unigrams, bigrams, and pos counts).
        It writes that dataframe to a csv file.
        There is no returned object. This function isn't necessary to generate a feature file.
        """
        feature_file = Config.input_data_file.replace(".csv", "_mid_process.csv")
        docs.to_csv(feature_file, index=False)

    # write_partial_process(docs)

    def get_vocab(docs, stops, u_limit=Config.num_unigrams, b_limit=Config.num_bigrams):
        """
        This function takes in the pandas dataframe with 9 columns (id, text, topic, tokens, pos, NE chunks, unigrams, bigrams, and pos counts).
        It first combines the text from all the messages into one string, then tokenizes that string.
        It also takes in the stopwords limit and uses an alphafilter to remove tokens with non-letter characters.
        It also takes in the number of unigrams and number of bigrams to be included in the feature file (both specified in Config).
         Using NLTK's FreqDist operation, it grabs the most frequent unigrams and bigrams (up to the specified limits) to use as features.
        If @mentions are not to be used (as specified in Config), it removes any tokens beginning with @.
        The returned objects are two lists:
         the first is the list of unigram features (length specified by Config.num_unigrams).
         the second is the list of bigrams features (length specified by Config.num_bigrams).
        """
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
        """
        This function takes in the pandas dataframe with 9 columns (id, text, topic, tokens, pos, NE chunks, unigrams, bigrams, and pos counts).
        It also takes in the list of unigram features, the list of bigram features, and the list of unique labels.
        It then creates a new dataframe with many columns. The first two columns contain each message's id, text.
        Next, it creates a column for each unigram feature. The value for this column is False if a message does not contain that unigram, and it is True if it does contain that unigram.
        Next, it creates a column for each bigram feature. The value for this column is False if a message does not contain that bigram, and it is True if it does contain that bigram.
        Next, if pos counts are to be used as features (specified in Config.pos_counts), it creates a list of unique pos tags identified by get_pos_counts.
         It creates a column for each pos tag. Then, it counts the number of times each pos tag appears in a given message.
         The value for each pos tag's column is the number of times that pos appears in a message.
        Next, if NE chunks are to be used as features (specified in Config.use_ne_chunks), it creates a list of unique NE.
         It checks to see if those NEs are in the list of unigram and bigram features.
         If not, it uses those NEs as features in the same way as unigrams and bigrams are used.
        Next, it creates a column with the labels for each message. Using the list of unique labels (generated by get_labels), it creates a column for each label.
         The value for each label column is False if a message does not take that label, and it is True if a message does contain that label.
         Then, this function drops the column that contains all labels for each message.
        Finally, this function writes this pandas dataframe with many columns (id, text, unigram features, bigram features, pos_count features, and labels) to the file specified in Config.
        There is no returned object.
        """
        try:
            os.mkdir(Config.output_dir)
        except FileExistsError:
            pass

        features = pd.DataFrame()
        features["id"] = docs["id"]
        features["message"] = docs["text"]

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

        if Config.use_ne_chunks:
            features["ne_chunks"] = docs["ne_chunks"]
            chunks = list(features["ne_chunks"])
            chunks = [f for c in chunks for f in c]                 # This line flattens the nested list of NE chunks
            chunks = list(set(chunks))                              # This line de-dupes the flattened list of NE chunks
            for chunk in chunks:
                if chunk not in (unigrams or bigrams):              # This if means that only NEs that aren't in
                    features[chunk] = False                         # unigrams and bigrams are used as features
                    features.loc[features["message"].str.contains(chunk), chunk] = True
            features = features.drop(["ne_chunks"], axis=1)

        features["topics"] = docs["topics"]
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

if __name__ == "__main__":
   main()
