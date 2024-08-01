import re
import enchant


def contains_link(text):
    # Regular expression to match URLs
    url_regex = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return re.search(url_regex, text) is not None


def contains_attachment(message):
    # Check if the message contains 'attachments' or 'files'
    return "attachments" in message or "files" in message


def filter_non_dict_words(text):
    # check spelling of words
    d = enchant.Dict("en_US")
    # Split the text into words
    words = text.split()
    # Filter words that are not in the dictionary
    filtered_words = [word for word in words if d.check(word)]
    # Join the filtered words back into a string
    return " ".join(filtered_words)


def general_preprocessing(data):
    # take out @'s and <channels!>

    data["text"] = data["text"].map(lambda x: re.sub(r"\<(.*[^>])>", r"", x))

    # normally we would remove punctuation and stopwords, but we need these for LSM, can uncomment out if needed later

    # #remove punctuation
    # translator = str.maketrans({key: None for key in string.punctuation +"123456789-"})
    # data['comments_processed'] = data['comments_processed'].map(lambda x: x.translate(translator)) #takes out punctuation

    # tokenize words
    # data_words = list(sentence_to_words(documents))

    # data_words = remove_stopwords(data_words)

    # lowercase
    data["text"] = data["text"].map(lambda x: x.lower())
    # data['date'] = data['timestamp'].map(lambda x: x.date())

    data["text"] = data["text"].apply(filter_non_dict_words)

    # do not lemmatize for LSM but do for the LSA

    # lemmatize and keep only nouns, verbs, adjectives and adverbs
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    # data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # print("after lemmatizing")
    # print(data_lemmatized)

    # filter out empty messages
    data = data[data["text"] != ""]

    return data
