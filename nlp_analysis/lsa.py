import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
from gensim import corpora
from gensim.models import LsiModel, LogEntropyModel
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
from gensim import matutils
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import math
import io


def token_stem_stop(docs):
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r"\w+")
    # create English stop words list
    en_stop = set(stopwords.words("english"))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    doc_set = docs.tolist()
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts


def prepare_corpus(processed_docs):
    """
    Input  : clean document
    Purpose : create term dictionary of our corpus and convert list of documents (corpus) into a word-doc matrix
    Output : term dictionary and word-doc matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(processed_docs)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [
        dictionary.doc2bow(doc) for doc in processed_docs
    ]  # doc2bow function returns tuples of type (word id, frequency)
    # generate LDA model
    return dictionary, doc_term_matrix


# trains different LSA models with varying number of topics and outputs the best model
def train_LSA_models(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Parameters:
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        stop: maximum number of topics
    Outputs:
        best_model: the LSA model with the highest coherence score
    """
    best_coherence = 0
    best_model = None
    for num_topics in range(start, stop, step):
        # train LSA model
        model = LsiModel(
            doc_term_matrix, num_topics=num_topics, id2word=dictionary, random_seed=123
        )
        coherencemodel = CoherenceModel(
            model=model, texts=doc_clean, dictionary=dictionary, coherence="c_v"
        )
        model_coherence = coherencemodel.get_coherence()
        if model_coherence > best_coherence:
            best_model = model
            best_coherence = model_coherence

    print(f"Best model with {best_model.num_topics} topics:")
    # for idx, topic in best_model.print_topics(num_topics=num_topics):
    #     print(f"Topic #{idx + 1}:")
    #     print(topic)
    return best_model


def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


# Normalize the bow representation of the document
def get_normalized_bow(doc, dictionary):
    bow = dictionary.doc2bow(doc)
    bow_dense = np.zeros(len(dictionary))
    for idx, value in bow:
        bow_dense[idx] = value
    normalized_bow = normalize_data(bow_dense.reshape(1, -1))
    return list(zip(range(len(normalized_bow[0])), normalized_bow[0]))


# Apply logent_transformation on normalized data
def apply_transformation(doc, dictionary, logent_transformation):
    normalized_bow = get_normalized_bow(doc, dictionary)
    return logent_transformation[normalized_bow]


def build_model(matrix, dictionary, df_processed, topic_proportion, step):

    # log entropy transformation
    logent_transformation = LogEntropyModel(matrix, dictionary)
    logent_corpus = logent_transformation[matrix]
    dense_logent_matrix = matutils.corpus2dense(
        logent_corpus, num_terms=len(dictionary)
    ).T
    filtered_corpus = matutils.Dense2Corpus(dense_logent_matrix.T)

    # Create a new dictionary with the filtered terms
    filtered_dictionary = dictionary
    best_model = train_LSA_models(
        filtered_dictionary,
        filtered_corpus,
        df_processed,
        stop=10,  # TODO: change back to: len(df_processed) // topic_proportion,
        step=step,
    )
    return best_model, logent_transformation


def get_matrix_centroid(matrix, num_topics):
    # create numpy array, rows = number of documents, cols = number of topics
    array = np.zeros((len(matrix), num_topics))
    for doc_idx, doc_dist in enumerate(matrix):
        for topic_idx, topic_weight in doc_dist:
            array[doc_idx][topic_idx] = topic_weight
    # average along the topic dimension
    centroid = np.mean(array, axis=0)
    return centroid


# add the doc-topic matrices to each document grouped by channel and time
def get_LSA_topic_dists(df, best_model, logent_transformation, dictionary):
    LSA_topic_dists = []
    for channel_id in df["channel_id"].unique():
        for time in df["timestamp"].unique():
            filtered_df = df[
                (df["channel_id"] == channel_id) & (df["timestamp"] == time)
            ]
            if len(filtered_df) == 0:
                continue

            # for each document in a channel within a time interval, get the distribution across all topics
            for i, doc in filtered_df.iterrows():
                doc_topic_dist = best_model[
                    logent_transformation[
                        (dictionary.doc2bow(doc["processed_for_LSA"]))
                    ]
                ]
                LSA_topic_dists.append(
                    {
                        "channel_id": channel_id,
                        "channel_name": filtered_df["channel_name"].iloc[0],
                        "timestamp": time,
                        "num_users": len(filtered_df["user_id"].unique()),
                        "user": doc["user_id"],
                        "matrix": doc_topic_dist,
                    }
                )
    LSA_topic_dists = pd.DataFrame(LSA_topic_dists)
    return LSA_topic_dists


def memory_cosine_sim(df, best_model, logent_transformation, dictionary):
    # option with memory
    LSA_topic_dists = get_LSA_topic_dists(
        df, best_model, logent_transformation, dictionary
    )  # TODO: don't call within this function! we want to reuse this dataframe for the semantic coherence method

    # calculate the running average centroids and compute cosine similarities with final group centroid
    cosine_similarities = []
    for channel_id in LSA_topic_dists["channel_id"].unique():
        df_channel = LSA_topic_dists[LSA_topic_dists["channel_id"] == channel_id]
        channel_name = df_channel["channel_name"].iloc[0]
        # calculate the final centroid for this channel
        channel_matrix = df_channel["matrix"]
        final_channel_centroid = get_matrix_centroid(
            channel_matrix, best_model.num_topics
        )

        # calculating running average centroids for all docs up to different times for each channel
        for time in df_channel["timestamp"].unique():
            channel_time_df = df_channel[df_channel["timestamp"] <= time]
            num_users = len(channel_time_df["user"].unique())
            # num_user_list = [num_users] * (num_users + 1)
            # channel_list = [channel_id] * (num_users + 1)
            # time_list = [time] * (num_users + 1)
            # channel_name_list = [df_filtered["channel_name"].iloc[0]] * (num_users + 1)

            # calculate running average centroid for all docs in the channel with a timestamp <= time
            channel_time_matrix = channel_time_df["matrix"]
            # if type(channel_time_matrix[0]) is tuple:
            #     channel_time_matrix = [channel_time_matrix]
            group_centroid = get_matrix_centroid(
                channel_time_matrix, best_model.num_topics
            )
            group_cosine_similarity = 1 - cosine(group_centroid, final_channel_centroid)

            cosine_similarities.append(
                {
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "timestamp": time,
                    "num_users": num_users,
                    "user": "group",
                    "cosine_sim": group_cosine_similarity,
                }
            )

            # calculate running average centroid for all docs from a specific user in the channel
            for user in channel_time_df["user"].unique():
                channel_time_user_df = channel_time_df[channel_time_df["user"] == user]
                channel_time_user_matrix = channel_time_user_df["matrix"]
                # if len(channel_time_user_matrix) > 0:
                #     if type(channel_time_user_matrix[0]) is tuple:
                #         channel_time_user_matrix = [channel_time_user_matrix]
                user_centroid = get_matrix_centroid(
                    channel_time_user_matrix, best_model.num_topics
                )
                user_cosine_similarity = 1 - cosine(
                    user_centroid, final_channel_centroid
                )

                cosine_similarities.append(
                    {
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "timestamp": time,
                        "num_users": None,
                        "user": user,
                        "cosine_sim": user_cosine_similarity,
                    }
                )

    cosine_similarities = pd.DataFrame(cosine_similarities)

    return cosine_similarities


def LSA_cosine_sim_vis(LSA_df, agg_type="date"):

    channels = LSA_df["channel_id"].unique()

    # styling
    fontsize = 20
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = fontsize

    # subplot columns
    cols = 1
    # subplot rows
    rows = math.ceil(len(channels) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 6))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    else:
        axs = axs.flatten()

    # Loop through each channel
    for i, channel in enumerate(channels):
        channel_df = LSA_df[LSA_df["channel_id"] == channel]
        channel_df = channel_df.sort_values(by="timestamp")
        if len(channel_df) >= 2:
            ax = axs[i]
            if agg_type == "date":
                ax.set_xlabel("Date")
            elif agg_type == "message":
                ax.set_xlabel("Number of Messages")
            elif agg_type == "time":
                ax.set_xlabel("Number of Time Intervals")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(
                str(channel_df["channel_name"].iloc[0]),
                fontsize=fontsize,
                fontweight="bold",
            )
            ax.set_ylim(0, 1.09)
            ax.set_yticks(ticks=np.arange(0, 1.1, 0.1))
            step_size = math.ceil(
                (max(channel_df["timestamp"]) - min(channel_df["timestamp"])).days / 10
            )
            # set x-axis ticks and format
            # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=step_size))
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.tick_params(axis="x", labelrotation=70)
            ax.text(
                0.95,
                0.95,
                (
                    "Average Number of Users: "
                    + str(
                        np.round(
                            np.mean(
                                channel_df[channel_df["user"] == "group"]["num_users"]
                            ),
                            2,
                        )
                    )
                ),
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
            )

            # plt.text(
            #     0.95,
            #     0.8,
            #     (
            #         "Average standard deviation in coherence over time: "
            #         + str(np.round(np.mean(channel_df["std"]), 2))
            #     ),
            #     transform=plt.gca().transAxes,
            #     fontsize=12,
            #     verticalalignment="top",
            #     horizontalalignment="right",
            #     bbox=dict(facecolor="white", alpha=0.5),
            # )

            legend_user = "group"
            for user, group in channel_df.groupby("user"):
                if user == legend_user:
                    ax.plot(
                        group["timestamp"], group["cosine_sim"], marker="o", label=user
                    )
                else:
                    ax.plot(group["timestamp"], group["cosine_sim"], marker="o")
            ax.legend(title=None, loc="lower right", frameon=False)

    # Hide unused subplots if there are any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(w_pad=0.2, h_pad=1)
    # plt.subplots_adjust(wspace=0.2, hspace=1)

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.5)
    buf.seek(0)
    plt.close(fig)
    return buf


def compute_LSA_analysis(df, topic_proportion, step, memory=True):
    """
    Inputs:
        df: dataframe that has already be preprocessed with general_preprocessing (remove non dict words, etc.) and aggregated
        agg_type: how messages are grouped into documents
        topic_proportion: 1/topic_proportion is the maximum number of topics to train for the LSA model

    """
    # tokenize the messages
    lsa_processed_docs = token_stem_stop(df["text"])
    df["processed_for_LSA"] = lsa_processed_docs
    # create dictionary of words and word-document matrix
    dictionary, matrix = prepare_corpus(lsa_processed_docs)

    best_model, logent_transformation = build_model(
        matrix, dictionary, lsa_processed_docs, topic_proportion, step
    )

    # compute cosine similarity LSA method
    LSA_cosine_sim_df = memory_cosine_sim(
        df, best_model, logent_transformation, dictionary
    )

    # compute semantic coherence LSA method

    return LSA_cosine_sim_df
    # else: # no memory of previous aggregation
    #     LSA_df = no_memory_coherence(test_output, best_model, logent_transformation, dictionary)

    # LSA_viz(LSA_df, agg_type)


# preprocessed_df = pd.read_csv("test_agg_w_luke.csv")
# # TODO: see if works without these steps
# preprocessed_df["timestamp"] = pd.to_datetime(preprocessed_df["timestamp"])
# preprocessed_df["text"] = preprocessed_df["text"].astype(str)
# compute_LSA_analysis(preprocessed_df, 20, 2, True)
