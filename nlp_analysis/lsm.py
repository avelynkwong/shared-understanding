import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import math
import io
import matplotlib.dates as mdates
from get_secrets import get_secret
import json
import requests
import datetime

liwc = get_secret("liwc_api_secrets")
LIWC_URL = liwc["URL"]
LIWC_API_KEY = liwc["API_KEY"]
LIWC_API_SECRET = liwc["API_SECRET"]


# Function to send POST request and handle response
def process_batch(batch):
    # convert df to list of dictionaries
    data = batch.rename(columns={"text": "content"})[["request_id", "content"]].to_dict(
        orient="records"
    )
    # convert list of dictionaries to JSON
    data_json = json.dumps(data)
    # Send POST request
    res = requests.post(LIWC_URL, auth=(LIWC_API_KEY, LIWC_API_SECRET), data=data_json)

    request_ids = []
    liwcs = []
    # check if response is JSON and process it
    try:
        resp_json = res.json()
        for item in resp_json["results"]:
            request_ids.append(item["request_id"])
            liwcs.append(item["liwc"])
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        print(f"Response content: {res.text}")
        resp_json = []
    return request_ids, liwcs


def get_LIWC_values(df, prev_lsm_run_date, todays_lsm_count):

    today = datetime.datetime.today().strftime("%Y-%m-%d")
    if today != prev_lsm_run_date:
        todays_lsm_count = 0  # reset count

    # limit to 1000 words per channel
    max_words_per_day = 20000
    max_words_per_channel = 5000
    keep = []
    for channel_id, group in df.groupby("channel_id"):
        cumulative_words = 0

        for idx, row in group.iterrows():
            word_count = len(row["text"].split())

            # check if adding the current message will exceed the word limit
            if cumulative_words + word_count <= max_words_per_channel:
                cumulative_words += word_count
                keep.append(idx)
            else:
                break

    todays_lsm_count += cumulative_words
    print("LSM count for today: ", todays_lsm_count)
    if todays_lsm_count > max_words_per_day:
        return pd.DataFrame(), todays_lsm_count, today

    # filter the original df to keep only the rows with the selected indices
    df = df.loc[keep]

    # Add a request_id column
    df["request_id"] = [f"req-{i+1}" for i in range(len(df))]
    # process dataframe in batches
    batch_size = 1000
    request_ids = []
    ppron = []
    ipron = []
    article = []
    prep = []
    negate = []
    adverb = []
    auxverb = []
    conj = []

    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df[start:end]
        r_ids, ls = process_batch(batch)

        for i in range(len(ls)):
            request_ids.append(r_ids[i])
            ppron.append(ls[i]["personal_pronouns"])
            ipron.append(ls[i]["impersonal_pronouns"])
            article.append(ls[i]["articles"])
            prep.append(ls[i]["prepositions"])
            negate.append(ls[i]["negations"])
            adverb.append(ls[i]["adverbs"])
            auxverb.append(ls[i]["auxiliary_verbs"])
            conj.append(ls[i]["conjunctions"])

    liwc_data = {
        "request_id": request_ids,
        "ppron": ppron,
        "ipron": ipron,
        "article": article,
        "prep": prep,
        "negate": negate,
        "adverb": adverb,
        "auxverb": auxverb,
        "conj": conj,
    }
    # convert responses to DataFrame
    liwc_data = pd.DataFrame(liwc_data)
    # merge the original df with the liwc df on 'request_id'
    result_df = df.merge(liwc_data, on="request_id", how="left")
    return result_df, todays_lsm_count, today


def avg_lsm_score(categories, person_a, person_b):
    lsm_per_category = []
    for category in categories:
        C_a = person_a[category].values[0]
        C_b = person_b[category].values[0]
        lsm_c = 1 - ((abs(C_a - C_b)) / (C_a + C_b + 0.001))
        lsm_per_category.append(lsm_c)
    lsm_overall = np.mean(lsm_per_category)
    return lsm_overall


def compute_lsm_scores(df):
    categories = [
        "ppron",
        "ipron",
        "article",
        "prep",
        "negate",
        "adverb",
        "auxverb",
        "conj",
    ]
    df = df.fillna(0)
    channels = df["channel_id"].unique()
    lsm_result = []
    for channel in channels:
        times = df[df["channel_id"] == channel]["timestamp"].unique()
        for time in times:
            users = df[(df["channel_id"] == channel) & (df["timestamp"] == time)][
                "user_id"
            ].unique()
            user_combinations = list(combinations(users, 2))
            for combination in user_combinations:
                if len(combination) == 0:
                    continue
                a, b = combination
                person_a = df[
                    (df["channel_id"] == channel)
                    & (df["timestamp"] == time)
                    & (df["user_id"] == a)
                ]
                person_b = df[
                    (df["channel_id"] == channel)
                    & (df["timestamp"] == time)
                    & (df["user_id"] == b)
                ]
                result = avg_lsm_score(categories, person_a, person_b)
                lsm_result.append(
                    {
                        "channel_id": channel,
                        "channel_name": df[df["channel_id"] == channel][
                            "channel_name"
                        ].iloc[0],
                        "timestamp": time,
                        "user_a": a,
                        "user_b": b,
                        "LSM": result,
                    }
                )

    lsm_result = pd.DataFrame(lsm_result)
    return lsm_result


# get the average LSM scores for each channel-day
def grouped_lsm_scores(df_result):
    avg_lsm = []
    for channel in df_result["channel_id"].unique():
        for time in df_result["timestamp"].unique():
            filtered_df = df_result[
                (df_result["channel_id"] == channel) & (df_result["timestamp"] == time)
            ]
            if len(filtered_df) > 0:
                average_lsm = filtered_df["LSM"].mean()
                avg_lsm.append(
                    {
                        "channel_id": channel,
                        "channel_name": filtered_df["channel_name"].iloc[0],
                        "timestamp": time,
                        "num_users": len(
                            set(
                                pd.concat(
                                    [filtered_df["user_a"], filtered_df["user_b"]]
                                ).tolist()
                            )
                        ),
                        "avg_LSM": average_lsm,
                    }
                )
    avg_lsm = pd.DataFrame(avg_lsm)
    return avg_lsm


def moving_avg_lsm(group_avg, window_size):
    if len(group_avg) < window_size:
        window_size = 1
    group_moving_avg = []
    for channel_id in group_avg["channel_id"].unique():
        channel_df = group_avg[group_avg["channel_id"] == channel_id]

        # sort by timestamp
        channel_df.loc[:, "timestamp"] = pd.to_datetime(channel_df.loc[:, "timestamp"])
        channel_df = channel_df.sort_values(by="timestamp")

        for i in range(len(channel_df) - window_size - 1):
            window = channel_df.iloc[i : i + window_size]
            moving_avg = window["avg_LSM"].mean()
            group_moving_avg.append(
                {
                    "channel_id": channel_id,
                    "channel_name": window["channel_name"].iloc[(window_size // 2)],
                    "timestamp": window["timestamp"].iloc[(window_size // 2)],
                    "num_users": window["num_users"].mean(),
                    "avg_LSM": moving_avg,
                }
            )
    group_moving_avg = pd.DataFrame(group_moving_avg)
    return group_moving_avg


def per_channel_vis_LSM(group_avg, agg_type="date"):
    channels = group_avg["channel_id"].unique()
    num_channels = len(channels)

    # styling
    fontsize = 20
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = fontsize

    # subplot columns
    cols = 1
    # subplot rows
    rows = math.ceil(num_channels / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 7))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    else:
        axs = axs.flatten()

    for i, channel in enumerate(channels):
        channel_df = group_avg[group_avg["channel_id"] == channel]
        channel_df.loc[:, "timestamp"] = pd.to_datetime(channel_df.loc[:, "timestamp"])
        channel_df = channel_df.sort_values(by="timestamp")
        ax = axs[i]
        if agg_type == "date":
            ax.set_xlabel("Date")
        elif agg_type == "message":
            ax.set_xlabel("Number of Messages")
        elif agg_type == "time":
            ax.set_xlabel("Number of Time Intervals")
        ax.set_ylabel("Average Shared Language (0-1)")
        ax.set_title(
            str(
                f'Average Language Style Matching Between Users for Channel: {channel_df["channel_name"].iloc[0]}'
            ),
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.set_ylim(0, 1.09)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", labelrotation=70)

        ax.text(
            0.95,
            0.95,
            ("Average Users: " + str(np.round(np.mean(channel_df["num_users"]), 2))),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

        # polynomial fitting?
        # x = [i for i in range(len(channel_df["timestamp"]))]
        # y = list(channel_df["avg_LSM"])
        # z = np.polyfit(x, y, 4)
        # p = np.poly1d(z)
        # ax.plot(
        #     channel_df["timestamp"],
        #     p(x),
        #     color="white",
        #     linestyle="--",
        #     linewidth=3,
        #     alpha=0.8,
        # )
        # ax.scatter(
        #     channel_df["timestamp"],
        #     channel_df["avg_LSM"],
        #     alpha=0.6,
        #     edgecolor="none",
        #     s=80,
        # )

        # moving average
        ax.plot(
            channel_df["timestamp"],
            channel_df["avg_LSM"],
            linestyle="--",
            linewidth=5,
            alpha=0.8,
            marker="*",
            markersize=15,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
