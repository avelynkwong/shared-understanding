import pandas as pd
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import math
import io
import matplotlib.dates as mdates


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
        ax = axs[i]
        if agg_type == "date":
            ax.set_xlabel("Date")
        elif agg_type == "message":
            ax.set_xlabel("Number of Messages")
        elif agg_type == "time":
            ax.set_xlabel("Number of Time Intervals")
        ax.set_ylabel("Average Shared Language (0-1)")
        ax.set_title(
            str(channel_df["channel_name"].iloc[0]),
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
