import pandas as pd
from itertools import combinations
import numpy as np


def LSM_basic(categories, person_a, person_b):
    lsm_per_category = []
    for category in categories:
        C_a = person_a[category].values[0]
        C_b = person_b[category].values[0]
        lsm_c = 1 - ((abs(C_a - C_b)) / (C_a + C_b + 0.001))
        lsm_per_category.append(lsm_c)
    lsm_overall = np.mean(lsm_per_category)
    return lsm_overall


def LSM_application(df):
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
                result = LSM_basic(categories, person_a, person_b)
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


def group_average(df_result):
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


msg_df = pd.read_csv("test_agg_w_luke.csv")
msg_df = LSM_application(msg_df)
msg_df = group_average(msg_df)
print(msg_df)
