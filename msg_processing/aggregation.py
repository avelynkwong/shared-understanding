import pandas as pd
import datetime


def message_aggregation(agg_choice, df):

    # TODO: see if works without this step
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # TODO: make these variables settable by user
    agg_num_messages = 10
    agg_time = datetime.timedelta(minutes=30)
    agg_date = datetime.timedelta(days=1)

    df_output = pd.DataFrame()
    # sort df by channel and then timestamp
    df_sorted = df.sort_values(
        by=["channel_id", "timestamp"], ascending=[True, True]
    ).reset_index(drop=True)
    channels = df_sorted["channel_id"].unique()
    # print(channels)
    for channel in channels:
        channel_df = df_sorted[df_sorted["channel_id"] == channel]
        channel_df = channel_df.reset_index(drop=True)
        # print(channel_df)
        users = channel_df["user_id"].unique()
        # print(users)

        for user in users:
            user_df = channel_df[channel_df["user_id"] == user]
            user_df = user_df.reset_index(drop=True)
            if agg_choice == "message":

                user_df.loc[:, "Group"] = user_df.index // agg_num_messages
                df_agg = (
                    user_df.groupby("Group")
                    .agg(
                        text=("text", " ".join),
                        start_timestamp=("timestamp", "first"),
                        end_timestamp=("timestamp", "last"),
                        group=("Group", "first"),
                    )
                    .reset_index(drop=True)
                )
                df_agg.loc[:, "user_id"] = user
                df_agg.loc[:, "channel_id"] = channel
                df_agg.loc[:, "channel_name"] = channel_df["channel_id"][0]
                df_agg.loc[:, "group"] = (df_agg.loc[:, "group"] + 1) * agg_num_messages
                df_agg = df_agg.rename(columns={"group": "timestamp"})
                df_output = pd.concat([df_output, df_agg], ignore_index=True, axis=0)

            elif agg_choice == "time":
                # Initialize variables for tracking current aggregation
                start_time = user_df.loc[0, "timestamp"]
                current_group = 0
                message_window = []

                for _, row in user_df.iterrows():
                    # within agg_time window, append messages to a list
                    if row["timestamp"] <= start_time + agg_time:
                        message_window.append(row["text"])

                    # exceeded window, aggregate all messages in the window and
                    # add to df_output
                    else:
                        # Aggregate messages for the current group
                        aggregated_text = " ".join(message_window)
                        df_agg = pd.DataFrame(
                            {
                                "timestamp": current_group,
                                "text": [aggregated_text],
                                "user_id": [user],
                                "channel_id": [channel],
                                "channel_name": channel_df["channel_name"][0],
                            }
                        )
                        # Append to output df
                        df_output = pd.concat([df_output, df_agg], ignore_index=True)

                        # reset aggregation variables for the next group
                        start_time = row["timestamp"]
                        message_window = [row["text"]]
                        current_group += 1

                # final aggregation for the last group
                aggregated_text = " ".join(message_window)
                df_agg = pd.DataFrame(
                    {
                        "timestamp": current_group + 1,
                        "text": [aggregated_text],
                        "user_id": [user],
                        "channel_id": [channel],
                        "channel_name": channel_df["channel_name"][0],
                    }
                )
                # append to output df
                df_output = pd.concat([df_output, df_agg], ignore_index=True, axis=0)

            elif agg_choice == "date":
                df_agg = (
                    user_df.groupby(user_df["timestamp"].dt.date)["text"]
                    .apply(" ".join)
                    .reset_index()
                )
                df_agg["user_id"] = user
                df_agg["channel_id"] = channel
                df_agg["channel_name"] = channel_df["channel_name"][0]

                # Append to output df
                df_output = pd.concat([df_output, df_agg], ignore_index=True, axis=0)
    return df_output
