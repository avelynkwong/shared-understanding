import pandas as pd
import datetime


def message_aggregation(agg_choice, df):

    # global variables, TODO: allow user to select from app
    agg_num_messages = 10
    agg_time = datetime.timedelta(minutes=30)
    agg_date = datetime.timedelta(days=1)

    df_output = pd.DataFrame()
    # sort df by channel and then timestamp
    df_sorted = df.sort_values(
        by=["channel_id", "timestamp"], ascending=[True, True]
    ).reset_index(drop=True)
    channels = df_sorted["channel_id"].unique()
    print(channels)
    for channel in channels:
        channel_df = df_sorted[df_sorted["channel_id"] == channel]
        channel_df = channel_df.reset_index(drop=True)
        users = channel_df["user_id"].unique()
        print(users)

        for user in users:
            user_df = channel_df[channel_df["user_id"] == user]
            user_df = user_df.reset_index(drop=True)

            # for each user in each channel, assign to a group based on
            # the number of messages sent by the user
            if agg_choice == "message":
                user_df.loc[:, "Group"] = user_df.index // agg_num_messages
                df_agg = (
                    user_df.groupby("Group")
                    .agg(
                        message=("message", " ".join),
                        start_timestamp=("timestamp", "first"),
                        end_timestamp=("timestamp", "last"),
                    )
                    .reset_index(drop=True)
                )
                df_agg.loc[:, "User"] = user
                df_agg.loc[:, "Channel"] = channel
                df_output = pd.concat([df_output, df_agg], ignore_index=True, axis=0)

            elif agg_choice == "time":
                # Initialize variables for tracking current aggregation
                start_time = user_df.loc[0, "timestamp"]
                current_group = 0
                aggregated_messages = []

                for index, row in user_df.iterrows():
                    if row["timestamp"] > start_time + agg_time:
                        # Aggregate messages for the current group
                        aggregated_comment = " ".join(aggregated_messages)
                        df_agg = pd.DataFrame(
                            {
                                "start_timestamp": [start_time],
                                "comment": [aggregated_comment],
                                "user_id": [user],
                                "channel_id": [channel],
                            }
                        )

                        # Append to output DataFrame
                        df_output = pd.concat([df_output, df_agg], ignore_index=True)

                        # Reset aggregation variables for the next group
                        start_time = row["timestamp"]
                        aggregated_messages = [row["message"]]
                        current_group += 1
                    else:
                        aggregated_messages.append(row["message"])

                # Final aggregation for the last group
                aggregated_comment = " ".join(aggregated_messages)
                df_agg = pd.DataFrame(
                    {
                        "start_timestamp": [start_time],
                        "message": [aggregated_comment],
                        "user_id": [user],
                        "channel_id": [channel],
                    }
                )

                # Append to output DataFrame
                df_output = pd.concat([df_output, df_agg], ignore_index=True, axis=0)

            elif agg_choice == "date":
                # start_date = datetime.day(user_df.loc[0, 'timestamp'])
                df_agg = (
                    user_df.groupby(user_df["timestamp"].dt.date)["message"]
                    .apply(" ".join)
                    .reset_index()
                )
                df_agg["user_id"] = user
                df_agg["channel_id"] = channel

                # Append to output DataFrame
                df_output = pd.concat([df_output, df_agg], ignore_index=True, axis=0)
                # print(df_output)
                # df_output = df_output.rename(columns={'timestamp': 'group'})

    # add an index column labelled "group" at the end
    df_output["group"] = range(1, len(df_output) + 1)
    return df_output
