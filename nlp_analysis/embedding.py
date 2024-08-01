import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
import io
from get_secrets import get_secret
from together import Together

together_api_key = get_secret("together_ai")["TOGETHER_API_KEY"]
together_api_client = Together(api_key=together_api_key)


# input is dataframe with documents aggregated by channel-user-days
def get_embeddings(df, model_name="sentence-transformers/msmarco-bert-base-dot-v5"):

    # list of msgs
    input_list = df["text"].to_list()

    # create embeddings
    response = together_api_client.embeddings.create(
        model=model_name,
        input=input_list,
    )
    df["embedding"] = [x.embedding for x in response.data]

    print("Embedded documents")
    # initialize PCA model
    pca = PCA(n_components=1)
    final_output = []
    for channel_id in df["channel_id"].unique():
        df_channel = df[df["channel_id"] == channel_id]
        channel_w_group = []
        # loop through all times in a channel and add individual embeddings
        # and group embeddings to a list
        for time in df_channel["timestamp"].unique():
            # get the average embedding for the entire group on a channel-day
            channel_time_df = df_channel[df_channel["timestamp"] == time]
            channel_time_embeddings = np.stack(channel_time_df["embedding"].values)
            group_embedding_avg = channel_time_embeddings.mean(axis=0)
            group_row = {
                "timestamp": time,
                "text": None,
                "user_id": "group",
                "channel_id": channel_id,
                "channel_name": channel_time_df["channel_name"].iloc[0],
                "embedding": group_embedding_avg,
            }
            # append all individual embeddings and group embedding to list
            channel_w_group.append(channel_time_df)
            channel_w_group.append(pd.DataFrame([group_row]))

        # before moving on to next channel, get PCA for all users in a channel (including group)
        df_channel_w_group = pd.concat(channel_w_group, ignore_index=True)
        channel_embeddings = np.stack(df_channel_w_group["embedding"].values)
        pca_result = pca.fit_transform(channel_embeddings)
        # print(pca.explained_variance_ratio_)
        df_channel_w_group["embeddings_1D"] = pca_result.flatten()
        final_output.append(df_channel_w_group)

    df_final_output = pd.concat(final_output, ignore_index=True)
    return df_final_output


# adds standard deviation and scales the PCA 1D embedding values to 0-1
# also add number of users per day
def vis_preprocessing(df):
    # columns to populate
    df["std"] = np.nan
    df["embeddings_1D_scaled"] = np.nan
    df["num_users"] = np.nan
    scaler = MinMaxScaler()

    # group by channel_id and timestamp
    grouped = df.groupby(["channel_id", "timestamp"])
    for (channel_id, time), group in grouped:
        # filter out rows where user_id is 'group'
        channel_time_df_without_group = group[group["user_id"] != "group"]
        if not channel_time_df_without_group.empty:
            # calculate the standard deviation of the 1D embeddings for all users on a channel-day
            std = np.std(channel_time_df_without_group["embeddings_1D"])
            # get num users on a channel-day
            n_users = len(channel_time_df_without_group["user_id"].unique())
            # assign to the row where user_id is 'group'
            mask = (
                (df["channel_id"] == channel_id)
                & (df["timestamp"] == time)
                & (df["user_id"] == "group")
            )
            df.loc[mask, "std"] = std
            df.loc[mask, "num_users"] = n_users
        # scale the 'embeddings_1D' within this group using MinMaxScaler
        scaled_embeddings = scaler.fit_transform(group[["embeddings_1D"]])
        # assign the scaled embeddings back to the original DataFrame
        df.loc[
            (df["channel_id"] == channel_id) & (df["timestamp"] == time),
            "embeddings_1D_scaled",
        ] = scaled_embeddings
    return df


def vis_perperson(df):
    channels = df["channel_id"].unique()

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
        channel_df = df[df["channel_id"] == channel]
        channel_df = channel_df.sort_values(by="timestamp")
        ax = axs[i]
        ax.set_xlabel("Date")
        ax.set_ylim(0, 1.09)
        ax.set_yticks(ticks=np.arange(0, 1.1, 0.1))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", labelrotation=70)
        ax.set_title(
            f"1D Embedding Space Visualization for Channel: {str(channel_df['channel_name'].iloc[0])}",
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.text(
            0.95,
            0.95,
            (
                "Average users per day: "
                + str(np.round(np.nanmean(channel_df["num_users"]), 2))
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )
        ax.text(
            0.95,
            0.80,
            (
                "Average standard deviation in embeddings over time: "
                + str(np.round(np.nanmean(channel_df["std"]), 2))
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

        for user, group in channel_df.groupby("user_id"):
            # sort by timestamp, then convert back to str
            group.loc[:, "timestamp"] = pd.to_datetime(group.loc[:, "timestamp"])
            if user == "group":
                ax.plot(
                    group["timestamp"],
                    group["embeddings_1D_scaled"],
                    marker="*",
                    label=user,
                    linestyle="--",
                    linewidth=5,
                    markersize=15,
                )
            else:
                ax.plot(
                    group["timestamp"],
                    group["embeddings_1D_scaled"],
                    marker="o",
                )
        ax.legend(title=None, loc="lower right", frameon=False)
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
