from get_secrets import get_secret
from together import Together
import pandas as pd
import os

together_api_key = get_secret("together_ai")["TOGETHER_API_KEY"]
together_api_client = Together(api_key=together_api_key)


def parse_output(output):
    lines = output.strip().split("\n")
    result = []
    for line in lines:
        if line.startswith("|"):
            data = line.split("|")
            if len(data) == 4:
                result.append(data[1:3])

    # create the df
    result = pd.DataFrame(result, columns=["user_id", "text"])
    return result


def get_LLM_summaries(df):
    print("Generating LLM summaries")
    batch_size = 100
    result = []

    for channel_id in df["channel_id"].unique():
        for time in df["timestamp"].unique():
            print("new channel day")
            channel_day_df = df[
                (df["channel_id"] == channel_id) & (df["timestamp"] == time)
            ]
            if channel_day_df.empty:
                continue
            for i in range(0, len(channel_day_df), batch_size):
                batch = channel_day_df.iloc[i : i + batch_size]
                conversation = "\n\n\n\n\nuser_id\tmessages\n" + "\n".join(
                    batch.apply(lambda row: f"{row['user_id']}\t{row['text']}", axis=1)
                )
                prompt = f"Please read the following transcript of a conversation and provide a summary of the messages for each unique user_id in the conversation. The messages are an aggregation of all their contributions to the conversation.  Please produce the output without headers and with a new line for each user, without an empty line in between different users, in the following format: |user_id|summary|. For example: |U02EZ29LH0Q|I'm currently working on a conference abstract|\n|U02AZ27LH4Q|I'm working late today, because I am a singer|.  The summary should be about half the total length of the aggregated messages for each user. The summary should be written in first-person perspective, not referencing the user. Please consider the entire conversation to gather context on the overall meaning of each person’s messages.  Please try to use the same keywords that are used in the raw text. Even if there is only one line in the conversation, just summarize that single line. The following data contains the user ids and messages to summarize: {conversation}"
                response = together_api_client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1,
                    # stop=[""]
                    # stream=True
                )

                out_cleaned = response.choices[0].message.content
                out_cleaned = parse_output(out_cleaned)
                out_cleaned["channel_id"] = channel_id
                out_cleaned["channel_name"] = channel_day_df["channel_name"].iloc[0]
                out_cleaned["timestamp"] = time
                result.append(out_cleaned)
    result = pd.concat(result, ignore_index=True)
    return result
