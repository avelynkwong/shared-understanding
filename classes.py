import pandas as pd
from datetime import datetime
import time


# class to hold slack data
class SlackData:
    def __init__(self, app) -> None:
        self.app = app
        self.start_date = "2020-01-01"
        self.end_date = "2024-01-01"
        self.msg_df = pd.DataFrame()
        self.all_conversations = {}
        self.selected_conversations = []

    # populate dataframe with messages between specified start and end times
    def get_paged_messages(
        self,
        bot_token,
        channel_name,
        start_time="0",
        end_time=str(time.time()),
    ):
        channel_id = self.all_conversations[channel_name]
        history = self.app.client.conversations_history(
            token=bot_token, channel=channel_id, oldest=start_time, latest=end_time
        )
        messages = history["messages"]
        has_more = history["has_more"]

        self.msg_df = self.add_messages_to_df(messages, channel_id)

        # paging through time, each page contains maximum 100 messages
        while has_more:
            # page += 1
            prev_ts = messages[-1]["ts"]
            history = self.app.client.conversations_history(
                token=bot_token, channel=channel_id, oldest=start_time, latest=prev_ts
            )
            messages = history["messages"]
            has_more = history["has_more"]
            self.msg_df = self.add_messages_to_df(messages, channel_name, channel_id)

    def add_messages_to_df(self, msg_list, channel_name, channel_id):
        # dict to store each message instance
        msg_dict = {}
        for msg in msg_list:
            ts = msg["ts"]
            year = datetime.fromtimestamp(float(ts)).year
            msg_dict["year"] = year
            msg_dict["channel_id"] = channel_id
            msg_dict["channel_name"] = channel_name
            msg_dict["user_id"] = msg["user"]
            msg_dict["timestamp"] = ts
            msg_dict["text"] = msg["text"]
            msg_dict["replies_cnt"] = msg.get("reply_count", 0)
            reacts = msg.get("reactions", None)
            reacts_cnt = 0
            if reacts:
                for react in reacts:
                    reacts_cnt += react["count"]
            msg_dict["reacts_cnt"] = reacts_cnt
            self.msg_df = pd.concat(
                [self.msg_df, pd.DataFrame([msg_dict])], ignore_index=True
            )
