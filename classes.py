import pandas as pd
from datetime import datetime, timedelta
import dataframe_image as dfi


# class to hold slack data
class SlackData:
    def __init__(self, app, bot_token) -> None:
        self.app = app
        self.start_date = str(
            (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        )
        self.end_date = str(datetime.today().strftime("%Y-%m-%d"))
        self.msg_df = pd.DataFrame()
        self.selected_conversations = []
        self.bot_token = bot_token
        self.all_conversations = {}
        self.find_conversations()

    def find_conversations(self):
        # list of conversations app has access to
        conversations = self.app.client.users_conversations(token=self.bot_token)[
            "channels"
        ]
        conversations = {c["name"]: c["id"] for c in conversations}
        self.all_conversations = conversations
        return

    # populate dataframe with messages from all selected channels
    def update_dataframe(self):
        print("Updating dataframe...")
        print(f"Range: {self.start_date}, {self.end_date}")
        # clear
        self.msg_df = pd.DataFrame()
        for c in self.selected_conversations:
            self.get_channel_messages(c)
        dfi.export(self.msg_df[:100], "df.png")

    # populate dataframe with messages from a single channel between specified start and end times
    def get_channel_messages(self, channel_name):
        print(f"Getting messages for channel {channel_name}")
        channel_id = self.all_conversations[channel_name]
        history = self.app.client.conversations_history(
            token=self.bot_token,
            channel=channel_id,
            oldest=self.str_timezone_to_unix(self.start_date),
            latest=self.str_timezone_to_unix(self.end_date),
            inclusive=True,
        )
        messages = history["messages"]
        has_more = history["has_more"]

        self.add_messages_to_df(messages, channel_name, channel_id)

        # paging through time, each page contains maximum 100 messages
        while has_more:
            # page += 1
            prev_ts = messages[-1]["ts"]
            history = self.app.client.conversations_history(
                token=self.bot_token,
                channel=channel_id,
                oldest=self.str_timezone_to_unix(self.start_date),
                latest=prev_ts,
            )
            messages = history["messages"]
            has_more = history["has_more"]
            self.add_messages_to_df(messages, channel_name, channel_id)

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

    def str_timezone_to_unix(self, str_time):
        dt = datetime.strptime(str_time, "%Y-%m-%d")
        # unix time
        unix = dt.timestamp()
        return unix
