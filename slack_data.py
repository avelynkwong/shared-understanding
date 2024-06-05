import pandas as pd
from datetime import datetime, timedelta
import dataframe_image as dfi
import matplotlib.pyplot as plt
import io
import time


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
        self.test_image = None

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
        # dfi.export(self.msg_df[:100], "df.png")
        # self.msg_df.to_csv("wip/message_df.csv")

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
            msg_dict["user_id"] = msg.get("user", None)
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

    def generate_image(self):

        # Data for knowledge convergence
        person1 = [
            (1, 0.3),
            (3, 0.4),
            (5, 0.4),
            (10, 0.5),
            (15, 0.6),
            (17, 0.7),
            (19, 0.9),
            (30, 1),
        ]
        person2 = [
            (1, 0.2),
            (4, 0.3),
            (7, 0.3),
            (8, 0.3),
            (13, 0.4),
            (12, 0.3),
            (18, 0.4),
            (27, 0.4),
        ]

        person3 = [
            (1, 0.4),
            (2, 0.5),
            (6, 0.5),
            (9, 0.6),
            (15, 0.6),
            (18, 0.8),
            (22, 0.9),
            (31, 1),
        ]
        # Styling
        plt.style.use("dark_background")
        plt.rcParams["font.size"] = 11

        # Plotting
        fig, ax = plt.subplots()
        ax.tick_params(axis="both", which="both", length=1)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.scatter(*zip(*person1), label="Person A", marker="o")
        ax.scatter(*zip(*person2), label="Person B", marker="*")
        ax.scatter(*zip(*person3), label="Person C", marker="d")
        ax.legend(frameon=False)
        ax.set_xlabel("Number of Documents")
        ax.set_ylabel("Coherence")
        ax.set_title("Team Knowledge Convergence")

        # Generate data
        # values = [random.randint(1, 100) for _ in range(20)]

        # Create plot
        # plt.style.use("dark_background")
        # fig, ax = plt.subplots()
        # ax.tick_params(axis="both", which="both", length=0)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.plot(values)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        self.test_image = buf

        plt.close(fig)

    def generate_homepage_view(self):
        view = (
            {
                "type": "home",
                "callback_id": "home_view",
                # body of the view
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "Welcome to Shared Understanding Homepage",
                            "emoji": True,
                        },
                    },
                    {"type": "divider"},
                    {
                        "type": "section",
                        "block_id": "section678",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Please select the conversations you would like to analyze.",
                        },
                        "accessory": {
                            "action_id": "select_conversations",
                            "type": "multi_external_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select items",
                            },
                            "min_query_length": 1,
                        },
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "plain_text",
                            "text": "Please select the start and end dates.",
                            "emoji": True,
                        },
                    },
                    {
                        "type": "actions",
                        "block_id": "actions1",
                        "elements": [
                            {
                                "type": "datepicker",
                                "initial_date": self.start_date,
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "Select a date",
                                    "emoji": True,
                                },
                                "action_id": "startdate_picked",
                            },
                            {
                                "type": "datepicker",
                                "initial_date": self.end_date,
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "Select a date",
                                    "emoji": True,
                                },
                                "action_id": "enddate_picked",
                            },
                        ],
                    },
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": "Shared Knowledge Visualizations",
                            "emoji": True,
                        },
                    },
                    {"type": "divider"},
                    {
                        "type": "image",
                        "block_id": "test_data",
                        "image_url": "https://loyal-positively-beetle.ngrok-free.app/test_image?t="
                        + str(time.time()),
                        "alt_text": "Knowledge Convergence Graph",
                    },
                ],
            },
        )

        return view[0]
