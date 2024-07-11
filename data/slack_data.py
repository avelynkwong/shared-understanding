import pandas as pd
import datetime
import dataframe_image as dfi
import matplotlib.pyplot as plt
import io
import time
from db.utils import get_consented_users
from dotenv import load_dotenv
import os
from nlp_analysis.data_preprocessing import *
from nlp_analysis.aggregation import *
from nlp_analysis.lsm import *

# maximum messages to store in dataframe
MAX_DF_SIZE = 5000

# env vars
load_dotenv()
URI = os.getenv("SLACK_URI")


# class to hold slack data, each installer will have an instance of this class
class SlackData:
    def __init__(self, app, bot_token, team_id) -> None:
        self.app = app
        self.start_date = str(
            (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        )
        self.end_date = str(datetime.datetime.today().strftime("%Y-%m-%d"))
        self.msg_df = pd.DataFrame()
        self.lsm_df = pd.DataFrame()
        self.selected_conversations = []
        self.bot_token = bot_token
        self.all_invited_conversations = {}
        self.get_invited_conversations()
        self.lsm_image = None
        self.team_id = team_id
        self.analysis_users_consented = (
            set()
        )  # number of users consented in the current selected conversations/time period
        self.consent_exclusions = 0  # messages excluded due to lack of consent
        self.subsampling_exclusions = 0  # messages excluded due to max df size limit

    def clear_analysis_data(self):
        self.msg_df = pd.DataFrame()
        self.analysis_users_consented = set()
        self.lsm_df = pd.DataFrame()

    def get_invited_conversations(self):
        # list of conversations app has access to (has been invited into channel)
        conversations = self.app.client.users_conversations(token=self.bot_token)[
            "channels"
        ]
        conversations = {c["name"]: c["id"] for c in conversations}
        self.all_invited_conversations = conversations
        return

    # populate dataframe with messages from all selected channels
    def update_dataframe(self):
        # print("Updating dataframe...")
        # print(f"Range: {self.start_date}, {self.end_date}")
        # clear
        self.msg_df = pd.DataFrame()

        # keep track of total number of message exclusions due to unconsenting users
        self.consent_exclusions = 0  # reset
        for c in self.selected_conversations:
            self.get_channel_messages(c)
        # dfi.export(self.msg_df[:100], "df.png")
        # self.msg_df.to_csv("message_df_raw.csv")

        # process the df messages
        if not self.msg_df.empty:

            total_rows = len(self.msg_df)

            # subsample the dataframe if it is too large
            if total_rows > MAX_DF_SIZE:
                subsample_rate = max(1, math.ceil(total_rows / MAX_DF_SIZE))
                self.msg_df = self.msg_df.iloc[::subsample_rate]
            if len(self.msg_df) > MAX_DF_SIZE:  # trim is still larger than MAX_DF_SIZE
                self.msg_df = self.msg_df.head(MAX_DF_SIZE)
            self.subsampling_exclusions = max(0, total_rows - MAX_DF_SIZE)

            self.msg_df = general_preprocessing(self.msg_df)
            print("preprocessed df messages")
            # self.msg_df.to_csv("message_df_postprocessed.csv")

    # populate dataframe with messages from a single channel between specified start and end times
    def get_channel_messages(self, channel_name):

        print(f"Getting messages for channel {channel_name}")
        channel_id = self.all_invited_conversations[channel_name]
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
        consented_users = get_consented_users(self.team_id)
        for msg in msg_list:
            subtype = msg.get("subtype", "*")
            user_id = msg.get("user", None)
            if subtype != "channel_join" and user_id in consented_users:
                self.analysis_users_consented.add(user_id)
                ts = datetime.datetime.fromtimestamp(
                    float(msg["ts"])
                )  # convert unix timestamp to datetime
                year = ts.year
                msg_dict["year"] = year
                msg_dict["channel_id"] = channel_id
                msg_dict["channel_name"] = channel_name
                msg_dict["user_id"] = msg.get("user", None)
                msg_dict["timestamp"] = ts
                msg_dict["text"] = str(msg["text"])
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
            elif (
                subtype != "channel_join" and user_id != None
            ):  # don't count bot users as unconsenting users
                # print(msg)
                self.consent_exclusions += 1

    def str_timezone_to_unix(self, str_time):
        dt = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        # unix time
        unix = dt.timestamp()
        return unix

    def create_lsm_vis(self):

        if not self.msg_df.empty:
            self.lsm_df = message_aggregation(self.msg_df)
            # TODO: add look and remove this hard coded value
            self.lsm_df = pd.read_csv("test_agg_w_luke.csv")

            # get lsm values and generate image
            self.lsm_df = LSM_application(self.lsm_df)
            self.lsm_df = group_average(self.lsm_df)
            self.lsm_image = per_channel_vis_LSM(self.lsm_df)

    def generate_homepage_view(self, user_id, bot_token, enterprise_id, team_id):
        vis_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Note that the generation of the following visualizations are rate-limited. If visualizations stop updating, please wait and try again later.",
                },
            },
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Latent Semantic Mapping",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_The number of messages excluded due to unconsenting users is: {self.consent_exclusions}_",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_The number of messages excluded due to limits on message processing: {self.subsampling_exclusions}_",
                },
            },
            {
                "type": "image",
                "block_id": "test_data",
                "image_url": f"{URI}/lsm_image?token={bot_token}&team_id={team_id}&t={str(time.time())}",
                "alt_text": "Knowledge Convergence Graph",
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "style": "primary",
                        "text": {
                            "type": "plain_text",
                            "text": "Submit Results",
                            "emoji": True,
                        },
                        "value": "submit_analysis",
                        "action_id": "submit_analysis",
                    }
                ],
            },
        ]

        error_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":exclamation: Please select channels and/or a valid date range containing messages.",
                },
            },
        ]

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
                        "block_id": "conversation_select",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"To get started, add this Slack application to one or more public channels and select the conversation(s) you would like to analyze. A maximum of {MAX_DF_SIZE} Slack messsages can be analyzed at once.",
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
                    {"type": "divider"},
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
                        "block_id": "datepicker",
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
                ],
            },
        )

        if not self.msg_df.empty:
            view[0]["blocks"].extend(vis_blocks)
        else:
            view[0]["blocks"].extend(error_blocks)

        return view[0]
