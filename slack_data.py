import pandas as pd
from datetime import datetime, timedelta
import dataframe_image as dfi
import matplotlib.pyplot as plt
import io
import time
from consent_form import generate_consent_form

# maximum messages to store in dataframe
MAX_DF_SIZE = 5000


# class to hold slack data, each installer will have an instance of this class
class SlackData:
    def __init__(self, app, bot_token) -> None:
        self.app = app
        self.start_date = str((datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"))
        self.end_date = str(datetime.today().strftime("%Y-%m-%d"))
        self.msg_df = pd.DataFrame()
        self.selected_conversations = []
        self.bot_token = bot_token
        self.all_invited_conversations = {}
        self.get_invited_conversations()
        self.test_image = None
        self.consented_users = set()

    def send_consent_form(self):

        # get channel with the most members
        max_members = 0
        largest_channel_id = None
        largest_channel_name = None
        channels_info = self.app.client.conversations_list(
            token=self.bot_token,
            types="public_channel",
            exclude_archived=True,
            limit=200,
        )
        public_channels = channels_info["channels"]
        for pc in public_channels:
            if pc["num_members"] > max_members:
                max_members = pc["num_members"]
                largest_channel_id = pc["id"]
                largest_channel_name = pc["name"]
        next_cursor = channels_info["response_metadata"]["next_cursor"]

        # page through all public channels in workspace
        while next_cursor:
            channels_info = self.app.client.conversations_list(
                token=self.bot_token,
                types="public_channel",
                exclude_archived=True,
                next_cursor=next_cursor,
                limit=200,
            )
            public_channels = channels_info["channels"]
            for pc in public_channels:
                if pc["num_members"] > max_members:
                    max_members = pc["num_members"]
                    largest_channel_id = pc["id"]
                    largest_channel_name = pc["name"]
            next_cursor = channels_info["response_metadata"]["next_cursor"]

        print(
            f"Largest channel: {largest_channel_name}, number of members: {max_members}"
        )

        # get a list of all members in the channel
        #! TODO: change back to actual largest_channel
        largest_channel_id = "C077M8073Q8"
        all_member_ids = []
        members_info = self.app.client.conversations_members(
            token=self.bot_token, channel=largest_channel_id
        )
        member_ids = members_info["members"]
        all_member_ids.extend(member_ids)
        next_cursor = members_info["response_metadata"]["next_cursor"]

        # page through members
        while next_cursor:
            members_info = self.app.client.conversations_members(
                token=self.bot_token, channel=largest_channel_id, cursor=next_cursor
            )
            member_ids = members_info["members"]
            all_member_ids.extend(member_ids)
            next_cursor = members_info["response_metadata"]["next_cursor"]

        # send consent form to each member
        form = generate_consent_form()
        for m in all_member_ids:
            # check if member is a bot
            is_bot = self.app.client.users_info(token=self.bot_token, user=m)["user"][
                "is_bot"
            ]
            # open a DM and send a message
            if not is_bot:
                response = self.app.client.conversations_open(
                    token=self.bot_token, users=m
                )
                channel_id = response["channel"]["id"]
                self.app.client.chat_postMessage(
                    text="Slack Data Consent Form",
                    token=self.bot_token,
                    channel=channel_id,
                    blocks=form,
                )

        # public_channels = {c["name"]: c["id"] for c in public_channels}
        # id = public_channels["general"]

        # form = generate_consent_form()
        # self.app.client.chat_postMessage(
        #     text="Slack Data Consent Form",
        #     token=self.bot_token,
        #     channel=largest_channel_id,
        #     blocks=form,
        # )

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
        print("Updating dataframe...")
        print(f"Range: {self.start_date}, {self.end_date}")
        # clear
        self.msg_df = pd.DataFrame()
        for c in self.selected_conversations:
            self.get_channel_messages(c)
        # dfi.export(self.msg_df[:100], "df.png")
        self.msg_df.to_csv("wip/message_df.csv")

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
        while has_more and len(self.msg_df.index) < MAX_DF_SIZE:
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

        if len(self.msg_df.index) > MAX_DF_SIZE:
            print(
                f"Attempting to pull more than {MAX_DF_SIZE} Slack messages. Clipping to {MAX_DF_SIZE}..."
            )
            self.msg_df = self.msg_df[:MAX_DF_SIZE]

    def add_messages_to_df(self, msg_list, channel_name, channel_id):
        # dict to store each message instance
        exclusions = 0
        msg_dict = {}
        for msg in msg_list:
            user_id = msg.get("user", None)
            if user_id in self.consented_users:
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
            else:
                exclusions += 1

        print(f"Number of messages excluded due to unconsenting users: {exclusions}")

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

    def generate_homepage_view(self, user_id, bot_token, enterprise_id, team_id):
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
                        "image_url": f"https://loyal-positively-beetle.ngrok-free.app/test_image?user_id={user_id}&token={bot_token}&enterprise_id={enterprise_id}&team_id={team_id}&t={str(time.time())}",
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
                ],
            },
        )

        return view[0]

    info = {
        "ok": True,
        "user": {
            "id": "U070EJ574TS",
            "team_id": "T7HD0J5GF",
            "name": "avelyn.wong",
            "deleted": False,
            "color": "73769d",
            "real_name": "Avelyn Wong",
            "tz": "America/New_York",
            "tz_label": "Eastern Daylight Time",
            "tz_offset": -14400,
            "profile": {
                "title": "",
                "phone": "",
                "skype": "",
                "real_name": "Avelyn Wong",
                "real_name_normalized": "Avelyn Wong",
                "display_name": "",
                "display_name_normalized": "",
                "fields": None,
                "status_text": "",
                "status_emoji": "",
                "status_emoji_display_info": [],
                "status_expiration": 0,
                "avatar_hash": "ae946ac378a5",
                "image_original": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_original.png",
                "is_custom_image": True,
                "first_name": "Avelyn",
                "last_name": "Wong",
                "image_24": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_24.png",
                "image_32": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_32.png",
                "image_48": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_48.png",
                "image_72": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_72.png",
                "image_192": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_192.png",
                "image_512": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_512.png",
                "image_1024": "https://avatars.slack-edge.com/2024-05-21/7166757687377_ae946ac378a5e5c54123_1024.png",
                "status_text_canonical": "",
                "team": "T7HD0J5GF",
            },
            "is_admin": True,
            "is_owner": False,
            "is_primary_owner": False,
            "is_restricted": False,
            "is_ultra_restricted": False,
            "is_bot": False,
            "is_app_user": False,
            "updated": 1716490321,
            "is_email_confirmed": True,
            "who_can_share_contact_card": "EVERYONE",
        },
    }
