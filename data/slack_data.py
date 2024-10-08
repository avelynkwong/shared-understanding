import pandas as pd
import datetime
import dataframe_image as dfi
import matplotlib.pyplot as plt
import io
import time
from db.utils import get_received_form_users, get_consented_users
from dotenv import load_dotenv
import os
from nlp_analysis.data_preprocessing import *
from nlp_analysis.aggregation import *
from nlp_analysis.lsm import *
from nlp_analysis.lsa import *
from nlp_analysis.embedding import *
from slack_sdk.errors import SlackApiError
from nlp_analysis.LLM_summarization import get_LLM_summaries

# maximum messages to store in dataframe
MAX_DF_SIZE = 5000
WINDOW_SIZE = 3
MIN_DF_SIZE = WINDOW_SIZE * 2

# env vars
load_dotenv()
URI = os.getenv("SLACK_URI")


# class to hold slack data, each installer will have an instance of this class
class SlackData:
    def __init__(self, app, bot_token, team_id, actor_user_id) -> None:
        self.app = app
        self.start_date = str(
            (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        )
        self.end_date = str(datetime.datetime.today().strftime("%Y-%m-%d"))
        self.msg_df = pd.DataFrame()
        self.lsm_df = pd.DataFrame()  # eventually holds lsm analysis results
        self.lsa_cosine_df = pd.DataFrame()  # eventually holds lsa cosine sim results
        self.lsa_coherence_df = pd.DataFrame()  # eventually holds lsa sem cohere res
        self.pp_embedding_df = pd.DataFrame()  # eventually holds pp embedding results
        self.group_embedding_df = pd.DataFrame()  # eventually holds group embedding res
        self.llm_summarized_df = pd.DataFrame()
        self.selected_conv_ids = []
        self.selected_conv_names = []
        self.user_name_to_id = (
            {}
        )  # map real names to user ids for database leader submission
        self.bot_token = bot_token
        self.all_invited_conversations = {}
        self.get_invited_conversations()
        self.team_id = team_id
        self.actor_user_id = actor_user_id
        self.consented_users = []
        self.analysis_users_consented = (
            set()
        )  # number of users consented in the current selected conversations/time period

        # keep track of reacts sent and attachments sent for post analysis
        self.reactions = pd.DataFrame(columns=["user_id", "react_type", "timestamp"])
        self.attachments = pd.DataFrame(columns=["user_id", "timestamp"])

        # Information to be displayed on homepage
        self.consent_exclusions = 0  # messages excluded due to lack of consent
        # self.subsampling_exclusions = 0  # messages excluded due to max df size limit
        self.exceeded_df_limit = False

    def reset_dates(self):
        self.start_date = str(
            (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        )
        self.end_date = str(datetime.datetime.today().strftime("%Y-%m-%d"))

    def clear_analysis_data(self):
        self.msg_df = pd.DataFrame()
        self.analysis_users_consented = set()
        self.lsm_df = pd.DataFrame()
        self.lsa_cosine_df = pd.DataFrame()
        self.lsa_coherence_df = pd.DataFrame()
        self.pp_embedding_df = pd.DataFrame()
        self.group_embedding_df = pd.DataFrame()
        self.reactions = pd.DataFrame(columns=["user_id", "react_type", "timestamp"])
        self.attachments = pd.DataFrame(columns=["user_id", "timestamp"])
        self.llm_summarized_df = pd.DataFrame()

    def get_invited_conversations(self):
        # list of conversations app has access to (has been invited into channel)
        conversations = self.app.client.users_conversations(token=self.bot_token)[
            "channels"
        ]
        conversations = {c["name"]: c["id"] for c in conversations}
        self.all_invited_conversations = conversations
        return

    # populate dataframe with messages from all selected channels
    def update_dataframe(self, actor_user_id, use_llm_summary=False):
        # publish loading view
        self.app.client.views_publish(
            token=self.bot_token,
            user_id=actor_user_id,
            view=self.generate_homepage_view(
                self.bot_token, self.team_id, self.actor_user_id, loading=True
            ),
        )

        # get the updated list of consented users every time there is a dataframe update
        # could include a new channel selection
        self.consented_users = get_consented_users(self.team_id)
        # clear old results
        self.clear_analysis_data()

        # keep track of total number of message exclusions due to unconsenting users
        self.consent_exclusions = 0  # reset
        for c in self.selected_conv_names:
            self.get_channel_messages(c, actor_user_id)
            if len(self.msg_df) > MAX_DF_SIZE:
                self.exceeded_df_limit = True
                return
        # self.msg_df.to_csv("message_df_raw.csv")

        # process the df messages
        if not self.msg_df.empty:

            total_rows = len(self.msg_df)
            print(f"Length of raw df {total_rows}")

            # show error message if date range contains too many messages
            if total_rows > MAX_DF_SIZE:
                self.exceeded_df_limit = True
            else:
                self.exceeded_df_limit = False
                self.msg_df = general_preprocessing(self.msg_df)
                print("preprocessed df messages")
                # aggregate the messages based on date
                self.msg_df = message_aggregation(self.msg_df)
                print(
                    f"length of df after message processing and aggregation: {len(self.msg_df)}"
                )

                if use_llm_summary:
                    try:
                        print("Getting LLM summaries")
                        self.llm_summarized_df = get_LLM_summaries(self.msg_df)
                        print(self.llm_summarized_df)
                    except:
                        print("Error producing LLM summaries")
                    self.llm_summarized_df = self.msg_df

            self.msg_df.to_csv("message_df_postprocessed.csv")

    # populate dataframe with messages from a single channel between specified start and end times
    def get_channel_messages(self, channel_name, actor_user_id):

        print(f"Getting messages for channel {channel_name}")
        channel_id = self.all_invited_conversations[channel_name]
        try:
            history = self.app.client.conversations_history(
                token=self.bot_token,
                channel=channel_id,
                oldest=self.str_datetime_to_unix(self.start_date),
                latest=self.str_datetime_to_unix(self.end_date, end=True),
                inclusive=True,
            )
            messages = history["messages"]
            has_more = history["has_more"]

            self.add_messages_to_df(messages, channel_name, channel_id, actor_user_id)

            # paging through time, each page contains maximum 100 messages
            while has_more:
                # page += 1
                prev_ts = messages[-1]["ts"]
                try:
                    history = self.app.client.conversations_history(
                        token=self.bot_token,
                        channel=channel_id,
                        oldest=self.str_datetime_to_unix(self.start_date),
                        latest=prev_ts,
                    )
                    messages = history["messages"]
                    has_more = history["has_more"]
                    self.add_messages_to_df(
                        messages, channel_name, channel_id, actor_user_id
                    )
                except SlackApiError as e:
                    if e.response["error"] == "ratelimited":
                        self.app.client.views_publish(
                            token=self.bot_token,
                            user_id=actor_user_id,
                            view=self.generate_homepage_view(
                                self.bot_token,
                                self.team_id,
                                self.actor_user_id,
                                slackapi_limit_exceeded=True,
                            ),
                        )
        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                self.app.client.views_publish(
                    token=self.bot_token,
                    user_id=actor_user_id,
                    view=self.generate_homepage_view(
                        self.bot_token,
                        self.team_id,
                        self.actor_user_id,
                        slackapi_limit_exceeded=True,
                    ),
                )

    def add_replies_to_df(self, replies, channel_name, channel_id):
        for reply in replies[1:]:  # exclude original message
            reply_dict = {}
            user_id = reply.get("user", None)
            if user_id in self.consented_users and "reply_count" not in reply:
                self.analysis_users_consented.add(user_id)
                ts = datetime.datetime.fromtimestamp(float(reply["ts"]))
                year = ts.year
                reply_dict["year"] = year
                reply_dict["channel_id"] = channel_id
                reply_dict["channel_name"] = channel_name
                reply_dict["user_id"] = user_id
                reply_dict["timestamp"] = ts
                reply_dict["text"] = str(reply["text"])
                reply_dict["replies_cnt"] = 0
                reacts = reply.get("reactions", None)

                if contains_link(reply["text"]):
                    attachment_record = {
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "attachment_type": "link",
                    }
                    self.attachments = pd.concat(
                        [
                            self.attachments if not self.attachments.empty else None,
                            pd.DataFrame([attachment_record]),
                        ],
                        ignore_index=True,
                    )
                if contains_attachment(reply):
                    attachment_record = {
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "attachment_type": "attachment",
                    }
                    self.attachments = pd.concat(
                        [
                            self.attachments if not self.attachments.empty else None,
                            pd.DataFrame([attachment_record]),
                        ],
                        ignore_index=True,
                    )

                reacts_cnt = 0
                if reacts:
                    for react in reacts:
                        reacts_cnt += react["count"]
                        for react_user_id in react["users"]:
                            record = {
                                "user_id": react_user_id,
                                "channel_id": channel_id,
                                "react_type": react["name"],
                                "timestamp": ts,
                            }
                            self.reactions = pd.concat(
                                [
                                    (
                                        self.reactions
                                        if not self.reactions.empty
                                        else None
                                    ),
                                    pd.DataFrame([record]),
                                ],
                                ignore_index=True,
                            )
                reply_dict["reacts_cnt"] = reacts_cnt
                self.msg_df = pd.concat(
                    [self.msg_df, pd.DataFrame([reply_dict])], ignore_index=True
                )

    def add_messages_to_df(self, msg_list, channel_name, channel_id, app_user_id):
        for msg in msg_list:
            # dict to store each message instance
            msg_dict = {}
            subtype = msg.get("subtype", "*")
            user_id = msg.get("user", None)
            if subtype != "channel_join" and user_id in self.consented_users:
                self.analysis_users_consented.add(user_id)
                ts = datetime.datetime.fromtimestamp(
                    float(msg["ts"])
                )  # convert unix timestamp to datetime
                year = ts.year
                msg_dict["year"] = year
                msg_dict["channel_id"] = channel_id
                msg_dict["channel_name"] = channel_name
                msg_dict["user_id"] = user_id
                msg_dict["timestamp"] = ts
                msg_dict["text"] = str(msg["text"])
                msg_dict["replies_cnt"] = msg.get("reply_count", 0)
                reacts = msg.get("reactions", None)
                reacts_cnt = 0

                if contains_link(msg_dict["text"]):
                    attachment_record = {
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "attachment_type": "link",
                    }
                    self.attachments = pd.concat(
                        [
                            self.attachments if not self.attachments.empty else None,
                            pd.DataFrame([attachment_record]),
                        ],
                        ignore_index=True,
                    )
                if contains_attachment(msg):
                    attachment_record = {
                        "user_id": user_id,
                        "channel_id": channel_id,
                        "attachment_type": "attachment",
                    }
                    self.attachments = pd.concat(
                        [
                            self.attachments if not self.attachments.empty else None,
                            pd.DataFrame([attachment_record]),
                        ],
                        ignore_index=True,
                    )

                if reacts:
                    for react in reacts:
                        reacts_cnt += react["count"]
                        for react_user_id in react["users"]:
                            react_record = {
                                "user_id": react_user_id,
                                "channel_id": channel_id,
                                "react_type": react["name"],
                                "timestamp": ts,
                            }
                            self.reactions = pd.concat(
                                [self.reactions, pd.DataFrame([react_record])],
                                ignore_index=True,
                            )
                msg_dict["reacts_cnt"] = reacts_cnt
                self.msg_df = pd.concat(
                    [self.msg_df, pd.DataFrame([msg_dict])], ignore_index=True
                )

                # add replies to the dataframe
                if "thread_ts" in msg:
                    # get a maximum of 100 replies
                    try:
                        replies = self.app.client.conversations_replies(
                            token=self.bot_token,
                            channel=channel_id,
                            ts=msg["thread_ts"],
                            limit=100,
                        )["messages"]
                        self.add_replies_to_df(replies, channel_name, channel_id)
                    except SlackApiError as e:
                        if e.response["error"] == "ratelimited":
                            self.app.client.views_publish(
                                token=self.bot_token,
                                user_id=app_user_id,
                                view=self.generate_homepage_view(
                                    self.bot_token,
                                    self.team_id,
                                    self.actor_user_id,
                                    slackapi_limit_exceeded=True,
                                ),
                            )

            elif (
                subtype != "channel_join" and user_id != None
            ):  # don't count bot users as unconsenting users
                # print(msg)
                self.consent_exclusions += 1

    def str_datetime_to_unix(self, str_time, end=False):
        dt = datetime.datetime.strptime(str_time, "%Y-%m-%d")
        if end:
            dt = dt + datetime.timedelta(hours=23, minutes=59, seconds=59)  # EOD
        # unix time
        unix = dt.timestamp()
        return unix

    def lsm_limit_img(self):
        fig, ax = plt.subplots(figsize=(15, 6))
        # Remove borders and axis
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Add text in the middle of the plot
        text = "LSM limit exceeded. Try again tomorrow."
        plt.text(
            0.5,
            0.5,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
            color="red",
            transform=ax.transAxes,
        )
        # save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.5)
        buf.seek(0)
        plt.close(fig)
        return buf

    def create_lsm_vis(self):
        # TODO: add look and remove the hard coded value to self.msg_df
        # get lsm values and generate image
        self.lsm_df = get_LIWC_values(self.msg_df, self.team_id)  # ACTUAL
        if self.lsm_df.empty:
            return self.lsm_limit_img()
        # self.lsm_df = get_LIWC_values(pd.read_csv("message_df_tiny.csv")) # FOR TESTING
        # self.lsm_df.to_csv("lsm_df_after_liwc.csv")
        self.lsm_df = compute_lsm_scores(self.lsm_df)
        if self.lsm_df.empty:  # no valid pairs of users on each channel-day
            return self.lsm_limit_img()
        self.lsm_df = grouped_lsm_scores(self.lsm_df)
        # lsm_df_avg = moving_avg_lsm(self.lsm_df, WINDOW_SIZE)
        lsm_image = per_channel_vis_LSM(self.lsm_df)
        return lsm_image

    def create_lsa_visualizations(self, method, use_llm_summary=False):
        if use_llm_summary:
            df = self.llm_summarized_df
        else:
            df = self.msg_df
        if method == "cosine_sim":
            self.lsa_cosine_df = compute_LSA_analysis(
                df,
                step=2,
                method=method,
            )
            lsa_cosine_img = LSA_cosine_sim_vis(self.lsa_cosine_df)
            return lsa_cosine_img
        elif method == "semantic_coherence":
            self.lsa_coherence_df = compute_LSA_analysis(
                df,
                step=2,
                method=method,
            )
            lsa_coherence_img = LSA_coherence_vis(self.lsa_coherence_df)
            return lsa_coherence_img

    def create_pp_embedding_vis(self):
        self.pp_embedding_df = get_embeddings(self.msg_df)
        self.pp_embedding_df = pp_vis_preprocessing(self.pp_embedding_df)
        pp_embedding_img = vis_perperson(self.pp_embedding_df)
        return pp_embedding_img

    def create_group_embedding_vis(self, use_llm_summary=False):
        if use_llm_summary:
            self.group_embedding_df = get_embeddings(self.llm_summarized_df)
        else:
            self.group_embedding_df = get_embeddings(self.msg_df)
        self.group_embedding_df = pairwise_comparison(self.group_embedding_df)
        self.group_embedding_df = grouped_cos_sims(self.group_embedding_df)
        group_embedding_img = vis_group(self.group_embedding_df)
        return group_embedding_img

    # make sure each channel has enough messages to apply moving average
    def enough_msgs(self):
        if self.msg_df.empty:
            return False
        for _, msgs in self.msg_df.groupby("channel_id"):
            if len(msgs) < MIN_DF_SIZE:
                return False
        return True

    def generate_homepage_view(
        self,
        bot_token,
        team_id,
        actor_user_id,
        loading=False,
        vis_error=False,
        slackapi_limit_exceeded=False,
    ):
        slack_limit_exceeded_block = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":exclamation: The SlackAPI rate limit has been exceeded. Please try again later with a smaller date range/number of conversations.",
                },
            },
        ]

        loading_block = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":hourglass_flowing_sand: Generating visualizations...",
                },
            },
        ]

        vis_error_block = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":exclamation: There was an error with visualization generation.",
                },
            },
        ]

        invalid_selection_block = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":exclamation: Please select channels and/or a valid date range containing more messages (each channel selected should contain at least {MIN_DF_SIZE} active channel-days for analysis). You may need to manually re-select dates if the app homepage has just been opened. For the selected conversations and date range, {self.consent_exclusions} users have not yet consented.",
                },
            },
        ]

        exceed_msg_limit_block = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": ":exclamation: The channels and/or date range you selected exceeds the maximum allowable number of messages for analysis. Please select a smaller date range or fewer number of channels.",
                },
            },
        ]

        view = {
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
        }

        if self.exceeded_df_limit:
            view["blocks"].extend(exceed_msg_limit_block)
        # insufficient messages to generate analysis, need enough messages in EACH channel
        elif not loading and not self.enough_msgs():
            view["blocks"].extend(invalid_selection_block)
        elif vis_error:
            view["blocks"].extend(vis_error_block)
        elif slackapi_limit_exceeded:
            view["blocks"].extend(slack_limit_exceeded_block)
        elif loading:
            view["blocks"].extend(loading_block)
        else:
            vis_blocks = [
                # {
                #     "type": "section",
                #     "text": {
                #         "type": "mrkdwn",
                #         "text": "Note that the generation of the following visualizations are rate-limited. If visualizations stop updating, please wait and try again later.",
                #     },
                # },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"_The number of messages excluded due to unconsenting users is: {self.consent_exclusions}_",
                    },
                },
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Knowledge Convergence",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "This graph relies on a method that describes each message as a combination of different topics (Dong, 2005). It assumes that the team’s 'knowledge' can be described by the topics represented by all messages sent during the entire time period. At each timestamp, it calculates how similar each individual's cumulative “knowledge” is to the team’s entire “knowledge.” A value close to 1 shows that the individual or team knowledge at time T is similar to the team’s total knowledge. Of interest in this graph is how quickly the team reaches it’s total 'knowledge'.",
                    },
                },
                {
                    "type": "image",
                    "block_id": "lsa_cosine_mg",
                    "image_url": f"{URI}/lsa_cosine_image?token={bot_token}&team_id={team_id}&actor_user_id={actor_user_id}&t={str(time.time())}",
                    "alt_text": "LSA Cosine Similarity",
                },
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Semantic Coherence",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "This graph relies on the same method above that describes each message as a combination of different topics (Dong, 2005).  Semantic coherence is a team-wide metric that measures how similar each message is (to what extent each message discusses the same topic), or how much each message represents similar knowledge. A value closer to one means that each person discusses each topic at similar frequencies, whereas a value closer to zero means each person is discussing different topics.",
                    },
                },
                {
                    "type": "image",
                    "block_id": "lsa_coherence_img",
                    "image_url": f"{URI}/lsa_coherence_image?token={bot_token}&team_id={team_id}&actor_user_id={actor_user_id}&t={str(time.time())}",
                    "alt_text": "LSA Semantic Coherence",
                },
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Embedding Space Visualization",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "This graph uses embedding spaces, which is a method that transforms words into vectors, encoding the semantic meaning of the word, such that similar words have similar vectors (Reimers, Gurevych, 2019). This graph represents the average distance (similarity) in word use across all combinations of pairs within the team. Unlike the other methods, this method acknowledges that team members might use different, but similar, words to mean the same thing.",
                    },
                },
                {
                    "type": "image",
                    "block_id": "group_embedding_img",
                    "image_url": f"{URI}/group_embedding_image?token={bot_token}&team_id={team_id}&actor_user_id={actor_user_id}&t={str(time.time())}",
                    "alt_text": "Group Embedding Space Visualization",
                },
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Language Style Matching",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "This graph measures the extent to which your team uses shared language, measured by how frequently each team member uses specific categories of words (Ireland and Pennebaker, 2010).  The y-axis represents the average similarity in frequency across all combinations of pairs within the team. A value of 1 represents identical language style and a value of 0 represents dissimilar language style across the team.",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Note: the maximum number of words analyzed per day for this method is 20,000. The visualization will stop generating until the next day if you have reached the limit.",
                    },
                },
                {
                    "type": "image",
                    "block_id": "lsm_img",
                    "image_url": f"{URI}/lsm_image?token={bot_token}&team_id={team_id}&actor_user_id={actor_user_id}&t={str(time.time())}",
                    "alt_text": "Language Style Matching Graph",
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
            view["blocks"].extend(vis_blocks)

        return view
