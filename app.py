import os
import time
from slack_bolt import App
import pandas as pd
from datetime import datetime

# globals
START_DATE='2020-01-01'
END_DATE='2024-01-01'

# utils
def add_messages_to_df(df, msg_list, channel_id):
    # dict to store each message instance
    msg_dict = {}
    for msg in msg_list:
        ts = msg['ts']
        year = datetime.fromtimestamp(float(ts)).year
        msg_dict['year'] = year
        msg_dict['channel_id'] = channel_id
        msg_dict['channel_name'] = conversations[channel_id]
        msg_dict['user_id'] = msg['user']
        msg_dict['timestamp'] = ts
        msg_dict['text'] = msg['text']
        msg_dict['replies_cnt'] = msg.get('reply_count', 0)
        reacts = msg.get('reactions', None)
        reacts_cnt = 0
        if reacts:
            for react in reacts:
                reacts_cnt += react['count']
        msg_dict['reacts_cnt'] = reacts_cnt
        df = pd.concat([df, pd.DataFrame([msg_dict])], ignore_index=True)
    return df

def get_paged_messages(df, bot_token, channel_id='C01BHE9FNQ6', start_time='0', end_time=str(time.time()), filename='dump'):

    history = app.client.conversations_history(
        token=bot_token,
        channel=channel_id,
        oldest=start_time,
        latest=end_time
    )
    messages = history['messages']
    has_more = history['has_more']
    
    df = add_messages_to_df(df, messages, channel_id)

    # paging through time, each page contains maximum 100 messages
    while has_more:
        # page += 1
        prev_ts = messages[-1]['ts']
        history = app.client.conversations_history(
            token=bot_token,
            channel=channel_id,
            oldest=start_time,
            latest=prev_ts
        )
        messages = history['messages']
        has_more = history['has_more']
        df = add_messages_to_df(df, messages, channel_id)
    
    return df

# app init
bot_token = os.environ.get("SLACK_BOT_TOKEN")
app = App(
    token=bot_token,
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# list of conversations slack app has access to
conversations = app.client.users_conversations(token=bot_token)['channels']
conversations = {c['id']: c['name'] for c in conversations}
print('The slack app has access to the following conversations: ')
print(conversations)

# df to store messages and metadata
msg_df = pd.DataFrame()
msg_df = get_paged_messages(msg_df, bot_token)
# print(msg_df)

# interactive components

@app.block_action("startdate_picked")
def set_start_date(ack, body, client, event, logger):
    ack()
    global START_DATE
    START_DATE = body['actions'][0]['selected_date']
    # update homescreen with correct timeframe

@app.block_action("enddate_picked")
def set_end_date(ack, body, client, event, logger):
    ack()
    global END_DATE
    END_DATE = body['actions'][0]['selected_date']
    # update homescreen with correct timeframe

    
@app.event("app_home_opened")
def load_homepage(client, event, logger):
    try:
        print(event)
        # views.publish is the method that your app uses to push a view to the Home tab
        client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view={
                "type": "home",
                "callback_id": "home_view",

                # body of the view
                "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Welcome to Shared Understanding Homepage",
                        "emoji": True
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Select a start date:"
                    },
                    "accessory": {
                        "type": "datepicker",
                        "initial_date": START_DATE,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select a date",
                            "emoji": True
                        },
                        "action_id": "startdate_picked"
                    }
		        },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Select an end date:"
                    },
                    "accessory": {
                        "type": "datepicker",
                        "initial_date": END_DATE,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select a date",
                            "emoji": True
                        },
                        "action_id": "enddate_picked"
                    }
                },
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")

# Ready? Start your app!
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))