import os
import json
import time
from slack_bolt import App
import pandas as pd
from collections import defaultdict

def add_messages_to_df(df, msg_list, channel_id):
    # dict to store each message instance
    msg_dict = defaultdict(None)
    for msg in msg_list:
        msg_dict['year'] = None
        msg_dict['channel_id'] = channel_id
        msg_dict['channel_name'] = conversations[channel_id]
        msg_dict['user_id'] = msg['user']
        msg_dict['timestamp'] = msg['ts']
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

    # page = 0
    # with open(f'{filename}_{page}.json', 'w') as f:
    #     json.dump(messages, f)


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
    
        # with open(f'{filename}_{page}.json', 'w') as f:
        #     json.dump(messages, f)
        return df


bot_token = os.environ.get("SLACK_BOT_TOKEN")

# Initialize your app with your bot token and signing secret
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

@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    try:
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
                "type": "section",
                "text": {
                "type": "mrkdwn",
                "text": "*Welcome to your _App's Home tab_* :tada:"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                "type": "mrkdwn",
                "text": "This button won't do much for now but you can set up a listener for it using the `actions()` method and passing its unique `action_id`. See an example in the `examples` folder within your Bolt app."
                }
            },
            {
                "type": "actions",
                "elements": [
                {
                    "type": "button",
                    "text": {
                    "type": "plain_text",
                    "text": "Click me!"
                    }
                }
                ]
            }
            ]
        }
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# Ready? Start your app!
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))