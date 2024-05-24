import os
import json
import time
from slack_bolt import App

def get_paged_messages(bot_token, channel_id='C01BHE9FNQ6', start_time='0', end_time=str(time.time()), filename='dump'):
    history = app.client.conversations_history(
        token=bot_token,
        channel=channel_id,
        oldest=start_time,
        latest=end_time
    )
    messages = history['messages']
    has_more = history['has_more']

    page = 0
    with open(f'{filename}_{page}.json', 'w') as f:
        json.dump(messages, f)

    # paging through time, each page contains maximum 100 messages
    # 
    while has_more:
        page += 1
        prev_ts = messages[-1]['ts']
        history = app.client.conversations_history(
            token=bot_token,
            channel=channel_id,
            oldest=start_time,
            latest=prev_ts
        )
        messages = history['messages']
        has_more = history['has_more']
        with open(f'{filename}_{page}.json', 'w') as f:
            json.dump(messages, f)


bot_token = os.environ.get("SLACK_BOT_TOKEN")

# Initialize your app with your bot token and signing secret
app = App(
    token=bot_token,
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Add functionality here later
# @app.event("app_home_opened") etc.
# channels = app.client.conversations_list(token=bot_token)['channels']
# for c in channels:
#     print(c['name'], c['id'])

get_paged_messages(bot_token)

# Ready? Start your app!
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))