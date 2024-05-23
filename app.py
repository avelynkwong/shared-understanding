import os
# Use the package we installed
from slack_bolt import App

# Initialize your app with your bot token and signing secret
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Add functionality here later
# @app.event("app_home_opened") etc.
channels = app.client.conversations_list(token=os.environ.get("SLACK_BOT_TOKEN"))['channels']
for c in channels:
    print(c['name'])

# Ready? Start your app!
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))