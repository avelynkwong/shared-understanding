import os
from slack_bolt import App
from classes import SlackData
from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi import SlackRequestHandler

# app init
bot_token = os.environ.get("SLACK_BOT_TOKEN")
app = App(token=bot_token, signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))
fastapi_app = FastAPI()
handler = SlackRequestHandler(app)

# slack data object (global)
slack_data = SlackData(app)


# interactive components
@app.block_action("startdate_picked")
def set_start_date(ack, body):
    ack()
    slack_data.start_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis


@app.block_action("enddate_picked")
def set_end_date(ack, body):
    ack()
    slack_data.end_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis


@app.options("select_conversations")
def list_conversations(ack):
    # list of conversations app has access to
    conversations = app.client.users_conversations(token=bot_token)["channels"]
    conversations = {c["name"]: c["id"] for c in conversations}
    slack_data.all_conversations = conversations
    conv_names = [
        {"text": {"type": "plain_text", "text": f"# {c}"}, "value": c}
        for c in conversations.keys()
    ]
    ack({"options": conv_names})
    # print("The slack app has access to the following conversations: ")
    # print(conversations)


@app.block_action("select_conversations")
def select_conversations(ack, body):
    ack()
    selected_convs = body["actions"][0]["selected_options"]
    selected_conv_names = [c["value"] for c in selected_convs]
    print(selected_conv_names)
    app.selected_conversations = selected_conv_names


@app.event("app_home_opened")
def load_homepage(client, event, logger):
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
                                "initial_date": slack_data.start_date,
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "Select a date",
                                    "emoji": True,
                                },
                                "action_id": "startdate_picked",
                            },
                            {
                                "type": "datepicker",
                                "initial_date": slack_data.end_date,
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "Select a date",
                                    "emoji": True,
                                },
                                "action_id": "enddate_picked",
                            },
                        ],
                    },
                ],
            },
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# API endpoints
@fastapi_app.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)


@fastapi_app.post("/slack/options")
async def slack_options(request: Request):
    return await handler.handle(request)


@fastapi_app.post("/slack/interactions")
async def slack_interactions(request: Request):
    return await handler.handle(request)


# Ready? Start your app!
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(fastapi_app, host="0.0.0.0", port=80)

    # app.start(port=int(os.environ.get("PORT", 80)))


response = {
    "type": "block_actions",
    "user": {
        "id": "U070EJ574TS",
        "username": "avelyn.wong",
        "name": "avelyn.wong",
        "team_id": "T7HD0J5GF",
    },
    "api_app_id": "A074F1LFQJK",
    "token": "NGkEYf0avFoPdZTB1gLhUSx0",
    "container": {"type": "view", "view_id": "V075A9CHFDH"},
    "trigger_id": "7193711691442.255442617559.d1e61ec0c5883a55f2a93d9fb71a4838",
    "team": {"id": "T7HD0J5GF", "domain": "olechowski-lab"},
    "enterprise": None,
    "is_enterprise_install": False,
    "view": {
        "id": "V075A9CHFDH",
        "team_id": "T7HD0J5GF",
        "type": "home",
        "blocks": [
            {
                "type": "header",
                "block_id": "ep6R5",
                "text": {
                    "type": "plain_text",
                    "text": "Welcome to Shared Understanding Homepage",
                    "emoji": True,
                },
            },
            {"type": "divider", "block_id": "xOwmu"},
            {
                "type": "section",
                "block_id": "section678",
                "text": {
                    "type": "mrkdwn",
                    "text": "Please select the conversations you would like to analyze.",
                    "verbatim": False,
                },
                "accessory": {
                    "type": "multi_external_select",
                    "action_id": "select_conversations",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Select items",
                        "emoji": True,
                    },
                    "min_query_length": 1,
                },
            },
            {
                "type": "section",
                "block_id": "JRUAH",
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
                        "action_id": "startdate_picked",
                        "initial_date": "2020-01-01",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select a date",
                            "emoji": True,
                        },
                    },
                    {
                        "type": "datepicker",
                        "action_id": "enddate_picked",
                        "initial_date": "2024-01-01",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select a date",
                            "emoji": True,
                        },
                    },
                ],
            },
        ],
        "private_metadata": "",
        "callback_id": "home_view",
        "state": {
            "values": {
                "actions1": {
                    "startdate_picked": {
                        "type": "datepicker",
                        "selected_date": "2020-01-01",
                    },
                    "enddate_picked": {
                        "type": "datepicker",
                        "selected_date": "2024-01-01",
                    },
                },
                "section678": {
                    "select_conversations": {
                        "type": "multi_external_select",
                        "selected_options": [
                            {
                                "text": {
                                    "type": "plain_text",
                                    "text": "# social-felix_the_cat",
                                    "emoji": True,
                                },
                                "value": "social-felix_the_cat",
                            }
                        ],
                    }
                },
            }
        },
        "hash": "1717014894.7yiVYWT4",
        "title": {"type": "plain_text", "text": "View Title", "emoji": True},
        "clear_on_close": False,
        "notify_on_close": False,
        "close": None,
        "submit": None,
        "previous_view_id": None,
        "root_view_id": "V075A9CHFDH",
        "app_id": "A074F1LFQJK",
        "external_id": "",
        "app_installed_team_id": "T7HD0J5GF",
        "bot_id": "B074FKDTNBZ",
    },
    "actions": [
        {
            "type": "multi_external_select",
            "action_id": "select_conversations",
            "block_id": "section678",
            "selected_options": [
                {
                    "text": {
                        "type": "plain_text",
                        "text": "# social-felix_the_cat",
                        "emoji": True,
                    },
                    "value": "social-felix_the_cat",
                }
            ],
            "initial_options": [],
            "placeholder": {
                "type": "plain_text",
                "text": "Select items",
                "emoji": True,
            },
            "action_ts": "1717014973.933609",
        }
    ],
}
