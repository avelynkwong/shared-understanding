from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_data import SlackData

app = App()
handler = SlackRequestHandler(app)
slack_data = SlackData(app, None)


@app.event("app_home_opened")
def load_homepage(client, event, logger, context):

    slack_data.bot_token = context.bot_token
    slack_data.find_conversations()

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


# INTERACTIVE COMPONENTS


# update the start date
@app.block_action("startdate_picked")
def set_start_date(ack, body):
    ack()
    slack_data.start_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis
    slack_data.update_dataframe()


# update the end date
@app.block_action("enddate_picked")
def set_end_date(ack, body):
    ack()
    slack_data.end_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis
    slack_data.update_dataframe()


# determine the list of conversations that the slack app has access to
@app.options("select_conversations")
def list_conversations(ack):
    # list of conversations app has access to
    conv_names = [
        {"text": {"type": "plain_text", "text": f"# {c}"}, "value": c}
        for c in slack_data.all_conversations.keys()
    ]
    ack({"options": conv_names})


# update the selected conversations
@app.action("select_conversations")
def select_conversations(ack, body):
    ack()
    selected_convs = body["actions"][0]["selected_options"]
    selected_conv_names = [c["value"] for c in selected_convs]
    slack_data.selected_conversations = selected_conv_names
    # update homescreen with selected conversations' analysis
    slack_data.update_dataframe()
    # update homepage
    client = WebClient(token=context.bot_token)
    try:
        client.views_publish(
            user_id=body["user"]["id"],
            view=slack_data.generate_homepage_view(context.bot_token),
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.action("consent_yes")
def add_consented_users(ack, body, context):
    ack()
    slack_data = user_data[context.bot_token]
    slack_data.consented_users.add(body["user"]["id"])
    channel_id = body["channel"]["id"]
    user_name = body["user"]["username"]
    update_message(context.bot_token, app.client, channel_id, user_name)


@app.action("consent_no")
def remove_consented_users(ack, body, context):
    ack()
    slack_data = user_data[context.bot_token]
    slack_data.consented_users.discard(body["user"]["id"])


# delete user data when uninstall occurs
@app.event("tokens_revoked")
def remove_user_data(event):
    revoked_tokens = event["tokens"]
    print(f"Tokens revoked: {revoked_tokens}, removing slack data...")
    # for t in revoked_tokens:
    #     user_data.pop(t, None)


# API ENDPOINTS
from fastapi import FastAPI, Request, Response

api = FastAPI()


@api.post("/slack/events")
async def endpoint(req: Request):
    return await handler.handle(req)


@api.get("/slack/install")
async def install(req: Request):
    return await handler.handle(req)


@api.get("/slack/oauth_redirect")
async def oauth_redirect(req: Request):
    return await handler.handle(req)


@api.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)


# API endpoint for posting the list of conversations to choose from
@api.post("/slack/options")
async def slack_options(request: Request):
    return await handler.handle(request)


@api.post("/slack/interactions")
async def slack_interactions(request: Request):
    return await handler.handle(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=80)
