from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_data import SlackData
from dotenv import load_dotenv
from slack_sdk import WebClient
from consent_form import update_message

# APP INITIALIZATION
load_dotenv()
app = App()
handler = SlackRequestHandler(app)
user_data = {}


# INTERACTIVE COMPONENTS


@app.event("app_home_opened")
def load_homepage(client, event, logger, context):
    # create new object if it doesn't exist
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
    slack_data = user_data[context.bot_token]

    # update homepage
    try:
        client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view=slack_data.generate_homepage_view(context.bot_token),
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# update the start date
@app.block_action("startdate_picked")
def set_start_date(ack, body, context, logger):
    ack()
    slack_data = user_data[context.bot_token]
    slack_data.start_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis
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


# update the end date
@app.block_action("enddate_picked")
def set_end_date(ack, body, context, logger):
    ack()
    slack_data = user_data[context.bot_token]
    slack_data.end_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis
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


# determine the list of conversations that the slack app has access to
@app.options("select_conversations")
def list_conversations(ack, context):
    slack_data = user_data[context.bot_token]
    # list of conversations app has access to
    conv_names = [
        {"text": {"type": "plain_text", "text": f"# {c}"}, "value": c}
        for c in slack_data.all_conversations.keys()
    ]
    ack({"options": conv_names})


# update the selected conversations
@app.action("select_conversations")
def select_conversations(ack, body, context, logger):
    ack()
    slack_data = user_data[context.bot_token]
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
    for t in revoked_tokens:
        user_data.pop(t, None)


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


@api.post("/slack/options")
async def slack_options(request: Request):
    return await handler.handle(request)


@api.post("/slack/interactions")
async def slack_interactions(request: Request):
    return await handler.handle(request)


# generate an image served at a url
@api.get("/test_image")
async def get_image(token: str, t: str):
    slack_data = user_data[token]
    slack_data = user_data[token]
    slack_data.generate_image()

    # Return the image as a response
    return Response(content=slack_data.test_image.read(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=80)
