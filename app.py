from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_data import SlackData
from dotenv import load_dotenv
from slack_sdk import WebClient
from consent_form import update_message
import os
from starlette.responses import RedirectResponse
import httpx
from slack_bolt.oauth.oauth_settings import OAuthSettings
from slack_sdk.oauth.installation_store import FileInstallationStore, Installation
from slack_sdk.oauth.state_store import FileOAuthStateStore
from slack_sdk.oauth import AuthorizeUrlGenerator

load_dotenv()
BOT_SCOPES = os.getenv("BOT_SCOPES")
USER_SCOPES = os.getenv("USER_SCOPES")
CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
REDIRECT_URI = os.getenv("SLACK_REDIRECT_URI")


# OAUTH SETUP
installation_store = FileInstallationStore(base_dir="./data/installations")
state_store = FileOAuthStateStore(expiration_seconds=300, base_dir="./data/states")
oauth_settings = OAuthSettings(
    client_id=os.getenv("SLACK_CLIENT_ID"),
    client_secret=os.getenv("SLACK_CLIENT_SECRET"),
    scopes=BOT_SCOPES,
    user_scopes=USER_SCOPES,
    installation_store=installation_store,
    state_store=state_store,
)

# APP INITIALIZATION
app = App(signing_secret=SLACK_SIGNING_SECRET, oauth_settings=oauth_settings)
handler = SlackRequestHandler(app)
user_data = {}

# INTERACTIVE COMPONENTS


@app.event("app_home_opened")
def load_homepage(client, context):

    # create new object if it doesn't exist
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
    slack_data = user_data[context.bot_token]

    # update homepage
    try:
        client.views_publish(
            # the user that opened your app's app home
            user_id=context.user_id,
            # the view object that appears in the app home
            view=slack_data.generate_homepage_view(context.bot_token),
        )
    except Exception as e:
        print(f"Error publishing home tab: {e}")


# update the start date
@app.block_action("startdate_picked")
def set_start_date(ack, body, context, logger):
    ack()
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
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
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
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
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
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
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
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
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
    slack_data = user_data[context.bot_token]
    slack_data.consented_users.add(body["user"]["id"])
    channel_id = body["channel"]["id"]
    user_name = body["user"]["username"]
    update_message(context.bot_token, app.client, channel_id, user_name)


@app.action("consent_no")
def remove_consented_users(ack, body, context):
    ack()
    if not context.bot_token in user_data:
        user_data[context.bot_token] = SlackData(app, context.bot_token)
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
async def event(req: Request):
    return await handler.handle(req)


@api.get("/slack/install")
async def install(req: Request):
    state = state_store.issue()
    url = f"https://slack.com/oauth/v2/authorize?scope={BOT_SCOPES}&user_scope={USER_SCOPES}&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    return RedirectResponse(url)


@api.get("/slack/oauth_redirect")
async def oauth_redirect(req: Request):
    code = req.query_params.get("code")
    client = WebClient()
    oauth_response = client.oauth_v2_access(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        code=code,
    )
    if oauth_response.get("ok"):
        installed_enterprise = oauth_response.get("enterprise") or {}
        is_enterprise_install = oauth_response.get("is_enterprise_install")
        installed_team = oauth_response.get("team") or {}
        installer = oauth_response.get("authed_user") or {}
        incoming_webhook = oauth_response.get("incoming_webhook") or {}
        bot_token = oauth_response.get("access_token")
        # NOTE: oauth.v2.access doesn't include bot_id in response
        bot_id = None
        enterprise_url = None
        if bot_token is not None:
            auth_test = client.auth_test(token=bot_token)
            bot_id = auth_test["bot_id"]
            if is_enterprise_install is True:
                enterprise_url = auth_test.get("url")

        installation = Installation(
            app_id=oauth_response.get("app_id"),
            enterprise_id=installed_enterprise.get("id"),
            enterprise_name=installed_enterprise.get("name"),
            enterprise_url=enterprise_url,
            team_id=installed_team.get("id"),
            team_name=installed_team.get("name"),
            bot_token=bot_token,
            bot_id=bot_id,
            bot_user_id=oauth_response.get("bot_user_id"),
            bot_scopes=oauth_response.get("scope"),  # comma-separated string
            user_id=installer.get("id"),
            user_token=installer.get("access_token"),
            user_scopes=installer.get("scope"),  # comma-separated string
            incoming_webhook_url=incoming_webhook.get("url"),
            incoming_webhook_channel=incoming_webhook.get("channel"),
            incoming_webhook_channel_id=incoming_webhook.get("channel_id"),
            incoming_webhook_configuration_url=incoming_webhook.get(
                "configuration_url"
            ),
            is_enterprise_install=is_enterprise_install,
            token_type=oauth_response.get("token_type"),
        )

        installation_store.save(installation)
        return {"message": "Thanks for installing!"}
    else:
        return {
            "message": "Installation Failed",
            "error": oauth_response.get("error"),
        }


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


# generate an image served at a url
@api.get("/test_image")
async def get_image(token: str, t: str):
    if not token in user_data:
        user_data[token] = SlackData(app, token)
    slack_data = user_data[token]
    slack_data = user_data[token]
    slack_data.generate_image()

    # Return the image as a response
    return Response(content=slack_data.test_image.read(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=80)


h = {
    "is_enterprise_install": False,
    "team_id": "T7HD0J5GF",
    "user_id": "U070EJ574TS",
    "actor_team_id": "T7HD0J5GF",
    "actor_user_id": "U070EJ574TS",
    "channel_id": "D075NU6GGE5",
    "logger": 22,
    "token": "xoxb-255442617559-7168693843828-wi0OXyiizJ9AP5XX36ssVNQz",
    "client": 8,
    "authorize_result": {
        "enterprise_id": None,
        "team_id": "T7HD0J5GF",
        "team": "Ready Lab",
        "url": "https://olechowski-lab.slack.com/",
        "bot_user_id": "U074YLDQTQC",
        "bot_id": "B074FKDTNBZ",
        "bot_token": "xoxb-255442617559-7168693843828-wi0OXyiizJ9AP5XX36ssVNQz",
        "bot_scopes": [
            "channels:history",
            "channels:read",
            "chat:write",
            "chat:write.public",
        ],
        "user_id": "U070EJ574TS",
        "user": "avelyn.wong",
        "user_token": "xoxp-255442617559-7014617242944-7244456330851-ab7c859cca819546d27a56f73da56493",
        "user_scopes": ["channels:history", "chat:write"],
    },
    "bot_id": "B074FKDTNBZ",
    "bot_user_id": "U074YLDQTQC",
    "bot_token": "xoxb-255442617559-7168693843828-wi0OXyiizJ9AP5XX36ssVNQz",
    "user_token": "xoxp-255442617559-7014617242944-7244456330851-ab7c859cca819546d27a56f73da56493",
    "ack": 3,
}
