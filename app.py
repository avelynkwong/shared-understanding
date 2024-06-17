from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_data import SlackData
from dotenv import load_dotenv
from slack_sdk import WebClient
from consent_form import update_message
from questionnaire_form import get_questionnaire
import os
from starlette.responses import RedirectResponse
from slack_bolt.oauth.oauth_settings import OAuthSettings

from slack_sdk.oauth.installation_store import FileInstallationStore, Installation
from slack_sdk.oauth.state_store import FileOAuthStateStore
from oauth.custom_installation_store import CustomFileInstallationStore
from oauth.custom_state_store import CustomFileOAuthStateStore

load_dotenv()
BOT_SCOPES = os.getenv("BOT_SCOPES")
USER_SCOPES = os.getenv("USER_SCOPES")
CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
REDIRECT_URI = os.getenv("SLACK_REDIRECT_URI")


# OAUTH SETUP
# installation_store = FileInstallationStore(base_dir="./data/installations")
installation_store = CustomFileInstallationStore()
# state_store = FileOAuthStateStore(expiration_seconds=300, base_dir="./data/states")
state_store = CustomFileOAuthStateStore(expiration_seconds=1000)
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


# UTILS


# create new slack data object if non-existent, send consent form if new installation
def get_slack_data(user_id, app, bot_token, enterprise_id, team_id):
    if not user_id in user_data:
        user_data[user_id] = SlackData(
            app,
            bot_token,
        )
        # send consent form if new installation
        if not installation_store.find_installation(
            user_id=user_id, enterprise_id=enterprise_id, team_id=team_id
        ):
            user_data[user_id].send_consent_form()

        # TODO: need to store old consented users in a database and retrieve them
        # if for some reason the app restarts/crashes, we need to recover who consented
        # to avoid sending the consent form again

    return user_data[user_id]


# INTERACTIVE COMPONENTS


@app.event("app_home_opened")
def load_homepage(client, context):

    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )

    # update homepage
    try:
        client.views_publish(
            token=context.bot_token,
            # the user that opened your app's app home
            user_id=context.user_id,
            # the view object that appears in the app home
            view=slack_data.generate_homepage_view(
                context.user_id,
                context.bot_token,
                context.enterprise_id,
                context.team_id,
            ),
        )
    except Exception as e:
        print(f"Error publishing home tab: {e}")


# update the start date
@app.block_action("startdate_picked")
def set_start_date(ack, body, context, logger):
    ack()
    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )
    slack_data.start_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis
    slack_data.update_dataframe()
    # update homepage
    try:
        app.client.views_publish(
            token=context.bot_token,
            user_id=body["user"]["id"],
            view=slack_data.generate_homepage_view(
                context.user_id,
                context.bot_token,
                context.enterprise_id,
                context.team_id,
            ),
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# update the end date
@app.block_action("enddate_picked")
def set_end_date(ack, body, context, logger):
    ack()
    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )
    slack_data.end_date = body["actions"][0]["selected_date"]
    # update homescreen with correct timeframe's analysis
    slack_data.update_dataframe()
    # update homepage
    try:
        app.client.views_publish(
            token=context.bot_token,
            user_id=context.user_id,
            view=slack_data.generate_homepage_view(
                context.user_id,
                context.bot_token,
                context.enterprise_id,
                context.team_id,
            ),
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# determine the list of conversations that the slack app has access to
@app.options("select_conversations")
def list_conversations(ack, context):
    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )
    # list of conversations app has access to
    slack_data.get_invited_conversations()
    conv_names = [
        {"text": {"type": "plain_text", "text": f"# {c}"}, "value": c}
        for c in slack_data.all_invited_conversations.keys()
    ]
    ack({"options": conv_names})


# update the selected conversations
@app.action("select_conversations")
def select_conversations(ack, body, context, logger):
    ack()
    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )
    selected_convs = body["actions"][0]["selected_options"]
    selected_conv_names = [c["value"] for c in selected_convs]
    slack_data.selected_conversations = selected_conv_names
    # update homescreen with selected conversations' analysis
    slack_data.update_dataframe()
    # update homepage
    try:
        app.client.views_publish(
            token=context.bot_token,
            user_id=context.user_id,
            view=slack_data.generate_homepage_view(
                context.user_id,
                context.bot_token,
                context.enterprise_id,
                context.team_id,
            ),
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.action("consent_yes")
def add_consented_users(ack, body, context):
    ack()
    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )
    slack_data.consented_users.add(body["user"]["id"])
    channel_id = body["channel"]["id"]
    user_name = body["user"]["username"]
    update_message(context.bot_token, app.client, channel_id, user_name)


@app.action("consent_no")
def remove_consented_users(ack, body, context):
    ack()
    slack_data = get_slack_data(
        context.user_id, app, context.bot_token, context.enterprise_id, context.team_id
    )
    slack_data.consented_users.discard(body["user"]["id"])


# listen for submission button click, open quest
@app.action("submit_analysis")
def display_questionnaire(ack, body, context):
    ack()
    questionnaire = get_questionnaire()
    app.client.views_open(
        token=context.bot_token, trigger_id=body["trigger_id"], view=questionnaire
    )


# listen for questionnaire submission button click
@app.view("questionnaire_form")
def handle_questionnaire_submission(ack, body):
    ack()
    values = body["view"]["state"]["values"]
    num_members = int(values["num_members"]["name_input"]["value"])
    industry = values["industry"]["industry"]["value"]
    work_type = values["work_type"]["industry_select"]["selected_option"]["value"]
    # TODO: submit questionnaire to cloud DB
    print(f"Questionnaire submitted! Values: {num_members, industry, work_type}")


# delete user data when uninstall occurs
@app.event("tokens_revoked")
def remove_user_data(event, body, context):
    revoked_users = event["tokens"]
    team_id = body["team_id"]
    enterprise_id = context["authorize_result"]["enterprise_id"]
    print(f"Tokens revoked: {revoked_users}, removing slack data...")
    for oauth_user in revoked_users["oauth"]:
        user_data.pop(oauth_user, None)
        installation_store.delete_installation(
            user_id=oauth_user, enterprise_id=enterprise_id, team_id=team_id
        )
    for bot_user in revoked_users["bot"]:
        user_data.pop(bot_user, None)
        installation_store.delete_installation(
            user_id=bot_user, enterprise_id=enterprise_id, team_id=team_id
        )


# API ENDPOINTS
from fastapi import FastAPI, Request, Response

api = FastAPI()


@api.post("/slack/events")
async def event(req: Request):
    return await handler.handle(req)


@api.get("/slack/install")
async def install(req: Request):
    state = state_store.issue()
    url = f"https://slack.com/oauth/v2/authorize?state={state}&scope={BOT_SCOPES}&user_scope={USER_SCOPES}&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    return RedirectResponse(url)


@api.get("/slack/oauth_redirect")
async def oauth_redirect(req: Request):
    state = req.query_params.get("state")
    if state_store.consume(state):
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

            # installation = Installation(
            #     app_id=oauth_response.get("app_id"),
            #     enterprise_id=installed_enterprise.get("id"),
            #     enterprise_name=installed_enterprise.get("name"),
            #     enterprise_url=enterprise_url,
            #     team_id=installed_team.get("id"),
            #     team_name=installed_team.get("name"),
            #     bot_token=bot_token,
            #     bot_id=bot_id,
            #     bot_user_id=oauth_response.get("bot_user_id"),
            #     bot_scopes=oauth_response.get("scope"),  # comma-separated string
            #     user_id=installer.get("id"),
            #     user_token=installer.get("access_token"),
            #     user_scopes=installer.get("user_scope"),  # comma-separated string
            #     incoming_webhook_url=incoming_webhook.get("url"),
            #     incoming_webhook_channel=incoming_webhook.get("channel"),
            #     incoming_webhook_channel_id=incoming_webhook.get("channel_id"),
            #     incoming_webhook_configuration_url=incoming_webhook.get(
            #         "configuration_url"
            #     ),
            #     is_enterprise_install=is_enterprise_install,
            #     token_type=oauth_response.get("token_type"),
            # )

            installation = {
                "app_id": oauth_response.get("app_id"),
                "enterprise_id": installed_enterprise.get("id"),
                "enterprise_name": installed_enterprise.get("name"),
                "enterprise_url": enterprise_url,
                "team_id": installed_team.get("id"),
                "team_name": installed_team.get("name"),
                "bot_token": bot_token,
                "bot_id": bot_id,
                "bot_user_id": oauth_response.get("bot_user_id"),
                "bot_scopes": oauth_response.get("scope"),  # comma-separated string
                "user_id": installer.get("id"),
                "user_token": installer.get("access_token"),
                "user_scopes": installer.get("user_scope"),  # comma-separated string
                "incoming_webhook_url": incoming_webhook.get("url"),
                "incoming_webhook_channel": incoming_webhook.get("channel"),
                "incoming_webhook_channel_id": incoming_webhook.get("channel_id"),
                "incoming_webhook_configuration_url": incoming_webhook.get(
                    "configuration_url"
                ),
                "is_enterprise_install": is_enterprise_install,
                "token_type": oauth_response.get("token_type"),
            }

            installation_store.save(installation)

            # create slack_data object and send consent form
            user_data[installation["user_id"]] = SlackData(
                app, installation["bot_token"]
            )
            user_data[installation["user_id"]].send_consent_form()
            # user_data[installation.user_id] = SlackData(app, installation.bot_token)
            # user_data[installation.user_id].send_consent_form()

            return {"message": "Thanks for installing!"}
        else:
            return {
                "message": "Installation Failed",
                "error": oauth_response.get("error"),
            }
    else:
        return {
            "message": "The state value is already expired",
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
async def get_image(user_id: str, token: str, enterprise_id: str, team_id: str, t: str):
    slack_data = get_slack_data(user_id, app, token, enterprise_id, team_id)
    slack_data.generate_image()

    # Return the image as a response
    return Response(content=slack_data.test_image.read(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=80)
