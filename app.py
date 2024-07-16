from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from data.slack_data import SlackData
from dotenv import load_dotenv
from slack_sdk import WebClient
from forms.consent_form import post_consent_confirmation, post_dissent_confirmation
from forms.questionnaire_form import get_questionnaire
import os
from starlette.responses import RedirectResponse
from slack_bolt.oauth.oauth_settings import OAuthSettings

# oauth imports
from oauth.custom_installation_store import CustomFileInstallationStore
from oauth.custom_state_store import CustomFileOAuthStateStore

# db utils
from db.utils import *

# consent
from forms.consent_form import generate_consent_form

# get secrets from AWS
from get_secrets import get_secret

# for timestamp
import datetime

# for rate limit exceeded image
import matplotlib.pyplot as plt
import io

load_dotenv()
slack_secrets = get_secret("slack_app_secrets")
BOT_SCOPES = os.getenv("BOT_SCOPES")
USER_SCOPES = os.getenv("USER_SCOPES")
CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
CLIENT_SECRET = slack_secrets["SLACK_CLIENT_SECRET"]
SLACK_SIGNING_SECRET = slack_secrets["SLACK_SIGNING_SECRET"]
REDIRECT_URI = os.getenv("SLACK_REDIRECT_URI")

# OAUTH SETUP
# installation_store = FileInstallationStore(base_dir="./data/installations")
installation_store = CustomFileInstallationStore()
# state_store = FileOAuthStateStore(expiration_seconds=300, base_dir="./data/states")
state_store = CustomFileOAuthStateStore(expiration_seconds=800)
oauth_settings = OAuthSettings(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scopes=BOT_SCOPES,
    user_scopes=USER_SCOPES,
    installation_store=installation_store,
    state_store=state_store,
)

# APP INITIALIZATION
app = App(signing_secret=SLACK_SIGNING_SECRET, oauth_settings=oauth_settings)
handler = SlackRequestHandler(app)
workspace_data = {}


# UTILS


# create new slack data object if non-existent
def get_slack_data(app, bot_token, team_id):
    if not team_id in workspace_data:
        workspace_data[team_id] = SlackData(app, bot_token, team_id)
    return workspace_data[team_id]


# INTERACTIVE COMPONENTS


@app.event("app_home_opened")
def load_homepage(client, context):

    slack_data = get_slack_data(app, context.bot_token, context.team_id)
    slack_data.clear_analysis_data()
    slack_data.reset_dates()

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


# when slack app joins a channel, send consent form to members in that channel if they haven't received it yet
@app.event("member_joined_channel")
def send_consent_form(event, client, context):
    joined_user_id = event.get("user")
    bot_user_id = client.auth_test().get("user_id")
    form = generate_consent_form()

    # check whether the member who has joined the channel is the slack app
    if joined_user_id == bot_user_id:

        add_user_consent(context.team_id, bot_user_id)  # add bot as a consenting user

        channel_id = event["channel"]
        all_channel_users = []
        members_info = client.conversations_members(
            token=context.bot_token, channel=channel_id
        )
        user_ids = members_info["members"]
        all_channel_users.extend(user_ids)
        next_cursor = members_info["response_metadata"]["next_cursor"]

        # page through members in the channel
        while next_cursor:
            members_info = client.conversations_members(
                token=context.bot_token, channel=channel_id, cursor=next_cursor
            )
            user_ids = members_info["members"]
            all_channel_users.extend(user_ids)
            next_cursor = members_info["response_metadata"]["next_cursor"]

        consented_users = get_consented_users(context.team_id)
        # get all users that haven't consented yet
        non_consented_users = list(set(all_channel_users) - set(consented_users))
        # send consent forms
        for m in non_consented_users:
            # check if member is a bot
            is_bot = client.users_info(token=context.bot_token, user=m)["user"][
                "is_bot"
            ]
            # open a DM and send a message
            if not is_bot:
                response = client.conversations_open(token=context.bot_token, users=m)
                channel_id = response["channel"]["id"]
                client.chat_postMessage(
                    text="Slack Data Consent Form",
                    token=context.bot_token,
                    channel=channel_id,
                    blocks=form,
                )

    else:  # send consent form to new user who has joined
        is_bot = client.users_info(token=context.bot_token, user=joined_user_id)[
            "user"
        ]["is_bot"]
        # open a DM and send a message
        if not is_bot:
            print("Sending consent form to new user")
            response = client.conversations_open(
                token=context.bot_token, users=joined_user_id
            )
            channel_id = response["channel"]["id"]
            client.chat_postMessage(
                text="Slack Data Consent Form",
                token=context.bot_token,
                channel=channel_id,
                blocks=form,
            )


# update the start date
@app.block_action("startdate_picked")
def set_start_date(ack, body, context, logger):
    ack()
    slack_data = get_slack_data(app, context.bot_token, context.team_id)
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
    slack_data = get_slack_data(app, context.bot_token, context.team_id)
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
    slack_data = get_slack_data(app, context.bot_token, context.team_id)
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
    slack_data = get_slack_data(app, context.bot_token, context.team_id)
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
    add_user_consent(context.team_id, body["user"]["id"])
    channel_id = body["channel"]["id"]
    user_name = body["user"]["username"]
    post_consent_confirmation(context.bot_token, app.client, channel_id, user_name)


@app.action("consent_no")
def remove_consented_users(ack, body, context):
    ack()
    delete_user_consent(context.user_id)
    channel_id = body["channel"]["id"]
    user_name = body["user"]["username"]
    post_dissent_confirmation(context.bot_token, app.client, channel_id, user_name)


# listen for submission button click, open questionnaire
@app.action("submit_analysis")
def display_questionnaire(ack, body, context):
    ack()
    questionnaire = get_questionnaire()
    app.client.views_open(
        token=context.bot_token, trigger_id=body["trigger_id"], view=questionnaire
    )


# listen for questionnaire submission button click
@app.view("questionnaire_form")
def handle_questionnaire_submission(ack, body, context):
    ack()
    values = body["view"]["state"]["values"]
    team_size = values["team_size"]["team_size"]["selected_option"]["value"]
    team_duration = values["team_duration"]["team_duration"]["selected_option"]["value"]
    collab_type = values["collaboration_type"]["collaboration_type"]["selected_option"][
        "value"
    ]
    industry = values["industry"]["industry"]["value"]
    task_type = values["task_type"]["task_type"]["selected_option"]["value"]
    task_type_other = values["task_type_other"]["task_type_other"]["value"]
    if task_type_other:
        task_type = task_type_other
    slack_data = get_slack_data(app, context.bot_token, context.team_id)

    # add lsm results to the database
    ts = datetime.datetime.now()
    print(
        f"Submitting analysis values: {team_size, team_duration, collab_type, industry, task_type, ts, len(slack_data.analysis_users_consented)}"
    )
    add_analysis_db(
        context.team_id,
        team_size,
        team_duration,
        collab_type,
        industry,
        task_type,
        ts,
        len(slack_data.analysis_users_consented),
        "lsm",
        slack_data.lsm_df.to_json(orient="records"),
    )

    # delete the analysis-specific data to free memory
    slack_data.clear_analysis_data()


# delete user data when uninstall occurs
@app.event("tokens_revoked")
def remove_user_data(event, body, context):
    revoked_users = event["tokens"]
    team_id = body["team_id"]
    enterprise_id = context["authorize_result"]["enterprise_id"]
    print(f"Tokens revoked: {revoked_users}, removing slack data...")
    for oauth_user in revoked_users["oauth"]:
        workspace_data.pop(oauth_user, None)
        installation_store.delete_installation(
            user_id=oauth_user, enterprise_id=enterprise_id, team_id=team_id
        )
    for bot_user in revoked_users["bot"]:
        workspace_data.pop(bot_user, None)
        installation_store.delete_installation(
            user_id=bot_user, enterprise_id=enterprise_id, team_id=team_id
        )

    # remove consent from db
    delete_team_consent(team_id=team_id)


# API ENDPOINTS
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# rate limiter
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

api = FastAPI()


# configure rate limiter, limit by team
def get_team_id_key(request: Request) -> str:
    return request.query_params.get("team_id")


limiter = Limiter(key_func=get_team_id_key)
api.state.limiter = limiter
# api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
api.add_middleware(SlowAPIMiddleware)


@api.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    fig, ax = plt.subplots(figsize=(15, 6))
    # Remove borders and axis
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Add text in the middle of the plot
    text = "Rate limit exceeded.\nPlease try again later."
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
    return Response(content=buf.read(), media_type="image/png")


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

            # create slack_data object
            workspace_data[installation["team_id"]] = SlackData(
                app, installation["bot_token"], installation["team_id"]
            )

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


# API endpoint for posting the list of conversations to choose from
@api.post("/slack/options")
async def slack_options(request: Request):
    return await handler.handle(request)


@api.post("/slack/interactions")
async def slack_interactions(request: Request):
    return await handler.handle(request)


# will only be generate lsm visual if df_limit_exceeded is false
# since the block containing the image is only shown in that case
@api.get("/lsm_image")
@limiter.limit(
    "10/minute"
)  # only generate lsm visualizations/computations if rate limit not exceeded
async def get_lsm_image(request: Request, token: str, team_id: str, t: str):
    slack_data = get_slack_data(app, token, team_id)
    lsm_image = slack_data.create_lsm_vis()  # updates lsm_image property based on data

    # Return the image as a response
    return Response(content=lsm_image.read(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
