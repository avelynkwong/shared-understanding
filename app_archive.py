import os
from slack_bolt import App
from slack_data import SlackData
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt.oauth.oauth_settings import OAuthSettings
from slack_sdk.oauth.installation_store import FileInstallationStore, Installation
from slack_sdk.oauth.state_store import FileOAuthStateStore
from slack_sdk.oauth import AuthorizeUrlGenerator
import html
from slack_sdk.web import WebClient

# env variables
client_id = os.environ["SLACK_CLIENT_ID"]
client_secret = os.environ.get("SLACK_CLIENT_SECRET")
redirect_uri = os.environ.get("SLACK_REDIRECT_URI")
signing_secret = os.environ.get("SLACK_SIGNING_SECRET")

# OAUTH
# Build https://slack.com/oauth/v2/authorize with sufficient query parameters
authorize_url_generator = AuthorizeUrlGenerator(
    client_id=client_id,
    scopes=["channels:history", "channels:read"],
    user_scopes=["channels:history"],
)
installation_store = FileInstallationStore(base_dir="./data/installations")
state_store = FileOAuthStateStore(expiration_seconds=600, base_dir="./data/states")
oauth_settings = OAuthSettings(
    client_id=client_id,
    scopes=["channels:history", "channels:read"],
    user_scopes=["channels:history"],
    redirect_uri=redirect_uri,
    install_page_rendering_enabled=False,
    install_path="/slack/install",
    redirect_uri_path="/slack/oauth_redirect",
    installation_store=installation_store,
    state_store=state_store,
)

# APP INITIALIZATION
app = App(signing_secret=signing_secret, oauth_settings=oauth_settings)
api = FastAPI()
handler = SlackRequestHandler(app)
# slack_data = SlackData(app, bot_token)  # instance of slackdata


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


@app.event("app_home_opened")
def load_homepage(client, event, logger):
    print(event)
    installation = installation_store.find_installation(
        enterprise_id=None, team_id="T7HD0J5GF"
    )
    bot_token = installation.bot_token
    slack_data = SlackData(app, bot_token)
    slack_data.get_invited_conversations()
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
@api.get("/slack/oauth_redirect")
async def oauth_redirect(request: Request):
    # # Extract state and code from the request
    # state = req.query_params.get("state")
    # code = req.query_params.get("code")

    # if code:
    #     if state_store.consume(state):
    #         client = WebClient()
    #         # Complete the installation by calling oauth.v2.access API method
    #         oauth_response = client.oauth_v2_access(
    #             client_id=client_id,
    #             client_secret=client_secret,
    #             redirect_uri=redirect_uri,
    #             code=code,
    #         )
    #         installed_enterprise = oauth_response.get("enterprise") or {}
    #         is_enterprise_install = oauth_response.get("is_enterprise_install")
    #         installed_team = oauth_response.get("team") or {}
    #         installer = oauth_response.get("authed_user") or {}
    #         incoming_webhook = oauth_response.get("incoming_webhook") or {}
    #         bot_token = oauth_response.get("access_token")
    #         # NOTE: oauth.v2.access doesn't include bot_id in response
    #         bot_id = None
    #         enterprise_url = None
    #         if bot_token is not None:
    #             auth_test = client.auth_test(token=bot_token)
    #             bot_id = auth_test["bot_id"]
    #             if is_enterprise_install is True:
    #                 enterprise_url = auth_test.get("url")

    #         installation = Installation(
    #             app_id=oauth_response.get("app_id"),
    #             enterprise_id=installed_enterprise.get("id"),
    #             enterprise_name=installed_enterprise.get("name"),
    #             enterprise_url=enterprise_url,
    #             team_id=installed_team.get("id"),
    #             team_name=installed_team.get("name"),
    #             bot_token=bot_token,
    #             bot_id=bot_id,
    #             bot_user_id=oauth_response.get("bot_user_id"),
    #             bot_scopes=oauth_response.get("scope"),  # comma-separated string
    #             user_id=installer.get("id"),
    #             user_token=installer.get("access_token"),
    #             user_scopes=installer.get("scope"),  # comma-separated string
    #             incoming_webhook_url=incoming_webhook.get("url"),
    #             incoming_webhook_channel=incoming_webhook.get("channel"),
    #             incoming_webhook_channel_id=incoming_webhook.get("channel_id"),
    #             incoming_webhook_configuration_url=incoming_webhook.get(
    #                 "configuration_url"
    #             ),
    #             is_enterprise_install=is_enterprise_install,
    #             token_type=oauth_response.get("token_type"),
    #         )

    #         # Store the installation
    #         installation_store.save(installation)

    #         return "Thanks for installing this app!"

    #     else:
    #         return Response(
    #             content=f"Try the installation again (the state value is already expired)",
    #             status_code=400,
    #         )

    # error = req.query_params.get("error") if "error" in req.query_params else ""
    # return Response(
    #     content=f"Something is wrong with the installation (error: {html.escape(error)})",
    #     status_code=400,
    # )
    return handler.handle(request)


@api.get("/slack/install")
async def install(request: Request):
    # state = state_store.issue()
    # url = authorize_url_generator.generate(state)
    # return HTMLResponse(
    #     content=f'<a href="{html.escape(url)}">'
    #     f'<img alt=""Add to Slack"" height="40" width="139" src="https://platform.slack-edge.com/img/add_to_slack.png" srcset="https://platform.slack-edge.com/img/add_to_slack.png 1x, https://platform.slack-edge.com/img/add_to_slack@2x.png 2x" /></a>'
    # )
    return handler.handle(request)


@api.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)


@api.post("/slack/options")
async def slack_options(request: Request):
    return await handler.handle(request)


@api.post("/slack/interactions")
async def slack_interactions(request: Request):
    return await handler.handle(request)


# Ready? Start your app!
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=80)

    # app.start(port=int(os.environ.get("PORT", 80)))
