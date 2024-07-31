import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
import time


def get_upload_url(bot_token, filename, file_length):
    client = WebClient(token=bot_token)
    try:
        response = client.files_getUploadURLExternal(
            filename=filename,
            length=file_length,
        )
        if response["ok"]:
            return response["upload_url"], response["file_id"]
    except SlackApiError as e:
        raise Exception(f"Error getting upload URL: {e.response['error']}")


def upload_file_to_url(upload_url, file_path, filename):
    with open(file_path, "rb") as file_content:
        response = requests.post(
            upload_url, files={"file": (filename, file_content, "application/pdf")}
        )
        if response.status_code != 200:
            raise Exception(
                f"Error uploading file: {response.status_code}, {response.text}"
            )


def create_consent_messages(
    user_name, installer, bot_token, channel_id, user_id, client
):
    consent_info = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Hi <@{user_name}>! <@{installer}> has installed the SharUn app – this is an app designed by researchers at the University of Toronto to measure Shared Understanding from your team’s Slack communication. This application only uses public channel messages (not private channels or DMs) and reports anonymized, aggregated values and visualizations, with feedback for how your team can improve. We are seeking consent to use the messages you’ve sent in public channels for these calculations, where only the final calculated values will be stored by researchers (we will never have access to the content of your messages). You can revoke your consent at any time. No one on your team will know whether you’ve consented or not. Please read the following consent form before deciding whether to consent.",
            },
        },
    ]
    response = client.conversations_open(token=bot_token, users=user_id)
    channel_id = response["channel"]["id"]
    client.chat_postMessage(
        text="Slack Data Consent Form",
        token=bot_token,
        channel=channel_id,
        blocks=consent_info,
    )

    # upload consent form
    file_path = "./forms/consent_agreement.pdf"
    filename = "consent_agreement.pdf"
    upload_url, file_id = get_upload_url(
        bot_token, filename, os.path.getsize(file_path)
    )
    upload_file_to_url(upload_url, file_path, filename)
    client.files_completeUploadExternal(
        token=bot_token,
        files=[{"id": file_id, "title": "consent_agreement"}],
        channel_id=channel_id,
    )

    time.sleep(2)

    # allow user to click consent-yes or no
    consent_button = [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Provide Consent"},
                    "style": "primary",
                    "action_id": "consent_yes",
                },
            ],
        },
    ]
    client.chat_postMessage(
        text="Slack Data Consent Form",
        token=bot_token,
        channel=channel_id,
        blocks=consent_button,
    )


def get_latest_message_ts(bot_token, client, channel_id):
    try:
        last_msg = client.conversations_history(
            token=bot_token,
            channel=channel_id,
            limit=1,  # Adjust the limit based on how many recent messages you want to fetch
        )["messages"][0]
        print(last_msg)
        return last_msg["ts"]
    except Exception as e:
        print(f"Error retrieving message history: {e}")
        return None


def post_consent_confirmation(bot_token, client, channel_id, user_name):
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Thank you for providing consent <@{user_name}>! If you wish to revoke your consent, please click the button below.",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Revoke Consent"},
                    "style": "danger",
                    "action_id": "consent_no",
                },
            ],
        },
    ]
    latest_ts = get_latest_message_ts(bot_token, client, channel_id)
    try:
        client.chat_update(
            token=bot_token,
            channel=channel_id,
            ts=latest_ts,
            blocks=blocks,
            text="Consent Confirmation",
        )
    except Exception as e:
        print(f"Error updating message: {e}")


def post_dissent_confirmation(bot_token, client, channel_id, user_name):
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"You have revoked your consent <@{user_name}>. If you wish to provide consent for future analysis, please click the button below.",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Provide Consent"},
                    "style": "primary",
                    "action_id": "consent_yes",
                },
            ],
        },
    ]
    latest_ts = get_latest_message_ts(bot_token, client, channel_id)
    try:
        client.chat_update(
            token=bot_token,
            channel=channel_id,
            ts=latest_ts,
            blocks=blocks,
            text="Dissent Confirmation",
        )
    except Exception as e:
        print(f"Error updating message: {e}")
