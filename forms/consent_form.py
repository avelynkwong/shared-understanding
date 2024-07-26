def generate_consent_form():
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "SharUn is requesting access to your public channel messages. Please click the button below if you consent to having your public channel messages accessed by this application.",
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
