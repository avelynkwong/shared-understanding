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
    try:
        client.chat_postMessage(
            text="Consent Confirmation",
            token=bot_token,
            channel=channel_id,
            blocks=blocks,
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
    try:
        client.chat_postMessage(
            text="Dissent Confirmation",
            token=bot_token,
            channel=channel_id,
            blocks=blocks,
        )
    except Exception as e:
        print(f"Error updating message: {e}")
