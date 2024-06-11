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
                # {
                #     "type": "button",
                #     "text": {"type": "plain_text", "text": "No"},
                #     "style": "danger",
                #     "action_id": "consent_no",
                # },
            ],
        },
    ]


def update_message(bot_token, client, channel_id, user_name):
    text = f"Thank you for providing consent <@{user_name}>! No further action is required."
    try:
        client.chat_postMessage(token=bot_token, channel=channel_id, text=text)
    except Exception as e:
        print(f"Error updating message: {e}")
