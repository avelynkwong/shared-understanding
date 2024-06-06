def generate_consent_form():
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "SharUn is requesting access to your public channel messages. Please indicate whether you consent to having your messages analyzed:",
            },
        },
        {
            "type": "section",
            "block_id": "poll",
            "text": {
                "type": "mrkdwn",
                "text": "Do you consent to having SharUn access your public messages?",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Yes"},
                    "style": "primary",
                    "action_id": "consent_yes",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "No"},
                    "style": "danger",
                    "action_id": "consent_no",
                },
            ],
        },
    ]


def update_message(bot_token, client, channel_id, user_name):
    text = f"<@{user_name}> provided consent!"
    try:
        client.chat_postMessage(token=bot_token, channel=channel_id, text=text)
    except Exception as e:
        print(f"Error updating message: {e}")
