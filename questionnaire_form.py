from slack_sdk.models.views import View
from slack_sdk.models.blocks import (
    InputBlock,
    PlainTextInputElement,
    StaticSelectElement,
    Option,
    SectionBlock,
)


def get_questionnaire():
    return View(
        type="modal",
        callback_id="questionnaire_form",
        title={"type": "plain_text", "text": "Questionnaire"},
        submit={"type": "plain_text", "text": "Submit"},
        blocks=[
            SectionBlock(
                text={
                    "type": "mrkdwn",
                    "text": "Please fill out the following form to submit your analysis results.",
                }
            ),
            InputBlock(
                block_id="industry",
                label={
                    "type": "plain_text",
                    "text": "What industry does your team work in?",
                },
                element=PlainTextInputElement(action_id="industry"),
            ),
            InputBlock(
                block_id="work_type",
                label={
                    "type": "plain_text",
                    "text": "Which of the following options best describes your work setting?",
                },
                element=StaticSelectElement(
                    action_id="industry_select",
                    options=[
                        Option(
                            text={"type": "plain_text", "text": "In-Person"},
                            value="in_person",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "Hybrid"},
                            value="hybrid",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "Remote"},
                            value="remote",
                        ),
                    ],
                ),
            ),
        ],
    )
