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
                block_id="team_size",
                label={
                    "type": "plain_text",
                    "text": "What size is the team represented by the channels selected?",
                },
                element=StaticSelectElement(
                    action_id="team_size",
                    options=[
                        Option(
                            text={"type": "plain_text", "text": "1-5"},
                            value="1-5",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "6-10"},
                            value="6-10",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "11-15"},
                            value="11-15",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "16-20"},
                            value="16-20",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "21-30"},
                            value="21-30",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "31-50"},
                            value="31-50",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "50+"},
                            value="50+",
                        ),
                    ],
                ),
            ),
            InputBlock(
                block_id="team_duration",
                label={
                    "type": "plain_text",
                    "text": "How long has this team existed?",
                },
                element=StaticSelectElement(
                    action_id="team_duration",
                    options=[
                        Option(
                            text={"type": "plain_text", "text": "< 3 months"},
                            value="< 3 months",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "3-6 months"},
                            value="3-6 months",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "7-12 months"},
                            value="7-12 months",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "1-3 years"},
                            value="1-3 years",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "4-5 years"},
                            value="4-5 years",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "5+ years"},
                            value="5+ years",
                        ),
                    ],
                ),
            ),
            InputBlock(
                block_id="collaboration_type",
                label={
                    "type": "plain_text",
                    "text": "How does this team typically collaborate?",
                },
                element=StaticSelectElement(
                    action_id="collaboration_type",
                    options=[
                        Option(
                            text={"type": "plain_text", "text": "Entirely in person"},
                            value="in-person",
                        ),
                        Option(
                            text={
                                "type": "plain_text",
                                "text": "Mostly in person, but sometimes online",
                            },
                            value="mostly in-person",
                        ),
                        Option(
                            text={
                                "type": "plain_text",
                                "text": "Mostly online, but sometimes in person",
                            },
                            value="mostly online",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "Entirely online"},
                            value="entirely online",
                        ),
                    ],
                ),
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
                block_id="task_type",
                label={
                    "type": "plain_text",
                    "text": "What tasks does this team typically work on?",
                },
                element=StaticSelectElement(
                    action_id="task_type",
                    options=[
                        Option(
                            text={"type": "plain_text", "text": "Creative design"},
                            value="creative design",
                        ),
                        Option(
                            text={
                                "type": "plain_text",
                                "text": "Project Management",
                            },
                            value="project management",
                        ),
                        Option(
                            text={
                                "type": "plain_text",
                                "text": "Administration",
                            },
                            value="administration",
                        ),
                        Option(
                            text={"type": "plain_text", "text": "Analysis"},
                            value="analysis",
                        ),
                        Option(
                            text={
                                "type": "plain_text",
                                "text": "Marketing/communication",
                            },
                            value="marketing/communication",
                        ),
                        Option(
                            text={
                                "type": "plain_text",
                                "text": "Other",
                            },
                            value="other",
                        ),
                    ],
                ),
            ),
            InputBlock(
                block_id="task_type_other",
                label={
                    "type": "plain_text",
                    "text": "If other, please specify:",
                },
                optional=True,
                element=PlainTextInputElement(action_id="task_type_other"),
            ),
        ],
    )
