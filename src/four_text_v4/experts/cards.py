"""Base specialist instructions and separate native-label classification cards."""
from .base_prompt import VERSION, ROLE_INSTRUCTIONS, role_instructions

CARDS = ROLE_INSTRUCTIONS

PROMPT_VERSION = VERSION
FIXED_EXAMPLE_MAILS = ()

TASK_GUIDANCE = {
    'spam': """Task meaning: distinguish legitimate correspondence from spam under the
supplied dataset definitions. Spam includes unwanted or unsolicited bulk promotion,
scams and other unwanted messages; it is not limited to phishing. Legitimate personal,
work, transactional and requested correspondence can contain offers, links or requests.
Interpret purpose, recipient relationship and requested behavior together. Commercial
wording alone does not establish spam. If consent or prior contact is unavailable,
keep that uncertainty visible instead of assuming either consent or its absence.
""",
    'phishing': """Task meaning: distinguish legitimate email from deceptive attempts to
obtain information, money or access, or induce harmful actions, under the supplied
dataset definitions. Examine how the claimed identity and context support the requested
action and who would receive its result. Links, deadlines, payments and advertising
alone do not establish phishing. Missing information alone proves neither safety nor
deception. A harmful request can use polite or ordinary language.
""",
}

SCORING_CARDS = {
    'action': """ACTION PERSPECTIVE: Assess the target class through who asks whom to do
what, the affected object, destination and conditions. Distinguish current requests
from historical, quoted or negated actions. Judge the request in its visible context;
the presence of an action, URL or attachment alone does not determine the class.
""",
    'context': """CONTEXT PERSPECTIVE: Assess the target class through the claimed sender
identity and relationship, available background, and whether these support the request.
Use concrete inconsistencies when present. Distinguish sender claims from verified
facts; missing metadata alone does not determine the class.
""",
    'persuasion': """PERSUASION PERSPECTIVE: Assess the target class through how influence
is linked to requested behavior, consequences, authority and restrictions. Ordinary
coordination can include deadlines and incentives. Judge the relationship and context,
not the mere presence or intensity of persuasive language.
""",
    'purpose': """PURPOSE PERSPECTIVE: Assess the target class through the stated purpose,
intended outcome, direct consequences, affected objects and recipients of the result.
Keep explicit purpose separate from inference; do not assume a hidden harmful outcome
without support from the visible email.
""",
}


def role_card(role, task_type=None):
    return ((TASK_GUIDANCE[task_type] + "\n") if task_type else "") + role_instructions(role) + "\n"
