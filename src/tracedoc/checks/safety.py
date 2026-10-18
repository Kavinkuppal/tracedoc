"""A suggested fix never tells an agent to act in the real world without the user's say-so.

Booking, buying, paying, sending a message outside the chat, cancelling, or deleting data has
effects the user can't undo. A fix that tells an agent to do one of these must also make it
ask the user to confirm first; one that says to skip asking ("book it without asking") is
never acceptable. `unconfirmed_action` is the rule every fix text is checked against: rule
templates when a finding is made (and in the tests, every template in the catalog), and the AI
judge's what_to_change, which is sent back once and then replaced by a safe fix.

Only actions on real-world things count: "send an email" does, "send the model less" or
"remove old messages from its prompt" (the agent's own state) don't.
"""

from __future__ import annotations

import re

_ALWAYS = r"book|reserve|buy|purchase|pay(?!\s+attention)|(?<!in\s)charge|refund|wire"
_SEND = r"send|sends|email|e-mail|text|post|publish|share|forward"
_SEND_WHAT = (
    r"e-?mails?|sms|text messages?|invites?|invitations?|letters?|newsletters?|tweets?|"
    r"payments?|money|funds|invoices?|offers?|quotes?"
)
_DELETE = r"delete|deletes|remove|removes|erase|wipe|drop|purge|overwrite"
_DELETE_WHAT = (
    r"files?|records?|database|tables?|repo|repository|branch(?:es)?|accounts?|users?|"
    r"customers?|e-?mails?|events?|calendars?|documents?|folders?|buckets?|rows?|backups?|"
    r"orders?|bookings?|reservations?|contacts?"
)
_CANCEL = r"cancel|cancels|void|terminate"
_CANCEL_WHAT = (
    r"bookings?|reservations?|orders?|subscriptions?|flights?|tickets?|appointments?|"
    r"meetings?|accounts?|payments?|plans?|policy|policies|contracts?"
)
_MOVE = r"transfer|transfers|move|moves|send"
_MOVE_WHAT = r"money|funds|payments?|\$\s?\d"
_SUBMIT = r"submit|submits|file|files|sign|signs|place|places"
_SUBMIT_WHAT = r"forms?|applications?|claims?|contracts?|orders?|tax returns?"
_GAP = r"(?:\s+[\w'\u2019$.-]+){0,4}?\s+"
"""Up to four words between a verb and what it acts on ("delete the user's old files")."""

ACTION = re.compile(
    rf"\b(?:(?:{_ALWAYS})\b"
    rf"|(?:{_SEND}){_GAP}(?:{_SEND_WHAT})\b"
    rf"|(?:{_DELETE}){_GAP}(?:{_DELETE_WHAT})\b"
    rf"|(?:{_CANCEL}){_GAP}(?:{_CANCEL_WHAT})\b"
    rf"|(?:{_MOVE}){_GAP}(?:{_MOVE_WHAT})"
    rf"|(?:{_SUBMIT}){_GAP}(?:{_SUBMIT_WHAT})\b"
    r"|check\s?out\b|place\s+(?:the|an?)\s+order)",
    re.IGNORECASE,
)
_WHO = r"(?:the\s+user|users|the\s+customer|the\s+client|them|they)"
ASKS_FIRST = re.compile(
    rf"\b(?:ask|asks|asking)\s+{_WHO}\b"
    rf"|\b(?:ask|asks|asking|request|requests|get|gets|getting|obtain|wait|waits|waiting)\s+"
    r"(?:for\s+)?(?:the\s+user's\s+|the\s+customer's\s+|their\s+|explicit\s+)?"
    r"(?:confirmation|approval|permission|consent|go-ahead|ok|okay)\b"
    rf"|\b(?:after|once|until|unless|only\s+if|only\s+when)\s+{_WHO}\s+"
    r"(?:confirm|confirms|confirmed|approve|approves|approved|agree|agrees|agreed|say|says|"
    r"said|accept|accepts|accepted|reply|replies)\b"
    r"|\b(?:user|customer)(?:'s|\u2019s)?\s+(?:confirmation|approval|consent|go-ahead|permission)\b"
    rf"|\bconfirm\s+with\s+{_WHO}\b|\b{_WHO}\s+(?:to\s+)?confirm\b"
    rf"|\bcheck(?:s|ing)?\s+with\s+{_WHO}\b",
    re.IGNORECASE,
)
SKIPS_ASKING = re.compile(
    r"\bwithout\s+(?:first\s+)?(?:asking|confirm\w*|approval|permission|consent|checking|"
    r"waiting)\b"
    r"|\b(?:don't|do\s+not|never|no\s+need\s+to|stop)\s+(?:ask|asking|confirm\w*|wait\w*|"
    r"check\w*\s+with)\b"
    r"|\bskip(?:s|ping)?\s+(?:the\s+)?(?:confirm\w*|approval|asking)\b"
    r"|\b(?:instead\s+of|rather\s+than)\s+(?:asking|confirm\w*|waiting)\b"
    r"|\b(?:ask|asks|confirm\w*|check\w*)\b(?:\s+[\w']+){0,3}?\s+(?:later|afterwards?)\b",
    re.IGNORECASE,
)


def unconfirmed_action(fix: str) -> str | None:
    """The real-world action a fix tells an agent to take without the user's confirmation
    ("book", "send an email", ...), or None if the fix is fine."""
    text = " ".join(fix.split())
    action = ACTION.search(text)
    if action is None:
        return None
    if SKIPS_ASKING.search(text) or not ASKS_FIRST.search(text):
        return action.group(0)
    return None


def safe_fix(fix: str, agent: str) -> str:
    """`fix` if it passes the rule, else the catalog's fix that makes `agent` ask first."""
    from ..catalog import label

    if unconfirmed_action(fix) is None:
        return fix
    return label("fix_confirm_first", agent=agent or label("unknown_agent"))
