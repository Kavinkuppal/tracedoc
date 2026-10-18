"""A suggested fix never tells an agent to act in the real world without the user's say-so.

Booking, buying, paying, sending a message outside the chat, cancelling, or deleting data has
effects the user can't undo. A fix that tells an agent to do one of these must also make it
ask the user to confirm first. One that says to skip asking ("book it without asking") is never
acceptable, and neither is asking or telling the user afterwards ("book the hotel, then confirm
the booking to the user"). `unconfirmed_action` is the rule every fix text is checked against:
rule templates when a finding is made (and in the tests, every template in the catalog), the AI
judge's what_to_change, which is sent back once and then replaced by a safe fix, and the
shortened fix on page 1 of summary.pdf.

An action is the verb in any form ("book", "books", "booking", "proceed with booking", "make
sure the flight is booked"), a noun with a verb that carries it out ("complete the
reservation", "issue refunds"), or a tool named for it being called ("call book_flight",
"search_hotels and book_hotel for New York"). Actions on the agent's own state don't count
("send the model less", "remove old messages from its prompt"), and neither does a tool being
changed rather than called ("make book_flight return an error") or an action the fix forbids
("never book without asking the user").

An action is confirmed when the fix asks the user for confirmation before it ("ask the user to
confirm, then book"), makes it wait for the user's confirmation later in the same sentence
("book only after the user confirms"), or says anywhere that the asking comes before that kind
of action ("before booking, ask the user to confirm"; asking before booking the hotel doesn't
cover booking the flight).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_I = re.IGNORECASE


def _any(*words: str) -> str:
    return "|".join(words)


# ---- what counts as an action ---------------------------------------------------------------

_FREE = _any(
    "book", "books", "booking", "rebook", "rebooks", "rebooking",
    "reserve", "reserves", "reserving",
    "buy", "buys", "buying",
    "purchase", "purchases", "purchasing",
    "pay", "pays", "paying",
    "charge", "charges", "charging",
    "refund", "refunds", "refunding",
)  # fmt: skip
"""Verbs that are an action on their own, in every form but the past ("booked" is below)."""
_NOUN_BEFORE = (
    r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)(?<!\bany\s)(?<!\ball\s)(?<!\beach\s)(?<!\bevery\s)"
    r"(?<!\bno\s)(?<!\bits\s)(?<!\bour\s)(?<!\bhis\s)(?<!\bher\s)(?<!\bthis\s)"
    r"(?<!\btheir\s)(?<!\byour\s)(?<!\bthese\s)(?<!\bthose\s)(?<!\bpast\s)(?<!\bin\s)(?<!\bof\s)"
    r"(?<!'s\s)(?<!\u2019s\s)(?<!s'\s)"
)
"""Not after a determiner or possessive: "the booking", "their purchases" and "in charge" name
a thing, not an action ("complete the booking" is caught as a noun action instead)."""
_MODIFIES = (
    r"(?!\s+(?:tools?|agents?|assistants?|details?|reference|references|numbers?|ids?|status"
    r"|info|information|confirmations?|pages?|steps?|flows?|data|records?|history|summary"
    r"|requests?|api|functions?|nodes?|systems?|forms?|links?|codes?|logic|lists?|options?"
    r"|sites?|platforms?|engines?|services?|process|workflows?|windows?|dates?|fees?|policy"
    r"|policies|customers?|users?|members?|plans?|tiers?|attention|errors?|results?|output)\b)"
)
"""Not before a noun it describes: "booking details", "paying customers", "pay attention"."""
_GAP = (
    r"(?:\s+(?!(?:with|in|from|of|for|to|on|at|by|about|into|than|that|which|who|when|where"
    r"|and|or|but|then|so)\b)[\w'\u2019$.-]+){0,4}?\s+"
)
"""Up to four words between a verb and what it acts on ("delete the user's old files"), none a
preposition or conjunction ("read the file with the orders" doesn't file orders)."""
_DETS = (
    r"(?:\s+(?:the|a|an|this|that|these|those|its|their|his|her|our|your|any|all|each|every"
    r"|pending|new|user's|customer's|requested|selected|chosen|matching|cheapest|first|same"
    r"|whole|full|remaining|final))*\s+"
)
"""Only articles and a few adjectives between a verb and its noun ("issue the pending refund")."""

_SEND = _any(
    "send", "sends", "sending", "email", "emails", "emailing", "e-mail", "e-mails",
    "e-mailing", "text", "post", "posts", "posting", "publish", "publishes", "publishing",
    "share", "shares", "sharing", "forward", "forwards", "forwarding",
)  # fmt: skip
_SEND_WHAT = (
    r"e-?mails?|sms|text\s+messages?|invites?|invitations?|letters?|newsletters?|tweets?|"
    r"payments?|money|funds|invoices?|offers?|quotes?|receipts?"
)
_DELETE = _any(
    "delete", "deletes", "deleting", "remove", "removes", "removing", "erase", "erases",
    "erasing", "wipe", "wipes", "wiping", "drop", "drops", "dropping", "purge", "purges",
    "purging", "overwrite", "overwrites", "overwriting", "destroy", "destroys", "destroying",
)  # fmt: skip
_DELETE_WHAT = (
    r"files?|records?|database|tables?|repo|repository|branch(?:es)?|accounts?|users?|"
    r"customers?|e-?mails?|events?|calendars?|documents?|folders?|buckets?|rows?|backups?|"
    r"orders?|bookings?|reservations?|contacts?"
)
_CANCEL = _any(
    "cancel", "cancels", "cancelling", "canceling", "void", "voids", "voiding", "terminate",
    "terminates", "terminating", "revoke", "revokes", "revoking",
)  # fmt: skip
_CANCEL_WHAT = (
    r"bookings?|reservations?|orders?|subscriptions?|flights?|tickets?|appointments?|"
    r"meetings?|accounts?|payments?|plans?|policy|policies|contracts?"
)
_MOVE = _any(
    "transfer", "transfers", "transferring", "move", "moves", "moving", "send", "sends",
    "sending", "wire", "wires", "wiring",
)  # fmt: skip
_MOVE_WHAT = r"money|funds|payments?|cash|\$\s?\d"
_SUBMIT = _any(
    "submit", "submits", "submitting", "file", "files", "filing", "sign", "signs", "signing",
    "place", "places", "placing",
)  # fmt: skip
_SUBMIT_WHAT = (
    r"forms?|applications?|claims?|contracts?|agreements?|orders?|tax\s+returns?|bets?|trades?"
)
_CARRY_OUT = _any(
    "complete", "completes", "completing", "finish", "finishes", "finishing", "finalize",
    "finalizes", "finalizing", "finalise", "finalises", "finalising", r"make(?!\s+sure)",
    r"makes(?!\s+sure)", r"making(?!\s+sure)", "process", "processes", "processing", "execute",
    "executes", "executing", "issue", "issues", "issuing", "initiate", "initiates",
    "initiating", "submit", "submits", "submitting", "place", "places", "placing", "secure",
    "secures", "securing", r"go\s+ahead\s+with", r"goes\s+ahead\s+with",
    r"going\s+ahead\s+with", r"proceed\s+with", r"proceeds\s+with", r"proceeding\s+with",
    r"carry\s+out", r"carries\s+out", r"carrying\s+out",
)  # fmt: skip
_ACT_NOUN = (
    r"bookings?|reservations?|purchases?|orders?|payments?|refunds?|cancellations?|deletions?|"
    r"transactions?|trades?|checkout|wire\s+transfers?|money\s+transfers?"
)
_DONE = r"booked|rebooked|reserved|bought|purchased|paid|charged|refunded|wired"
"""Past forms count only where the fix asks for the result: "make sure the flight is booked"
(not "never say the flight is booked")."""
_DONE_TO = {  # past forms that need their object: "make sure the email is sent"
    r"cancell?ed|voided": _CANCEL_WHAT,
    r"deleted|removed|erased|wiped|purged": _DELETE_WHAT,
    r"sent|emailed|e-mailed": _SEND_WHAT,
    r"submitted|filed|signed|placed": _SUBMIT_WHAT,
}
_ENSURE = r"\b(?:make\s+sure|makes\s+sure|making\s+sure|ensure|ensures|ensuring)\b"
_NOT_DONE = r"(?<!\bnot\s)(?<!n't\s)(?<!\bnever\s)"
"""Not "make sure the flight isn't booked twice"."""

ACTION = re.compile(
    rf"{_NOUN_BEFORE}\b(?:{_FREE})\b{_MODIFIES}"
    rf"|\b(?:{_SEND}){_GAP}(?:{_SEND_WHAT})\b"
    rf"|\b(?:{_DELETE}){_GAP}(?:{_DELETE_WHAT})\b"
    rf"|\b(?:{_CANCEL}){_GAP}(?:{_CANCEL_WHAT})\b"
    rf"|\b(?:{_MOVE}){_GAP}(?:{_MOVE_WHAT})"
    rf"|\b(?:{_SUBMIT}){_GAP}(?:{_SUBMIT_WHAT})\b"
    rf"|\b(?:{_CARRY_OUT}){_DETS}(?:{_ACT_NOUN})\b{_MODIFIES}"
    rf"|{_ENSURE}(?:\s+[\w'\u2019$.-]+){{0,5}}?\s+(?:(?:is|are|be|been|gets|get|got)\s+)?"
    rf"{_NOT_DONE}(?:{_DONE})\b"
    rf"|\bget(?:s|ting)?\s+(?:it|them|(?:the|their|this|your)\s+[\w'\u2019]+)\s+(?:{_DONE})\b"
    + "".join(
        rf"|{_ENSURE}(?:\s+[\w'\u2019$.-]+){{0,3}}?\s+(?:{what})\s+(?:is|are|be|gets|get|got)\s+"
        rf"{_NOT_DONE}(?:{done})\b"
        for done, what in _DONE_TO.items()
    )
    + r"|\bcheck(?:s|ing)?\s?out\s+(?:the\s+)?(?:cart|basket|order)\b",
    _I,
)
"""An action on a real-world thing, as a verb, a noun with a verb that carries it out, or a
result the fix asks for."""

_TOOL_STEMS = r"book|rebook|reserve|buy|purchase|pay|charge|refund|send|email|delete|cancel|wire"
TOOL = re.compile(
    rf"\b(?:(?:{_TOOL_STEMS})_\w+|place_orders?(?:_\w+)?)\b"
    rf"|\b(?:(?:{_TOOL_STEMS})[A-Z]\w*|placeOrders?\w*)\b"
)
"""A tool named for an action: book_flight, cancel_order, send_email, refundPayment."""
_TOOL_CHANGED_BEFORE = re.compile(
    r"\b(?:make|makes|let|lets|wrap|wraps|wrapping|fix|fixes|change|changes|update|updates"
    r"|modify|modifies|rename|renames|bind|binds|define|defines|document|documents|test|tests"
    r"|mock|mocks|validate|validates|around|for|of|in|on|by|from)\s+(?:the\s+)?$",
    _I,
)
_TOOL_CHANGED_AFTER = re.compile(
    r"^(?:'s|\u2019s|\s+(?:returns?|returned|returning|raises?|raised|fails?|failed|failing"
    r"|errors?|errored|accepts?|accepted|rejects?|rejected|validates?|validated|checks?"
    r"|checked|should|must|needs?|needed|is|are|was|were|has|had|takes?|took|tool|tools"
    r"|schema|docstring|description|arguments?|args|parameters?|params?|signature|results?"
    r"|outputs?|responses?|succeeds?|succeeded|confirms?|confirmed|bound|binding|definition"
    r"|implementation|code|function|wrapper|times?\s+out|timed\s+out|call\s+(?:fails?|failed"
    r"|succeeds?|succeeded|returns?|returned)|repeats?|repeated|gives|gave|shows|showed|finds"
    r"|found|says|said|keeps|kept|loops|looped|works|worked|responds|responded|produces"
    r"|produced|yields|yielded|comes\s+back|came\s+back)\b)",
    _I,
)
"""A tool being changed, described or bound, not called: "make book_flight return an error",
"after book_flight returns", "when book_flight repeats itself", "give hotel_agent the
book_hotel tool"."""

_NEGATED = re.compile(
    r"\b(?:never|not|don't|dont|doesn't|didn't|won't|cannot|can't|mustn't|shouldn't"
    r"|no\s+longer|instead\s+of|rather\s+than|avoid|avoids|avoiding|stop|stops|refuse\s+to"
    r"|refuses\s+to)\s+(?:(?:ever|directly|automatically|immediately|yet|actually|just|try\s+to"
    r"|attempt\s+to|go\s+ahead\s+and|be\s+allowed\s+to|call|calls|use|uses|invoke|invokes|run"
    r"|runs|trigger|triggers|execute|executes|the)\s+)*$",
    _I,
)
"""The fix forbids the action: "never book", "do not call book_flight", "instead of booking"."""
_LIMITS = re.compile(
    r"\b(?:cap|caps|capping|limit|limits|limiting|restrict|restricts|restricting|throttle"
    r"|throttles|how\s+many\s+times|at\s+most|no\s+more\s+than)\b[^,.;:!?]*$",
    _I,
)
"""The fix limits an action the agent already takes rather than asking for it: "cap how many
times flight_agent can call book_flight"."""

# ---- what counts as asking first ------------------------------------------------------------

_WHO = (
    r"(?:the\s+user|users|the\s+customer|customers|the\s+client|clients|the\s+traveler"
    r"|the\s+guest|them|they)"
)
_OK = r"(?:confirmation|approval|permission|consent|go-ahead|ok|okay|sign-off|yes)"
_YES = (
    r"(?:confirm|confirms|confirmed|approve|approves|approved|agree|agrees|agreed|accept"
    r"|accepts|accepted|okays|okayed|says\s+yes|said\s+yes|say\s+yes|says\s+so|said\s+so"
    r"|gives\s+the\s+go-ahead|gave\s+the\s+go-ahead)"
)
ASKS = re.compile(
    r"\b(?:ask|asks|asking|request|requests|requesting|get|gets|getting|obtain|obtains"
    r"|obtaining|seek|seeks|seeking|wait|waits|waiting|need|needs|require|requires|requiring)"
    rf"\s+(?:for\s+)?(?:the\s+user's\s+|the\s+customer's\s+|the\s+client's\s+|their\s+"
    rf"|explicit\s+|a\s+)?(?:explicit\s+)?{_OK}\b"
    rf"|\b(?:ask|asks|asking)\s+{_WHO}\s+(?:to\s+|for\s+(?:a\s+|their\s+)?)?(?:confirm|confirmation"
    r"|approve|approval|agree|okay|ok|accept|permission|consent|sign\s+off|say\s+yes)\b"
    rf"|\b(?:ask|asks|asking)\s+{_WHO}\s+(?:whether|if)\b"
    rf"|\b(?:ask|asks|asking)(?:\s+{_WHO})?\s+(?:first|before)\b"
    rf"|\b(?:confirm|confirms|confirming|check|checks|checking)\s+with\s+{_WHO}\b"
    rf"|\b(?:wait|waits|waiting)\s+for\s+{_WHO}\s+to\s+(?:confirm|approve|agree|reply|answer"
    r"|say\s+yes)\b"
    r"|\b(?:wait|waits|waiting)\s+for\s+(?:the\s+user's|the\s+customer's|their)\s+(?:reply"
    rf"|answer|response|{_OK})\b"
    rf"|\b(?:wait|waits|waiting)\s+until\s+{_WHO}\s+{_YES}\b"
    rf"|\b{_WHO}\s+(?:must|has\s+to|have\s+to|needs\s+to|need\s+to|should)\s+(?:first\s+)?"
    r"(?:confirm|approve|agree)\b",
    _I,
)
"""Asking the user for confirmation: "ask the user to confirm", "wait for their approval"."""
CONFIRMED = re.compile(
    rf"\b(?:after|once|when|if|until)\s+{_WHO}\s+(?:(?:has|have|explicitly|first)\s+)*{_YES}\b"
    rf"|\b(?:after|once|upon|following)\s+(?:asking|confirming\s+with|checking\s+with)\s+{_WHO}\b"
    r"|\b(?:after|once|with|upon|following)\s+(?:getting\s+|receiving\s+|obtaining\s+)?"
    rf"(?:the\s+user's\s+|the\s+customer's\s+|the\s+client's\s+|their\s+|explicit\s+)"
    rf"(?:explicit\s+)?{_OK}\b"
    r"|\b(?:after|once|when)\s+(?:it\s+is\s+|it's\s+|it\s+has\s+been\s+)?(?:confirmed|approved)"
    rf"\s+by\s+{_WHO}\b"
    rf"|\b(?:after|once|when|if)\s+(?:you|it|the\s+agent)\s+(?:have\s+|has\s+)?(?:asked|ask"
    rf"|asks)\s+{_WHO}\b",
    _I,
)
"""Waiting for the user's confirmation, after the action in the same sentence: "book it only
after the user confirms", "with the user's approval"."""
_BEFORE = re.compile(
    r"\bbefore\s+(?:(?:it|you|they|he|she|the\s+agent|\w+_\w+|calling|using|invoking|running)"
    r"\s+){0,2}$",
    _I,
)
SKIPS_ASKING = re.compile(
    r"\bwithout\s+(?:first\s+)?(?:asking|confirm\w*|approval|permission|consent|checking"
    r"|waiting|the\s+user's\s+(?:confirmation|approval|consent|permission))\b"
    r"|\bwithout\s+(?:the\s+user|the\s+customer|them)\s+(?:confirm\w*|approv\w*|agree\w*)"
    r"|\b(?:don't|do\s+not|never|no\s+need\s+to|stop)\s+(?:ask|asking|confirm\w*|wait\w*"
    r"|check\w*\s+with)\b"
    r"|\bskip(?:s|ping)?\s+(?:the\s+)?(?:confirm\w*|approval|asking)\b"
    r"|\b(?:instead\s+of|rather\s+than)\s+(?:asking|confirm\w*|waiting|checking)\b"
    r"|\b(?:ask|asks|confirm\w*|check\w*)\b(?:\s+[\w']+){0,3}?\s+(?:later|afterwards?)\b",
    _I,
)
"""Not asking: "without asking", "don't ask for confirmation", "ask the user later"."""
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)]*(?=\s|$)")
_CLAUSE_END = re.compile(r"[.;:!?]")


@dataclass(frozen=True)
class _Mention:
    start: int
    end: int
    text: str


def _mentions(text: str) -> list[_Mention]:
    """Every action the text names, in order, overlaps merged."""
    found = [_Mention(m.start(), m.end(), m.group(0)) for m in ACTION.finditer(text)]
    for m in TOOL.finditer(text):
        if _TOOL_CHANGED_BEFORE.search(text[max(0, m.start() - 40) : m.start()]):
            continue
        if _TOOL_CHANGED_AFTER.match(text[m.end() :]):
            continue
        found.append(_Mention(m.start(), m.end(), m.group(0)))
    found.sort(key=lambda m: (m.start, -m.end))
    merged: list[_Mention] = []
    for m in found:
        if merged and m.start < merged[-1].end:
            continue
        merged.append(m)
    return merged


def _negated(text: str, m: _Mention) -> bool:
    """Forbidden or only limited, so not asked for."""
    before = text[max(0, m.start - 80) : m.start]
    return bool(_NEGATED.search(before) or _LIMITS.search(before))


def _sentence(text: str, pos: int) -> tuple[int, int]:
    """The start and end of the sentence around `pos`."""
    start = 0
    for end in _SENTENCE_END.finditer(text):
        if end.end() > pos:
            return start, end.end()
        start = end.end()
    return start, len(text)


_KINDS = (
    ("cancel", re.compile(r"cancel|void|terminat|revok", _I)),
    ("refund", re.compile(r"refund", _I)),
    ("delete", re.compile(r"delet|remov|eras|wip|drop|purg|overwrit|destroy", _I)),
    ("book", re.compile(r"book|reserv", _I)),
    ("send", re.compile(r"send|sent|e-?mail|text|post|publish|shar|forward", _I)),
    ("pay", re.compile(r"pay|paid|charg|invoice|wire|wiring|money|funds|cash|\$", _I)),
    ("buy", re.compile(r"buy|bought|purchas|order|check\s?out", _I)),
)
_OBJECTS = re.compile(
    r"\b(flight|hotel|room|ticket|seat|car|table|order|item|invoice|payment|refund"
    r"|subscription|account|appointment|meeting|email|file|record|reservation|booking|trip"
    r"|cart|product|package)s?\b",
    _I,
)
"""What an action is done to, so that asking before booking a hotel doesn't cover a flight."""
_GENERIC = {"booking", "reservation"}
"""Say nothing about what is booked."""


def _kind(m: _Mention) -> str:
    return next((kind for kind, stem in _KINDS if stem.search(m.text)), m.text.lower())


def _object(text: str, m: _Mention) -> str | None:
    """The thing acted on: a tool's name says it (book_hotel), else the first such word in or
    just after the mention ("cancel the order", "book the cheapest flight")."""
    if TOOL.fullmatch(m.text):
        parts = re.split(r"_|(?<=[a-z])(?=[A-Z])", m.text, maxsplit=1)
        where = parts[1].replace("_", " ") if len(parts) == 2 else ""
    else:
        after = re.match(r"(?:\s+[\w'\u2019$.-]+){0,5}", text[m.end :])
        where = (m.text.split(None, 1)[1:] or [""])[0] + (after.group(0) if after else "")
    found = _OBJECTS.search(where)
    thing = found.group(1).lower() if found else None
    return None if thing in _GENERIC else thing


def _waits_before(text: str, mentions: list[_Mention]) -> list[tuple[str, str | None]]:
    """Actions a sentence makes wait for the user ("ask the user to confirm the flight before
    calling book_flight"), as (kind, object): they are confirmed wherever the fix names them."""
    waits = []
    for m in mentions:
        start, end = _sentence(text, m.start)
        if _BEFORE.search(text[start : m.start]) and ASKS.search(text[start:end]):
            waits.append((_kind(m), _object(text, m)))
    return waits


def _confirmed(text: str, m: _Mention, waits: list[tuple[str, str | None]]) -> bool:
    if ASKS.search(text[: m.start]):
        return True  # asked first: "ask the user to confirm, then book"
    _, end = _sentence(text, m.start)
    if CONFIRMED.search(text[m.end : end]):
        return True  # "book it only after the user confirms"
    # "before booking, ask the user to confirm", in this sentence or another one
    kind, thing = _kind(m), _object(text, m)
    return any(k == kind and (thing is None or o is None or o == thing) for k, o in waits)


def _forbids(text: str, skip: re.Match[str], mentions: list[_Mention]) -> bool:
    """A "without asking" that is part of forbidding the action: "never book without asking"."""
    clause = max((c.end() for c in _CLAUSE_END.finditer(text, 0, skip.start())), default=0)
    return any(clause <= m.start < skip.start() and _negated(text, m) for m in mentions)


def unconfirmed_action(fix: str) -> str | None:
    """The real-world action a fix tells an agent to take without the user's confirmation
    ("book", "book_hotel", "complete the reservation"), or None if the fix is fine."""
    text = " ".join(fix.split())
    mentions = _mentions(text)
    actions = [m for m in mentions if not _negated(text, m)]
    if not actions:
        return None
    if any(not _forbids(text, s, mentions) for s in SKIPS_ASKING.finditer(text)):
        return actions[0].text
    waits = _waits_before(text, mentions)
    return next((m.text for m in actions if not _confirmed(text, m, waits)), None)


def asks_for_confirmation(sentence: str) -> bool:
    """The sentence asks the user to confirm, or waits for their confirmation."""
    return bool(ASKS.search(sentence) or CONFIRMED.search(sentence))


def safe_fix(fix: str, agent: str) -> str:
    """`fix` if it passes the rule, else the catalog's fix that makes `agent` ask first."""
    from ..catalog import label

    if unconfirmed_action(fix) is None:
        return fix
    return label("fix_confirm_first", agent=agent or label("unknown_agent"))
