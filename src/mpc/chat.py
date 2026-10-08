"""A member's conversation with their personal agent: stored in Postgres,
recent turns cached in Redis (written only after the transaction commits)."""

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from mpc.context.repo import on_commit
from mpc.context.visibility import TEAM_ABBR
from mpc.models import Member, Message
from mpc.redis_store import push_turn, recent_turns
from mpc.tokens import estimate_tokens

HISTORY_TURNS = 12


def history(session: Session, r: redis.Redis | None, member_id: str) -> list[dict]:
    """Recent personal-channel turns as chat messages, starting with a user turn."""
    turns = recent_turns(r, member_id, HISTORY_TURNS) if r is not None else []
    if not turns:  # cache miss: fall back to Postgres
        rows = session.scalars(
            select(Message)
            .where(Message.channel == f"personal:{member_id}")
            .order_by(Message.turn.desc())
            .limit(HISTORY_TURNS)
        )
        turns = [{"role": m.role, "content": m.content} for m in reversed(list(rows))]
    msgs = [{"role": t["role"], "content": t["content"]} for t in turns]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def record_turn(
    session: Session,
    r: redis.Redis | None,
    member: Member,
    text: str,
    reply: str,
    reply_tokens: int = 0,
) -> tuple[int, int]:
    """Store one user message and the agent's reply; returns their turns."""
    agent_id = f"agent_{TEAM_ABBR[member.team_id]}_{member.id}"
    channel = f"personal:{member.id}"
    user_msg = Message(member_id=member.id, agent_id=agent_id, channel=channel, role="user",
                       content=text, tokens=estimate_tokens(text))
    session.add(user_msg)
    session.flush()
    bot_msg = Message(member_id=member.id, agent_id=agent_id, channel=channel, role="assistant",
                      content=reply, tokens=reply_tokens or estimate_tokens(reply))
    session.add(bot_msg)
    session.flush()
    if r is not None:
        ut, bt, mid = user_msg.turn, bot_msg.turn, member.id

        def _cache() -> None:
            push_turn(r, mid, "user", text, ut)
            push_turn(r, mid, "assistant", reply, bt)

        on_commit(session, _cache)
    return user_msg.turn, bot_msg.turn
