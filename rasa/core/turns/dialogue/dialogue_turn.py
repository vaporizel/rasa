from __future__ import annotations
from typing import List, Any


from rasa.core.turns.turn import Actor, DefinedTurn
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


class DialogueTurn(DefinedTurn):
    """A dialogue turn.

    A dialogue is a communication between two participants. We assume that this
    communication is turn-based: At any point in time, exactly one of the participants
    has time to think about what to do next and then communicates, while the other
    participant waits.

    In this basic, the two participants are the bot and a user. After each bot turn,
    the bot can decide who is next. After a user turn, the bot is always next.
    Moreover, we can actually observe what our bot is thinking during its turn.

    Args:
        actor: describes whose turn it is (user or bot)
        events: In case of a user turn, this list should just contain a single event
           describing the user's utterance. In case of a bot turn, this sequence of
           events might reveal the inner workings of our bot and should end with a
           communication, i.e. an `ActionExecuted`.
    """

    @classmethod
    def parse(
        cls, tracker: DialogueStateTracker, domain: Domain, **kwargs: Any
    ) -> List[DialogueTurn]:
        turns = []
        bot_turn_events = []
        for current_event in enumerate(tracker.events):
            if isinstance(current_event, UserUttered):
                if bot_turn_events:
                    bot_turn = DialogueTurn(
                        DefinedTurn._credentials,
                        actor=Actor.BOT,
                        events=bot_turn_events,
                    )
                    turns.append(bot_turn)
                    bot_turn_events = []
                user_turn = DialogueTurn(
                    DefinedTurn._credentials, actor=Actor.USER, events=[current_event]
                )
                turns.append(user_turn)
            elif isinstance(current_event, ActionExecuted):
                bot_turn_events.append(current_event)
                bot_turn = DialogueTurn(
                    DefinedTurn._credentials, actor=Actor.BOT, events=bot_turn_events
                )
                turns.append(bot_turn)
                bot_turn_events = []
            else:
                bot_turn_events.append(current_event)
        # Note that we drop all remaining events.
        return turns
