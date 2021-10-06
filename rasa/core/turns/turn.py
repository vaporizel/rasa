from __future__ import annotations
from abc import abstractmethod
from enum import Enum
import logging
from typing import (
    Any,
    Optional,
    Text,
    List,
)

import rasa.shared.core.constants
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class Actor(str, Enum):
    """Bot or user."""

    USER = rasa.shared.core.constants.USER
    BOT = "bot"


class Turn:
    """A sequence of events that can be attributed to either the bot or the user.

    Args:
        actor: describes whose turn it is (user or bot)
        events: a sequence of events belonging to the `actor`
    """

    def __init__(self, actor: Actor, events: List[Event]) -> None:
        self.actor = actor
        self.events = events

    def __repr__(self) -> Text:
        return f"{self.__class__.__name__}({self.actor}):{self.events}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Turn):
            return NotImplemented
        return self.actor == other.actor and self.events == other.events

    def __neq__(self, other: Any) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(f"{self.actor}{self.events}")

    @staticmethod
    def get_index_of_last_user_turn(turns: List[Turn]) -> Optional[int]:
        """Returns the index of the last user turn, or None if there is no such turn.

        Returns:
            the index of the last user turn, or None if there is no such turn
        """
        return next(
            (
                idx
                for idx in range(len(turns) - 1, -1, -1)
                if turns[idx].actor == Actor.USER
            ),
            None,
        )


class DefinedTurn(Turn):
    """A turn that has a defined meaning and can only be created from a tracker."""

    _credentials = object()

    def __init__(self, credentials: object, actor: Actor, events: List[Event]):
        """Private constructor for stateful turns. Use `parse` to generate these turns.

        Args:
            credentials: special object that allows you to use this method
            actor: describes whose turn it is (user or bot)
            events: a sequence of events belonging to the `actor`
        """
        if credentials != DefinedTurn._credentials:
            raise RuntimeError(
                f"{self.__class__.__name__} should only be created via `parse`."
            )
        super().__init__(actor=actor, events=events)

    @classmethod
    @abstractmethod
    def parse(
        cls, tracker: DialogueStateTracker, domain: Domain, **kwargs: Any
    ) -> List[Turn]:
        pass
