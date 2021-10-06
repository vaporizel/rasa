from dataclasses import dataclass
from typing import Generic, Optional, List


from rasa.core.turns.to_dataset.dataset import (
    TurnSequenceValidation,
    TurnSequenceModifier,
    TurnType,
)
from rasa.core.turns.turn import Actor
from rasa.shared.core.events import ActionExecuted


class EndsWithBotTurn(TurnSequenceValidation[TurnType]):
    def __call__(self, turns: List[TurnType]) -> bool:
        return turns[-1].actor == Actor.BOT


@dataclass
class HasMinLength(TurnSequenceValidation[TurnType]):
    min_length: int

    def __call__(self, turns: List[TurnType]) -> bool:
        return len(turns) >= self.min_length


@dataclass
class EndsWithPredictableActionExecuted(TurnSequenceValidation[TurnType]):
    def __call__(self, turns: List[TurnType]) -> bool:
        if not turns[-1].events:
            return False
        first_event_of_last_turn = turns[-1].events[0]
        return (
            isinstance(first_event_of_last_turn, ActionExecuted)
            and not first_event_of_last_turn.unpredictable
        )


@dataclass
class KeepMaxHistory(TurnSequenceModifier[TurnType], Generic[TurnType]):

    max_history: Optional[int] = None
    offset_for_training: int = 0

    # FIXME: use another parameter instead of just hard coding the +1?
    def apply(self, turns: List[TurnType], training: bool) -> List[TurnType]:
        """Keeps the last `max_history`(+1) turns during inference (training)."""
        if self.max_history is not None:
            keep = (
                (self.max_history + self.offset_for_training)
                if training
                else self.max_history
            )
            turns = turns[-keep:]
        return turns

    def modify(self, turns: List[TurnType], inplace: bool,) -> List[TurnType]:
        if self.max_history:
            turns = turns[-self.max_history :]
        return turns
