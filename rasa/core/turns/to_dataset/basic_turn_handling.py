from dataclasses import dataclass
from typing import Generic, Optional, List


from rasa.core.turns.to_dataset.dataset import (
    TurnSequenceValidation,
    TurnSequenceModifier,
    TurnType,
)
from rasa.core.turns.turn import Actor


class EndsWithBotTurn(TurnSequenceValidation[TurnType]):
    def __call__(self, turns: List[TurnType]) -> bool:
        return turns[-1].actor == Actor.BOT


@dataclass
class HasMinLength(TurnSequenceValidation[TurnType]):
    min_length: int

    def __call__(self, turns: List[TurnType]) -> bool:
        return len(turns) >= self.min_length


@dataclass
class KeepMaxHistory(TurnSequenceModifier[TurnType], Generic[TurnType]):

    max_history: Optional[int] = None

    def __call__(self, turns: List[TurnType], training: bool,) -> List[TurnType]:
        if self.max_history:
            turns = turns[-self.max_history :]
        return turns
