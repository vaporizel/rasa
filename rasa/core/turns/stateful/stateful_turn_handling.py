from __future__ import annotations
from dataclasses import dataclass
from typing import List

from rasa.core.turns.turn import Actor
from rasa.core.turns.stateful.stateful_turn import StatefulTurn
from rasa.core.turns.to_dataset.dataset import TurnSequenceModifier
from rasa.shared.core.constants import (
    ACTION_UNLIKELY_INTENT_NAME,
    PREVIOUS_ACTION,
    USER,
)
from rasa.shared.nlu.constants import ACTION_NAME, ENTITIES, INTENT, TEXT


@dataclass
class RemoveTurnsWithPrevActionUnlikelyIntent(TurnSequenceModifier[StatefulTurn]):
    """Remove turns where the previous action substate is an action unlikely intent."""

    switched_on: bool = True

    def __call__(
        self, turns: List[StatefulTurn], training: bool,
    ) -> List[StatefulTurn]:
        if not self.switched_on:
            return turns
        return [
            turn
            for turn in turns
            if turn.state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
            != ACTION_UNLIKELY_INTENT_NAME
        ]


@dataclass
class DuringPredictionIfLastTurnWasUserTurnKeepEitherTextOrNonText(
    TurnSequenceModifier[StatefulTurn]
):
    """Removes (intent and entities) or text from the last turn if it was a user turn.

    Only does so during inference. During training, nothing is removed.

    TODO: why was this always applied - shouldn't it only be done if intent/entities
    are targets? (in that case, the new label extractors would take care of this)
    """

    keep_text: bool = True

    def __call__(
        self, turns: List[StatefulTurn], training: bool,
    ) -> List[StatefulTurn]:
        if not training:
            if turns and turns[-1].actor == Actor.USER:
                last_turn = turns[-1]
                remove = [INTENT, ENTITIES] if self.keep_text else [TEXT]
                for key in remove:
                    last_turn.state.get(USER, {}).pop(key, None)
        return turns


@dataclass
class RemoveUserTextIfIntentFromEveryTurn(TurnSequenceModifier[StatefulTurn]):
    """Removes the text if there is an intent in the user substate of every turn.

    This is always applied - during training as well as during inference.
    """

    def __call__(self, turns: List[StatefulTurn], training: bool) -> List[StatefulTurn]:

        for turn in turns:
            user_substate = turn.state.get(USER, {})
            if TEXT in user_substate and INTENT in user_substate:
                del user_substate[TEXT]

        return turns
