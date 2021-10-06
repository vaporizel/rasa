from __future__ import annotations
from dataclasses import dataclass
from typing import List
import copy

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

    def modify(self, turns: List[StatefulTurn], inplace: bool,) -> List[StatefulTurn]:
        if not self.switched_on:
            return turns
        return [
            turn
            for turn in turns
            if turn.state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
            != ACTION_UNLIKELY_INTENT_NAME
        ]


@dataclass
class IfLastTurnWasUserTurnKeepEitherTextOrNonText(TurnSequenceModifier[StatefulTurn]):
    """Removes (intent and entities) or text from the last turn if it was a user turn.

    Only does so during inference. During training, nothing is removed.

    TODO: why was this always applied - shouldn't it only be done if intent/entities
    are targets? (in that case, the new label extractors would take care of this)
    """

    keep_text: bool = True

    def modify(self, turns: List[StatefulTurn], inplace: bool,) -> List[StatefulTurn]:

        if turns and turns[-1].actor == Actor.USER:
            last_turn = turns[-1]
            if not inplace:
                last_turn = copy.deepcopy(last_turn)
            remove = [INTENT, ENTITIES] if self.keep_text else [TEXT]
            for key in remove:
                last_turn.state.get(USER, {}).pop(key, None)
            if not inplace:
                turns[-1] = last_turn
        return turns


@dataclass
class RemoveUserTextIfIntentFromEveryTurn(TurnSequenceModifier[StatefulTurn]):
    """Removes the text if there is an intent in the user substate of every turn.

    This is always applied - during training as well as during inference.
    """

    def modify(self, turns: List[StatefulTurn], inplace: bool) -> List[StatefulTurn]:

        for idx in range(len(turns)):
            turn = turns[idx]
            if inplace:
                turn = copy.deepcopy(turn)
            user_substate = turn.state.get(USER, {})
            if TEXT in user_substate and INTENT in user_substate:
                del user_substate[TEXT]
            if inplace:
                turns[idx] = turn
        return turns
