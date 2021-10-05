from __future__ import annotations
from abc import ABC
from typing import (
    Any,
    List,
    Optional,
    Dict,
    Text,
    Tuple,
    TypeVar,
)

from rasa.core.turns.turn import Turn
from rasa.core.turns.stateful.stateful_turn import StatefulTurn
from rasa.core.turns.to_dataset.dataset import LabelFromTurnsExtractor
from rasa.shared.core.constants import PREVIOUS_ACTION, USER
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    INTENT,
    TEXT,
)


ACTION_NAME_OR_TEXT = "action_name_or_text"


T = TypeVar("T")


class ExtractAttributeFromLastUserTurn(
    LabelFromTurnsExtractor[StatefulTurn, Tuple[Optional[Text], Optional[T]]], ABC
):
    """Extracts an attribute from the user-substate of the last user turn.

    The information will be removed from the user sub-state of the last user turn
    and of all following turns.

    Along with the attribute, the user text will be returned. However, the user text
    will not be removed from any substate.
    """

    def __init__(self, attribute: Text) -> None:
        self._attribute = attribute

    def __call__(
        self, turns: List[StatefulTurn], training: bool = True,
    ) -> Tuple[List[Turn], Tuple[Optional[Text], Optional[T]]]:
        last_user_turn_idx = Turn.get_index_of_last_user_turn(turns)
        if training:
            last_user_turn = turns[last_user_turn_idx]
            state = last_user_turn.state.get(USER, {})
            raw_info = (
                last_user_turn.state.get(TEXT, None),
                state.get(self._attribute, None),
            )
            # it is the last user turn, but the states in all subsequent bot turns
            # contain information about the last user turn
            for idx in range(last_user_turn_idx, len(turns)):
                turns[idx].state.get(USER, {}).pop(self._attribute, None)

        else:
            raw_info = (None, None)
        return turns, raw_info

    def from_domain(self, domain: Domain,) -> List[Tuple[Optional[Text], T]]:
        raise NotImplementedError()


class ExtractIntentFromLastUserTurn(LabelFromTurnsExtractor[StatefulTurn, Text]):
    """Extract the intent from the last user turn.

    During training, the intent will be removed from the user sub-state of the last
    user turn and of all following turns.
    During inference, nothing will be removed.
    """

    def __init__(self) -> None:
        self.extractor = ExtractAttributeFromLastUserTurn(attribute=INTENT)

    def __call__(
        self, turns: List[StatefulTurn], training: bool = True,
    ) -> Tuple[List[Turn], Tuple[Optional[Text], T]]:
        turns, (_, intent) = self.extractor(turns, training=training)
        if training and intent is None:
            raise RuntimeError("Could not extract an intent...")
        return turns, intent

    def from_domain(self, domain: Domain,) -> List[Text]:
        return domain.intents


class ExtractEntitiesFromLastUserTurn(
    LabelFromTurnsExtractor[StatefulTurn, Optional[Dict[Text, Any]]]
):
    """Extract the entities from the last user turn.

    During training, the entities will be removed from the user sub-state of the
    last user turn and of all following turns.
    During inference, nothing will be removed.
    """

    def __init__(self) -> None:
        self.extractor = ExtractAttributeFromLastUserTurn(attribute=ENTITIES)

    def __call__(
        self, turns: List[StatefulTurn], training: bool = True,
    ) -> Tuple[List[Turn], Optional[Dict[Text, Any]]]:
        turns, (_, entities) = self.extractor(turns, training=training)
        return turns, entities


class ExtractEntitiesAndTextFromLastUserTurn(
    LabelFromTurnsExtractor[StatefulTurn, Tuple[Text, Dict[Text, Any]]]
):
    """Extract the entities from the last user turn, and also return the text.

    During training, the entities will be removed from the user sub-state of the last
    user turn and of all following turns. The text will *not* be removed from
    any sub-state.
    During inference, nothing will be removed.
    """

    def __init__(self) -> None:
        self.extractor = ExtractAttributeFromLastUserTurn(attribute=INTENT)

    def __call__(
        self, turns: List[StatefulTurn], training: bool = True,
    ) -> Tuple[List[Turn], Tuple[Optional[Text], T]]:
        return self.extractor(turns, training=training)


class ExtractActionFromLastTurn(LabelFromTurnsExtractor[StatefulTurn, Text]):
    """Extracts the action from the last turn and removes that turn.

    During training, the last turn will be removed from the given sequence.
    During inference, the turns remain unchanged.
    """

    def __call__(
        self, turns: List[StatefulTurn], training: bool = True,
    ) -> Tuple[List[StatefulTurn], Text]:
        if training:
            prev_action = turns[-1].state.get(PREVIOUS_ACTION, {})
            # we prefer the action name but will use action text, if there is no name
            action = prev_action.get(ACTION_NAME, None)
            if not action:
                action = prev_action.get(ACTION_TEXT, None)
            if not action:
                raise RuntimeError("There must be an action we can extract....")
            # remove the whole turn
            turns = turns[:-1]
        else:
            action = ""
        return turns, action

    def from_domain(self, domain: Domain,) -> List[Text]:
        return domain.action_names_or_texts
