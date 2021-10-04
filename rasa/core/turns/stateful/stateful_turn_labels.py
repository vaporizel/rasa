from __future__ import annotations
from typing import (
    List,
    Optional,
    Dict,
    Text,
    Text,
    Tuple,
)

import numpy as np

from rasa.core.turns.turn import Turn
from rasa.core.turns.stateful.stateful_turn import StatefulTurn
from rasa.core.turns.utils.attribute_featurizer import FeaturizerUsingInterpreter
from rasa.core.turns.utils.multihot_encoder import MultiHotEncoder
from rasa.core.turns.utils.entity_tags_encoder import EntityTagsEncoder
from rasa.core.turns.utils.trainable import Trainable
from rasa.core.turns.dataset_from_turns import (
    FeaturizedLabelsFromTurns,
    LabelsFromTurns,
)
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.core.constants import PREVIOUS_ACTION, USER
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    INTENT,
    TEXT,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message


ACTION_NAME_OR_TEXT = "action_name_or_text"


class FeaturizeEntityFromLastUserTurnViaEntityTagEncoder(
    FeaturizedLabelsFromTurns[StatefulTurn], Trainable
):
    """Extracts and encodes the entity information from the last user state.

    Entity information will be removed from the user sub-state of the last user turn
    and of all following turns.
    However, a featurization of the entities is only computed during training.
    """

    def __init__(self, bilou_tagging: bool = False) -> None:
        super().__init__()
        self._bilou_tagging = bilou_tagging
        self._entity_tags_encoder

    def train(self, domain: Domain) -> None:
        self._entity_tags_encoder = EntityTagsEncoder(
            domain=domain, bilou_tagging=self.bilou_tagging,
        )
        self._trained = True

    def extract_and_featurize(
        self,
        turns: List[StatefulTurn],
        training: bool = True,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> Tuple[List[StatefulTurn], Optional[Dict[Text, List[Features]]]]:
        self.raise_if_not_trained()
        if interpreter is None:
            raise RuntimeError("Needs interpreter")

        last_user_turn_idx = Turn.get_index_of_last_user_turn(turns)
        if last_user_turn_idx is None:
            raise ValueError(
                "There is no user turn from which we could extract entity labels."
            )

        # encode the entities
        encoded_entities = None
        if training:
            last_user_turn = turns[last_user_turn_idx]
            encoded_entities = self._encode_entities(
                last_user_turn, interpreter=interpreter
            )

        # it is the last user turn, but the states in all subsequent bot turns
        # contain information about the last user turn
        for idx in range(last_user_turn_idx, len(turns)):
            turns[idx].state.get(USER, {}).pop(ENTITIES, None)

        return turns, encoded_entities

    def _encode_entities(
        self, turn: StatefulTurn, interpreter: NaturalLanguageInterpreter,
    ) -> Dict[Text, List[Features]]:

        # Don't bother encoding anyting if there are less than 2 entity tags,
        # because we won't train any entity extactor anyway.
        if self._entity_tags_encoder.entity_tag_spec.num_tags < 2:
            return {}

        if not USER in turn.state:
            return []

        # # FIXME: move this to postprocessing ->> if TEXT is removed then we also remove ENTITIES from the output.
        # # train stories support both text and intent,
        # # but if intent is present, the text is ignored
        # if INTENT in turn.state[USER]:
        #     return {}

        entities = turn.state[USER].get(ENTITIES, {})

        if not entities:
            return []

        text = turn.state[USER][TEXT]
        message = interpreter.featurize_message(Message(data={TEXT: text}))
        text_tokens = message.get(TOKENS_NAMES[TEXT])

        return self._entity_tags_encoder.encode(
            text_tokens=text_tokens, entities=entities
        )

    def encode_all(self, domain: Domain, interpreter: NaturalLanguageInterpreter):
        raise NotImplementedError()


class FeaturizeIntentFromLastUserTurnViaInterpreter(LabelsFromTurns[StatefulTurn]):
    """Extracts and encodes the intent information from the last user state.

    Removes intent information from the user sub-state of the last user turn
    and of all following turns, during training as well as during inference.
    However, a featurization of the intent is only computed during training.
    """

    def extract(
        self,
        turns: List[StatefulTurn],
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        training: bool = True,
    ) -> Tuple[List[StatefulTurn], Optional[Dict[Text, List[Features]]]]:
        self.raise_if_not_trained()
        last_user_turn_idx = Turn.get_index_of_last_user_turn(turns)
        if last_user_turn_idx is None:
            raise ValueError(
                "There is no user turn from which we could extract entity labels."
            )

        # encode the intent
        encoded_intent = None
        if training:
            last_user_turn = turns[last_user_turn_idx]
            encoded_intent = self._encode_intent(last_user_turn)

        # it is the last user turn, but the states in all subsequent bot turns
        # contain information about the last user turn
        for idx in range(last_user_turn_idx, len(turns)):
            turns[idx].state.get(USER, {}).pop(INTENT, None)

        return turns, encoded_intent

    def _encode_intent(
        self,
        turn: StatefulTurn,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> Dict[Text, List[Features]]:
        intent = turn.state.get(USER, {}).get(INTENT, None)
        if not intent:
            return {}
        return FeaturizerUsingInterpreter.featurize(
            message_data={INTENT: intent}, interpreter=interpreter
        )

    def encode_all(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> List[Dict[Text, List[Features]]]:
        return [self._encode_intent(intent, interpreter) for intent in domain.intents]


class ExtractActionFromLastTurn(LabelsFromTurns[StatefulTurn]):
    """Extracts the action from the last turn and removes that turn.

    The last turn will be removed from the given sequence only during training.
    During inference, the turns remain unchanged.
    """

    def extract(
        self, turns: List[StatefulTurn], training: bool = True,
    ) -> Tuple[List[Turn], Dict[Text, Text]]:
        if training:
            prev_action = turns[-1].state.get(PREVIOUS_ACTION, {})
            # we prefer the action name but will use action text, if there is no name
            action = prev_action.get(ACTION_NAME, None)
            if not action:
                action = prev_action.get(ACTION_TEXT, None)
            labels = {ACTION_NAME_OR_TEXT: action}
            # remove the whole turn
            turns = turns[-1]
        else:
            labels = {}
        return turns, labels

    def extract_all(self, domain: Domain,) -> Dict[Text, List[Text]]:
        return {ACTION_NAME_OR_TEXT: domain.action_names_or_texts}


class IndexFeaturizerFromExtractor(FeaturizedLabelsFromTurns[StatefulTurn], Trainable):
    def __init__(self, extractor=LabelsFromTurns[StatefulTurn]):
        self._extractor = extractor

    def train(self, domain: Domain) -> None:
        self._multihot_encoders = {
            key: MultiHotEncoder(values)
            for key, values in self._extractor.extract_all(domain=domain).items()
        }
        self._trained = True

    def extract_and_featurize(
        self,
        turns: List[StatefulTurn],
        training: bool = True,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> Tuple[List[Turn], Dict[Text, np.ndarray]]:
        self.raise_if_not_trained()
        extracted = self._extractor.extract(turns, training=training)
        if extracted:
            # Note that we only use an index array here to be consistent with the types
            # (and because we need to wrap it later on anyway).
            extracted = {
                key: self._multihot_encoders[key].encode_as_index_array([value])
                for key, value in extracted
            }
        return turns, extracted

    def featurize_all(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> Dict[Text, Dict[Text, List[np.ndarray]]]:
        self.raise_if_not_trained()
        return {
            key: {value: self._multihot_encoders[key].encode_as_index_array([value])}
            for key, value in self._extractor.extract_all(domain=domain).items()
        }
