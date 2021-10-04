from __future__ import annotations

from typing import (
    List,
    Optional,
    Type,
    Dict,
    Text,
    Union,
    Text,
    Tuple,
)

import scipy.sparse

from rasa.core.turns.stateful_turn import StatefulTurn
from rasa.core.turns.utils.attribute_featurizer import FeaturizerUsingInterpreter
from rasa.core.turns.utils.multihot_encoder import MultiHotEncoder
from rasa.core.turns.utils.entity_tags_encoder import EntityTagsEncoder
from rasa.core.turns.turn_featurizer import (
    InputPostProcessor,
    OutputExtractor,
    TrainableOutputExtractor,
    TurnFeaturizer,
)
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.core.constants import ACTIVE_LOOP, PREVIOUS_ACTION, SLOTS, USER
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    INTENT,
    TEXT,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message


class BasicStatefulTurnFeaturizer(TurnFeaturizer[StatefulTurn]):
    """Featurize a single stateful turn.

    We assume here that the events that have been used to generate the Stateful
    turns have been augmented with NLU information already (i.e. via annotations
    in the training data or via the application of an NLU pipeline during prediction
    time).
    That is, we solely used to turn information into `Features` here.

    More specifically, we require that:
    1. a turn's state always contains an action name in its 'previous action'
       substate - if there is an previous action substate - and an intent in
       its 'user' substate - if there is a user substate.
    2. the attributes of all substates are mutually different.
    """

    def __init__(self):
        self._multihot_encoders = None

    def train(self, domain: Domain) -> None:
        self._multihot_encoders = {
            attribute: MultiHotEncoder(dimension_names=dimension_names)
            for attribute, dimension_names in [
                (INTENT, domain.intents),
                # NOTE: even though this contains action names and texts, we will
                # only create multihot encodings for action names
                (ACTION_NAME, domain.action_names_or_texts),
                (ENTITIES, domain.entity_states),
                (SLOTS, domain.slot_states),
                (ACTIVE_LOOP, domain.form_names),
            ]
        }

    def featurize(
        self, turn: StatefulTurn, interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List[Features]]:
        """Encodes the given state with the help of the given interpreter.

        Here, we assume that ...
        1. a turn's state always contains an action name
           in its 'previous action' substate and an intent in its 'user' substate.
        2. the attributes of all substates are mutually different.

        Args:
            state: The state to encode
            interpreter: The interpreter used to encode the state

        Returns:
            A dictionary of state_type to list of features.
        """
        state = turn.state
        features = {}

        # This method is called during both prediction and training,
        # `self._use_regex_interpreter == True` means that core was trained
        # separately, therefore substitute interpreter based on some trained
        # nlu model with default RegexInterpreter to make sure
        # that prediction and train time features are the same
        if self._use_regex_interpreter and not isinstance(
            interpreter, RegexInterpreter
        ):
            interpreter = RegexInterpreter()

        # We always featurize the previous action substate. However, if the turn is
        # attributed to the user, then the previous action will just be an
        # `ActionListen`.
        if PREVIOUS_ACTION in state:
            sub_state = state[PREVIOUS_ACTION]

            # Attempt to featurize the substate via the interpreter, but...
            features_via_interpreter = FeaturizerUsingInterpreter.featurizer(
                sub_state, interpreter
            )
            features.update(features_via_interpreter)

            # ... note that the only featurizer that could featurize this attribute
            # is the `CountVectorizer`, which will create sparse sequence and sentence
            # features.
            # We replace these features with some sparse sentence features:
            if ACTION_NAME in features:
                features[ACTION_NAME] = self._sum_sparse_sequence_features(
                    features[ACTION_NAME]
                )

            # If there is an action name but no features yet, we create some:
            if ACTION_NAME not in features:
                # NOTE: here we assume there always is an action name
                features[ACTION_NAME] = self._multihot_encoding_for_action_name(
                    action_name=sub_state[ACTION_NAME]
                )

        # Remember that the user state might be empty despite the turn belonging to
        # the user because of the edge case where a turn just captures an action listen.
        if turn.actor == USER and USER in state:

            sub_state = state[USER]

            # Only featurize attributes != entities via the featurizers pipeline
            not_featurized_via_interpreter = {ENTITIES}

            # # We do not featurize the "text" if an intent is present
            # # FIXME: this is a replacement for  the
            # #  self._remove_user_text_if_intent(trackers_as_states)
            # # step  - that is only happenign in prediction_states functions
            # # TODO: move this to  post processing ...
            # if INTENT in sub_state:
            #     not_featurized_via_interpreter.add(TEXT)

            # Create some features via the interpreter
            features_via_interpreter = FeaturizerUsingInterpreter.featurize(
                sub_state,
                interpreter,
                attributes_excluded_from_result=not_featurized_via_interpreter,
            )
            features.update(features_via_interpreter)

            # Use a simple multihot encoding for entities
            if ENTITIES in sub_state:
                features[ENTITIES] = self._multihot_encoding_for_entities(
                    sub_state.get(ENTITIES)
                )

            # The only featurizer that could featurize this attribute is the
            # CountVectorizer, which will create sparse sequence and sentence
            # features.
            # We replace these features with some sparse sentence features:
            if INTENT in features:
                features[INTENT] = self._sum_sparse_sequence_features(features[INTENT])

            # If there is an intent but no features yet, we create some:
            if INTENT not in features:
                # NOTE: here we assume there always is an intent in the substate
                features[INTENT] = self._multihot_encoding_for_intent(sub_state[INTENT])

        # We always featurize the context, i.e. slots and loop information.
        if SLOTS in state:
            features[SLOTS] = self._multihot_encoding_for_slots_as_features(
                slots_as_features=state[SLOTS]
            )
        if ACTIVE_LOOP in state:
            features[ACTIVE_LOOP] = self._multihot_encoding_for_active_loop(
                active_loop_name=state[ACTIVE_LOOP]["name"]
            )
        return features

    def _multihot_encoding_for_intent(self, intent: Text) -> Features:
        return self._multihot_encoders[INTENT].encode_as_sparse_sentence_feature(
            {intent: 1}
        )

    def _multihot_encoding_for_action_name(
        self, action_name: Optional[Text]
    ) -> Features:
        return self._multihot_encoders[ACTION_NAME].encode_as_sparse_sentence_feature(
            {action_name: 1} if action_name else {}
        )

    def _multihot_encoding_for_entities(self, entities: List[Text]) -> Features:
        return self._multihot_encoders[ENTITIES].encode_as_sparse_sentence_feature(
            {entity: 1 for entity in (entities or [])}
        )

    def _multihot_encoding_for_active_loop(self, active_loop_name: Text) -> Features:
        return self._multihot_encoders[ACTIVE_LOOP].encode_as_sparse_sentence_feature(
            {active_loop_name: 1}
        )

    def _multihot_encoding_for_slots_as_features(
        self, slots_as_features: Dict[Text, Union[Text, Tuple[float, ...]]]
    ) -> Features:
        # cf. `slot_states` in `domain.py`
        dim_to_value = {
            f"{slot_name}_{i}": value
            for slot_name, slot_as_feature in slots_as_features.items()
            for i, value in enumerate(slot_as_feature)
            if isinstance(slot_as_feature, list)
        }
        return self._multihot_encoders[SLOTS](dim_to_value)

    @staticmethod
    def _sum_sparse_sequence_features(features: List[Features],) -> List[Features]:
        return [
            Features(
                scipy.sparse.coo_matrix(feature.features.sum(0)),
                FEATURE_TYPE_SENTENCE,
                feature.attribute,
                feature.origin,
            )
            for feature in features
            if feature.is_sparse() and feature.feature_type == FEATURE_TYPE_SEQUENCE
        ]


class EncodeEntityFromLastUserTurn(TrainableOutputExtractor[StatefulTurn]):
    """Encode the ... and removes corresponding entities from input encodings. """

    def __init__(self, bilou_tagging: bool = False) -> None:
        super().__init__()
        self._bilou_tagging = bilou_tagging
        self._entity_tags_encoder

    def compatible_with(self, turn_featurizer_type: Type[TurnFeaturizer]) -> bool:
        return turn_featurizer_type == BasicStatefulTurnFeaturizer

    def train(self, domain: Domain) -> None:
        self._entity_tags_encoder = EntityTagsEncoder(
            domain=domain, bilou_tagging=self.bilou_tagging,
        )
        self._trained = True

    def extract(
        self,
        turns: List[StatefulTurn],
        turns_as_inputs: List[Dict[Text, Features]],
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> Tuple[List[Dict[Text, Features]], Union]:

        self.raise_if_not_trained()

        pass

        last_user_turn = ...

        # NOTE: cannot reuse encoding from featurizer,...

    def encode_entities(
        self, turn: StatefulTurn, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:

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

    def encode_all(self, domain: Domain):
        raise NotImplementedError()


class ExtractIntentFromLastUserTurn(OutputExtractor[StatefulTurn]):
    def compatible_with(self, turn_featurizer_type: Type[TurnFeaturizer]) -> bool:
        return turn_featurizer_type == BasicStatefulTurnFeaturizer

    def extract(
        self,
        turns: List[StatefulTurn],
        turns_as_inputs: List[Dict[Text, Features]],
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> Tuple[List[Dict[Text, Features]], Union]:

        # NOTE: since it's looking for last user state this alway sworks
        pass

        last_user_turn = ...

        # NOTE: CAN reuse encoding from featurizer,... just pop from last user turn

    def encode_all(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> List[Dict[Text, List[Features]]]:
        """Encodes all relevant labels from the domain using the given interpreter.

        Args:
            domain: The domain that contains the labels.
            interpreter: The interpreter used to encode the labels.

        Returns:
            A list of encoded labels.
        """
        return [self._encode_intent(intent, interpreter) for intent in domain.intents]


class ExtractActionFromLastTurn(OutputExtractor[StatefulTurn]):
    def compatible_with(self, turn_featurizer_type: Type[TurnFeaturizer]) -> bool:
        return turn_featurizer_type == BasicStatefulTurnFeaturizer

    def extract(
        self,
        turns: List[StatefulTurn],
        turns_as_inputs: List[Dict[Text, Features]],
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> Tuple[List[Dict[Text, Features]], Union]:
        if not training:
            return turns_as_inputs, {}
        pass

        # TODO: just pop from last turn..

    def encode_all_labels(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter,
    ) -> List[Dict[Text, List[Features]]]:
        """Encode all action from the domain.
        Args:
            domain: The domain that contains the actions.
            precomputations: Contains precomputed features and attributes.
        Returns:
            A list of encoded actions.
        """
        return [
            self._encode_action(action, interpreter)
            for action in domain.action_names_or_texts
        ]


# FIXME: apply to state and featurization... ?
class DropTextIfIntentDuringPredictionTime(InputPostProcessor):
    def __call__(self, turns_as_inputs: List[Dict[Text, Features]], training: bool):
        if training:
            return turns_as_inputs

        # TODO:
