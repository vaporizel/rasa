import logging
from typing import Tuple, List, Optional, Dict, Text, Any
import numpy as np


from rasa.core.turns.stateful.stateful_turn import StatefulTurn
from rasa.core.turns.stateful.stateful_turn_featurizers import (
    BasicStatefulTurnFeaturizer,
)
from rasa.core.turns.stateful.stateful_turn_labels import (
    ExtractActionFromLastTurn,
    ExtractEntitiesAndTextFromLastUserTurn,
)
from rasa.core.turns.stateful.stateful_turn_handling import (
    IfLastTurnWasUserTurnKeepEitherTextOrNonText,
    RemoveTurnsWithPrevActionUnlikelyIntent,
    RemoveUserTextIfIntentFromEveryTurn,
)
from rasa.core.turns.to_dataset.basic_label_encoders import (
    ApplyEntityTagsEncoder,
    IndexerFromLabelExtractor,
)
from rasa.core.turns.to_dataset.basic_turn_handling import (
    EndsWithPredictableActionExecuted,
    KeepMaxHistory,
)
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.turns.to_dataset.dataset import (
    DatasetFromTurns,
    FeaturizedDatasetFromTurns,
    LabelFeaturizationPipeline,
)
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import ACTION_NAME, ENTITY_TAGS, INTENT, ENTITIES
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.training_data.features import Features


# from rasa.utils.tensorflow.constants import LABEL_PAD_ID

FEATURIZER_FILE = "featurizer.json"

logger = logging.getLogger(__name__)


class InvalidStory(RasaException):
    """Exception that can be raised if story cannot be featurized."""

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidStory, self).__init__()

    def __str__(self) -> Text:
        return self.message


class TrackerFeaturizer:
    """Base class for actual tracker featurizers."""

    def __init__(
        self, state_featurizer: Optional[SingleStateFeaturizer] = None
    ) -> None:
        """Initializes the tracker featurizer.

        Args:
            state_featurizer: The state featurizer used to encode tracker states.
        """
        pass

    def prepare_for_featurization(
        self,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
    ) -> None:
        """

        training

        """
        pass

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
    ]:
        """

        used by ML-policies during training

        """
        pass

    def create_state_features(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[Dict[Text, List[Features]]]]:
        """

        used by ML policies during inference

        """
        pass

    def training_states_and_labels(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """

        used by rule policies during training (no unit tests for this)

        """
        pass

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[State]]:
        """

        used by rule-policies during inference
        """
        raise NotImplementedError(
            "Featurizer must have the capacity to create feature vector"
        )


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):

    LABEL_NAME = "action"

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
    ) -> None:
        super().__init__(state_featurizer)
        self.max_history = max_history
        self.remove_duplicates = remove_duplicates

        # (2.2) un-featurized dataset generators - for prediction

        will_be_set_later = True  # i.e. Reminder for an ugly workaround
        modifiers = [
            RemoveTurnsWithPrevActionUnlikelyIntent(switched_on=will_be_set_later),
            KeepMaxHistory(max_history=max_history),
            IfLastTurnWasUserTurnKeepEitherTextOrNonText(
                on_training=False, keep_text=will_be_set_later
            ),
            RemoveUserTextIfIntentFromEveryTurn(on_training=False),
        ]
        self._ignore_action_unlikely = modifiers[0]
        self._last_turn_handling = modifiers[-2]

        label_extractors = []

        self._raw_inference = DatasetFromTurns(
            filters=None,
            ignore_duplicates=remove_duplicates,
            modifiers=modifiers,
            label_extractors=label_extractors,
        )

    def prepare_for_featurization(
        self,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
    ) -> None:
        """

        preparation for create_state_features (inference) and
        featurize_trackers (training) - used by ML policies

        """

        # (1.1) featurized dataset generators - for training

        filters = [EndsWithPredictableActionExecuted()]  # FIXME

        preprocessing = [RemoveUserTextIfIntentFromEveryTurn()]

        will_be_set_later = True  # ugly workaround - is this flexibility needed?
        modifiers = [
            RemoveTurnsWithPrevActionUnlikelyIntent(switched_on=will_be_set_later),
            KeepMaxHistory(max_history=self.max_history, offset_for_training=+1),
        ]
        self._ignore_action_unlikely = modifiers[0]

        turn_featurizer = BasicStatefulTurnFeaturizer()

        label_handling = [
            (
                ACTION_NAME,
                LabelFeaturizationPipeline(
                    extractor=ExtractActionFromLastTurn(),
                    featurizer=None,
                    indexer=IndexerFromLabelExtractor(),
                ),
            ),
            # (
            #     INTENT,
            #     LabelFeaturizationPipeline(
            #         extractor=ExtractIntentFromLastUserTurn(),
            #         featurizer=LabelFeaturizerViaLookup(attribute=INTENT),
            #         indexer=IndexerFromLabelExtractor(),
            #     ),
            # ),
            (
                ENTITY_TAGS,
                LabelFeaturizationPipeline(
                    extractor=ExtractEntitiesAndTextFromLastUserTurn(),
                    featurizer=ApplyEntityTagsEncoder(bilou_tagging=bilou_tagging),
                    indexer=None,  # this won't work
                ),
            ),
        ]
        self._feat_training = FeaturizedDatasetFromTurns(
            preprocessing=preprocessing,
            filters=filters,
            ignore_duplicates=self.remove_duplicates,
            modifiers=modifiers,
            turn_featurizer=turn_featurizer,
            label_handling=label_handling,
        )

        self._feat_training.train_featurizers_and_indexers(
            domain=domain, interpreter=interpreter
        )

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
    ]:
        """

        used by ML policies during training

        """

        # FIXME: was is this done here *and* public - if so, why?
        self.prepare_for_featurization(
            domain=domain, interpreter=interpreter, bilou_tagging=bilou_tagging
        )

        trackers_as_states = []
        trackers_as_labels = []
        trackers_as_entities = []

        # TODO: do we need this flexibility?
        self._ignore_action_unlikely.switched_on = ignore_action_unlikely_intent

        for tracker in trackers:

            stateful_turns = StatefulTurn.parse(
                tracker=tracker,
                domain=domain,
                omit_unset_slots=False,
                ignore_rule_only_turns=False,  # no need to during training..
                rule_only_data=False,  # no need to during training..
            )

            for inputs, label_features, label_indices in self._feat_training.generate(
                turns=stateful_turns, interpreter=interpreter, training=True
            ):

                trackers_as_states.append(inputs)
                trackers_as_labels.append(label_indices[ACTION_NAME])
                trackers_as_entities.append(label_features[ENTITY_TAGS])

        return trackers_as_states, np.array(trackers_as_labels), trackers_as_entities

    def create_state_features(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[Dict[Text, List[Features]]]]:
        """

        used by ML policies during inference

        """
        pass

    def training_states_and_labels(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """

        used by rule policies during training (no unit tests for this)

        """
        pass

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[State]]:
        """

        used by rule-policies during inference

        """

        # TODO: is this really needed? - these should be fixed per policy
        self._ignore_action_unlikely.switched_on = ignore_action_unlikely_intent
        self._last_turn_handling.keep_text = use_text_for_last_user_input

        trackers_as_states = []
        for tracker in trackers:

            stateful_turns = StatefulTurn.parse(
                tracker=tracker,
                domain=domain,
                omit_unset_slots=False,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=False,
            )

            modified_turns, _ = next(
                self._raw_inference.generate(stateful_turns, training=False)
            )

            trackers_as_states.append([turn.state for turn in modified_turns])

        return trackers_as_states


'''
class IntentMaxHistoryTrackerFeaturizer(MaxHistoryTrackerFeaturizer):
    """Truncates the tracker history into `max_history` long sequences.

    Creates training data from trackers where intents are the output prediction
    labels. Tracker state sequences which represent policy input are truncated
    to not excede `max_history` states.
    """

    LABEL_NAME = "intent"

    @classmethod
    def _convert_labels_to_ids(
        cls, trackers_as_intents: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        """Converts a list of labels to a matrix of label ids.

        The number of rows is equal to `len(trackers_as_intents)`. The number of
        columns is equal to the maximum number of positive labels that any training
        example is associated with. Rows are padded with `LABEL_PAD_ID` if not all rows
        have the same number of labels.

        Args:
            trackers_as_intents: Positive example label ids
                associated with each training example.
            domain: The domain of the training data.

        Returns:
           A matrix of label ids.
        """
        # store labels in numpy arrays so that it corresponds to np arrays
        # of input features
        label_ids = [
            [domain.intents.index(intent) for intent in tracker_intents]
            for tracker_intents in trackers_as_intents
        ]

        return np.array(cls._pad_label_ids(label_ids))

    @staticmethod
    def _pad_label_ids(label_ids: List[List[int]]) -> List[List[int]]:
        """Pads label ids so that all are of the same length.

        Args:
            label_ids: Label ids of varying lengths

        Returns:
            Label ids padded to be of uniform length.
        """
        # If `label_ids` is an empty list, no padding needs to be added.
        if not label_ids:
            return label_ids

        # Add `LABEL_PAD_ID` padding to labels array so that
        # each example has equal number of labels
        multiple_labels_count = [len(a) for a in label_ids]
        max_labels_count = max(multiple_labels_count)
        num_padding_needed = [max_labels_count - len(a) for a in label_ids]

        padded_label_ids = []
        for ids, num_pads in zip(label_ids, num_padding_needed):
            padded_row = list(ids) + [LABEL_PAD_ID] * num_pads
            padded_label_ids.append(padded_row)
        return padded_label_ids

    def training_states_labels_and_entities(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms trackers to states, intent labels, and entity data.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_action_unlikely_intent: Whether to remove `action_unlikely_intent`
                from training states.

        Returns:
            Trackers as states, labels, and entity data.
        """
        example_states = []
        example_entities = []

        # Store of example hashes (of both states and labels) for removing
        # duplicate training examples.
        hashed_examples = set()
        # Mapping of example state hash to set of
        # positive labels associated with the state.
        state_hash_to_label_set: defaultdict[int, Set[Text]] = defaultdict(set)

        logger.debug(
            f"Creating states and {self.LABEL_NAME} label examples from "
            f"collected trackers "
            f"(by {type(self).__name__}({type(self.state_featurizer).__name__}))..."
        )
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:

            for states, label, entities in self._extract_examples(
                tracker,
                domain,
                omit_unset_slots=omit_unset_slots,
                ignore_action_unlikely_intent=ignore_action_unlikely_intent,
            ):

                if self.remove_duplicates:
                    hashed = self._hash_example(states, label)
                    if hashed in hashed_examples:
                        continue
                    hashed_examples.add(hashed)

                # Store all positive labels associated with a training state.
                state_hash = self._hash_example(states)

                # Only add unique example states unless `remove_duplicates` is `False`.
                if (
                    not self.remove_duplicates
                    or state_hash not in state_hash_to_label_set
                ):
                    example_states.append(states)
                    example_entities.append(entities)

                state_hash_to_label_set[state_hash].add(label[0])

                pbar.set_postfix({f"# {self.LABEL_NAME}": f"{len(example_states):d}"})

        # Collect positive labels for each state example.
        example_labels = [
            list(state_hash_to_label_set[self._hash_example(state)])
            for state in example_states
        ]

        self._remove_user_text_if_intent(example_states)

        logger.debug(f"Created {len(example_states)} {self.LABEL_NAME} examples.")

        return example_states, example_labels, example_entities

    def _extract_examples(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Iterator[Tuple[List[State], List[Text], List[Dict[Text, Any]]]]:
        """Creates an iterator over training examples from a tracker.

        Args:
            tracker: The tracker from which to extract training examples.
            domain: The domain of the training data.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_action_unlikely_intent: Whether to remove `action_unlikely_intent`
                from training states.

        Returns:
            An iterator over example states, labels, and entity data.
        """
        tracker_states = self._create_states(
            tracker, domain, omit_unset_slots=omit_unset_slots
        )
        events = tracker.applied_events()

        if ignore_action_unlikely_intent:
            tracker_states = self._remove_action_unlikely_intent_from_states(
                tracker_states
            )
            events = self._remove_action_unlikely_intent_from_events(events)

        label_index = 0
        for event in events:

            if isinstance(event, ActionExecuted):
                label_index += 1

            elif isinstance(event, UserUttered):

                sliced_states = self.slice_state_history(
                    tracker_states[:label_index], self.max_history
                )
                label = [event.intent_name or event.text]
                entities = [{}]

                yield sliced_states, label, entities

    @staticmethod
    def _cleanup_last_user_state_with_action_listen(
        trackers_as_states: List[List[State]],
    ) -> List[List[State]]:
        """Removes the last tracker state if the previous action is `action_listen`.

        States with the previous action equal to `action_listen` correspond to states
        with a new user intent. This information is what `UnexpecTEDIntentPolicy` is
        trying to predict so it needs to be removed before obtaining a prediction.

        Args:
            trackers_as_states: Trackers converted to states

        Returns:
            Filtered states with last `action_listen` removed.
        """
        for states in trackers_as_states:
            if not states:
                continue
            last_state = states[-1]
            if rasa.shared.core.trackers.is_prev_action_listen_in_state(last_state):
                del states[-1]

        return trackers_as_states

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[State]]:
        """Transforms trackers to states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.
            ignore_action_unlikely_intent: Whether to remove any states containing
                `action_unlikely_intent` from prediction states.

        Returns:
            Trackers as states for prediction.
        """

        trackers_as_states = []
        for tracker in trackers:

            stateful_turns = StatefulTurn.parse(
                tracker=tracker,
                domain=domain,
                omit_unset_slots=False,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=rule_only_data,
            )

            pipeline = []
            if ignore_action_unlikely_intent:
                pipeline.append(RemoveTurnsWithPrevActionUnlikelyIntent())
            pipeline += [
                RemoveTextOrNonTextFromLastTurnIfUserTurn(
                    remove_non_text=use_text_for_last_user_input
                ),
                RemoveUserTextIfIntentFromEveryTurn(),
                RemoveLastTurnIfUserTurn(),
                KeepMaxHistory(self.max_history),
            ]

            modified_turns = TurnSequenceModifier.apply_all(pipeline, stateful_turns)
            trackers_as_states.append([turn.state for turn in modified_turns])

        return trackers_as_states


'''
