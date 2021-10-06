from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Set,
    Text,
    TypeVar,
    Generic,
    Tuple,
    Optional,
)
from dataclasses import dataclass
import logging

import numpy as np

from rasa.core.turns.turn import Turn
from rasa.core.turns.to_dataset.utils.trainable import Trainable
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

TurnType = TypeVar("TurnType")
RawLabelType = TypeVar("RawLabelType")
FeatureCollection = Dict[Text, List[Features]]


def turns_to_str(turns: List[TurnType]) -> Text:
    indent = " " * 2
    turns = "\n".join(f"{indent}{idx:2}. {turn}" for idx, turn in enumerate(turns))
    return f"[\n{turns}\n]"


class TurnSubSequenceGenerator(Generic[TurnType], ABC):
    """Generates sub-sequencesfrom a sequence of turns."""

    def __init__(
        self,
        preprocessing: Optional[List[TurnSequenceModifier[TurnType]]],
        filters: Optional[List[TurnSequenceValidation[TurnType]]],
        ignore_duplicates: bool,
        modifiers: Optional[List[TurnSequenceModifier[TurnType]]],
        filter_results: bool,
    ) -> None:
        """
        Args:
            filters: only applied during training
            ...
        """
        self._preprocessing = preprocessing or []
        self._filters = filters or []
        self._ignore_duplicates = ignore_duplicates
        self._cache: Set[Tuple[int, ...]] = set()
        self._modifiers = modifiers or []
        self._filter_results = filter_results

    def generate_subsequence(
        self, turns: List[TurnType], training: bool, limit: Optional[int] = None,
    ) -> Iterator[List[Turn]]:
        """

        During training, the whole sequence of turns is processed.

        During validation, we extract multiple (or no) sub-sequences of turns from the
        given sequence of turns: Each subsequence of turns from the 0-th to the i-th
        turn that passes the given sequence filters and is not a duplicate of any
        other subsequence created so far.

        In both cases, the same modifiers are applied.

        Args:
            training: Indicates whether we are in training mode.

        """
        steps = [len(turns) + 1] if not training else range(1, len(turns) + 1)
        num_generated = 0

        logger.debug(f"Generate subsequences from:\n{turns_to_str(turns)}")

        preprocessed_turns = TurnSequenceModifier.modify_all(
            self._preprocessing, turns, inplace=True,
        )

        logger.debug(f"Applied preprocessing:\n{turns_to_str(turns)}")

        for idx in steps:

            if limit and num_generated >= limit:
                return

            # we'll make a copy of this subsequence, once we know we continue with it
            subsequence = preprocessed_turns[:idx]

            logger.debug(
                f"Attempt to generate from subsequence:\n{turns_to_str(subsequence)}"
            )

            # during training - skip if it does not pass filters
            if training and not TurnSequenceValidation.apply_all(
                self._filters, subsequence
            ):
                logger.debug(f"Failed (did not pass filters {self._filters})")
                continue

            # during training - skip if it is a duplicate
            if training and self._ignore_duplicates:
                id = tuple(hash(turn) for turn in subsequence)
                if id in self._cache:
                    logger.debug(f"Failed (duplicate of other subsequence)")
                    continue
                else:
                    self._cache.add(id)

            # always apply postprocessing
            subsequence = TurnSequenceModifier.apply_all(
                self._modifiers, subsequence, training=training,
            )

            if self._modifiers:
                logger.debug(f"Modified subsequence:\n{turns_to_str(subsequence)}")

            # check if filters still pass (we modified the sequence)
            if training and not TurnSequenceValidation.apply_all(
                self._filters, subsequence
            ):
                logger.debug(f"Failed (did not pass filters {self._filters})")
                continue

            num_generated += 1
            yield subsequence


@dataclass
class TurnSequenceValidation(Generic[TurnType]):
    """Determines whether or not a given list of turns satisfies some criteria."""

    def __call__(self, turns: List[TurnType]) -> bool:
        pass

    @classmethod
    def apply_all(
        self, validations: List[TurnSequenceValidation[TurnType]], turns: List[TurnType]
    ) -> bool:
        return all(validation(turns) for validation in validations)


@dataclass
class TurnSequenceModifier(Generic[TurnType], ABC):
    """Returns a modified list of turns.

    Must not modify the given list of turns.
    """

    on_training: bool = True
    on_inference: bool = True

    @abstractmethod
    def modify(self, turns: List[TurnType], inplace: bool) -> List[TurnType]:
        """Returns a modified turn sequence.

        Args:
            turns: a list of turns
            inplace: if this is set to `True` then the single turns of the given
               sequence of turns may be modified inplace; otherwise the given turns
               must not be modified but the returned turn sequence may contain new
               turn objects
        Returns:
            a modified turn sequence
        """
        pass

    def modify_all(
        modifiers: List[TurnSequenceModifier[TurnType]],
        turns: List[TurnType],
        inplace: bool,
    ) -> List[TurnType]:
        for modifier in modifiers:
            turns = modifier.modify(turns, inplace=inplace)
        return turns

    def apply(self, turns: List[TurnType], training: bool) -> List[TurnType]:
        """Modifies turns inplace during inference, and not-inplace during training.

        During inference, our datasets only generate one sequence from the turn
        sequence extracted from a tracker. During training, we generate several
        sub-sequences from the same turn sequence (extracted from the same tracker).
        Hence, inplace modifications could only have undesired side-effects during
        training.
        """
        if training and self.on_training:
            return self.modify(turns, inplace=True)
        if not training and self.on_inference:
            return self.modify(turns, inplace=False)
        return turns

    @staticmethod
    def apply_all(
        modifiers: List[TurnSequenceModifier[TurnType]],
        turns: List[TurnType],
        training: bool,
    ) -> List[TurnType]:
        for modifier in modifiers:
            turns = modifier.apply(turns, training=training)
        return turns


class DatasetFromTurns(TurnSubSequenceGenerator[TurnType], Generic[TurnType]):
    """Generates labeled and modified subsequences of turns."""

    def __init__(
        self,
        label_extractors: Dict[Text, LabelFromTurnsExtractor[TurnType, Any]],
        preprocessing: Optional[List[Optional[TurnSequenceModifier[TurnType]]]] = None,
        filters: Optional[List[Optional[TurnSequenceValidation[TurnType]]]] = None,
        ignore_duplicates: bool = False,
        modifiers: Optional[List[Optional[TurnSequenceModifier[TurnType]]]] = None,
        filter_results: bool = False,
    ):
        super().__init__(
            preprocessing=preprocessing,
            filters=filters,
            ignore_duplicates=ignore_duplicates,
            modifiers=modifiers,
            filter_results=filter_results,
        )
        self._label_extractors = label_extractors

    def generate(
        self, turns: List[TurnType], training: bool,
    ) -> Iterator[Tuple[List[Turn], Optional[Dict[Text, Any]]]]:
        for processed_turns in self.generate_subsequence(turns, training=training):
            processed_turns, outputs = LabelFromTurnsExtractor.apply_all(
                self._label_extractors, turns=processed_turns, training=training,
            )
            logger.debug(f"Extracted labels:\n{turns_to_str(processed_turns)}")
            yield processed_turns, outputs


class FeaturizedDatasetFromTurns(
    TurnSubSequenceGenerator[TurnType], Generic[TurnType], Trainable
):
    """Generates and encodes labeled and modified subsequences of turns."""

    # TODO: add __len__ implementation that skips the featurization

    def __init__(
        self,
        turn_featurizer: TurnFeaturizer[TurnType],
        label_handling: List[Tuple[Text, LabelFeaturizationPipeline[TurnType, Any]]],
        preprocessing: Optional[List[Optional[TurnSequenceModifier[TurnType]]]] = None,
        filters: Optional[List[Optional[TurnSequenceValidation[TurnType]]]] = None,
        ignore_duplicates: bool = False,
        modifiers: Optional[List[Optional[TurnSequenceModifier[TurnType]]]] = None,
        filter_results: bool = False,
    ):
        super().__init__(
            preprocessing=preprocessing,
            filters=filters,
            ignore_duplicates=ignore_duplicates,
            modifiers=modifiers,
            filter_results=filter_results,
        )
        self._turn_featurizer = turn_featurizer
        self._label_handling = label_handling

    def train_featurizers_and_indexers(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._turn_featurizer.train(domain=domain, interpreter=interpreter)
        for _, pipeline in self._label_handling:
            if pipeline.featurizer:
                pipeline.featurizer.train(domain=domain, interpreter=interpreter)
            if pipeline.indexer:
                pipeline.indexer.train(domain=domain, extractor=pipeline.extractor)
        self._trained = True

    def generate(
        self,
        turns: List[TurnType],
        interpreter: NaturalLanguageInterpreter,
        training: bool,
    ) -> Iterator[
        Tuple[List[FeatureCollection], FeatureCollection, Dict[Text, np.ndarray]]
    ]:
        self.raise_if_not_trained()
        for subsequence in self.generate_subsequence(turns, training=training):

            processed_turns = subsequence

            # extract and featurize labels during training
            collected_features = {}
            collected_indices = {}
            if training:

                for label_name, label_handling in self._label_handling:
                    processed_turns, next_features, next_indices = label_handling(
                        processed_turns, training=training, interpreter=interpreter
                    )  # TODO: return raw labels as well for debugging
                    collected_features[label_name] = next_features
                    collected_indices[label_name] = next_indices
                logger.debug(f"Extracted labels:\n{turns_to_str(processed_turns)}")

            # featurize the (remaining) input (during training)
            processed_turns_featurized = [
                self._turn_featurizer.featurize(turn, interpreter=interpreter)
                for turn in processed_turns
            ]
            yield processed_turns_featurized, collected_features, collected_indices


@dataclass
class LabelFeaturizationPipeline(Generic[TurnType, RawLabelType]):
    """Extracts labels, featurizes them and converts them to labels."""

    extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    featurizer: Optional[LabelFeaturizer[RawLabelType]]
    indexer: Optional[LabelIndexer[TurnType, RawLabelType]]

    def __call__(
        self,
        turns: List[TurnType],
        training: bool = True,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> Tuple[List[TurnType], List[Features], np.ndarray]:
        turns, extracted_label = self.extractor(turns, training)
        features = []
        indices = np.array([])
        if training:
            if self.featurizer:
                features = self.featurizer.featurize(
                    extracted_label, interpreter=interpreter
                )
            if self.indexer and extracted_label:
                indices = self.indexer.index(extracted_label)
        return turns, features, indices


class LabelFromTurnsExtractor(Generic[TurnType, RawLabelType]):
    """Extracts label information from a sequence of turns."""

    @abstractmethod
    def __call__(
        self, turns: List[TurnType], training: bool = True,
    ) -> Tuple[List[TurnType], RawLabelType]:
        pass

    def from_domain(self, domain: Domain) -> List[RawLabelType]:
        # TODO: do we need to be able to handle a new domain here?
        raise NotImplementedError()

    @classmethod
    def apply_all(
        cls,
        label_extractors: List[Tuple[Text, LabelFromTurnsExtractor[TurnType, Any]]],
        turns: List[TurnType],
        training: bool = True,
    ) -> Dict[Text, Any]:
        outputs = {}
        for name, extractor in label_extractors:
            turns, extracted = extractor(turns=turns, training=training,)
            if extracted:
                outputs[name] = extracted
        return turns, outputs


class TurnFeaturizer(Generic[TurnType], Trainable, ABC):
    """Featurize a single input turn."""

    @abstractmethod
    def featurize(
        self,
        turn: TurnType,
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> Dict[Text, List[Features]]:
        pass

    @abstractmethod
    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        pass


class LabelFeaturizer(Generic[RawLabelType], Trainable):
    """Converts a label to `Features`."""

    def featurize(
        self, raw_label: RawLabelType, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        self.raise_if_not_trained()
        return self._featurize(raw_label=raw_label, interpreter=interpreter)

    @abstractmethod
    def _featurize(
        self, raw_label: RawLabelType, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        pass

    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._train(domain=domain, interpreter=interpreter)
        self._trained = True

    @abstractmethod
    def _train(self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]):
        pass


class LabelIndexer(Generic[TurnType, RawLabelType], Trainable):
    """Converts a label to an index."""

    def index(self, raw_label: Optional[RawLabelType],) -> np.ndarray:
        self.raise_if_not_trained()
        return self._index(raw_label=raw_label)

    @abstractmethod
    def _index(self, raw_label: Optional[RawLabelType]) -> np.ndarray:
        pass

    def train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        self._train(domain=domain, extractor=extractor)
        self._trained = True

    @abstractmethod
    def _train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        pass
