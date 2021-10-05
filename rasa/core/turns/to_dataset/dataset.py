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
import copy

import numpy as np

from rasa.core.turns.turn import Turn
from rasa.core.turns.to_dataset.utils.trainable import Trainable
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features


TurnType = TypeVar("TurnType")
RawLabelType = TypeVar("RawLabelType")
FeatureCollection = Dict[Text, List[Features]]


class TurnSubSequenceGenerator(Generic[TurnType], ABC):
    """Generates sub-sequencesfrom a sequence of turns."""

    def __init__(
        self,
        initial_validations: Optional[List[TurnSequenceValidation[TurnType]]] = None,
        ignore_duplicates: bool = False,
        modifiers: Optional[List[TurnSequenceModifier[TurnType]]] = None,
        final_validation: Optional[List[TurnSequenceValidation[TurnType]]] = None,
    ) -> None:
        """
        Args:
            filters: only applied during training
            ...
        """
        self._initial_validations = [f for f in initial_validations if f] or []
        self._ignore_duplicates = ignore_duplicates
        self._cache: Set[Tuple[int, ...]] = set()
        self._modifiers = [p for p in modifiers if p] or []
        self._final_validations = [f for f in final_validation if f] or []

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
        steps = [len(turns)] if not training else range(1, len(turns))
        num_generated = 0

        for idx in steps:

            if limit and num_generated >= limit:
                return

            # we'll make a copy of this subsequence, once we know we continue with it
            next_sequence = turns[:idx]

            # during training - skip if it does not pass filters
            if training and (
                not all(_filter(next_sequence) for _filter in self._initial_validations)
            ):
                continue

            # during training - skip if it is a duplicate
            if training and self._ignore_duplicates:
                id = tuple(hash(turn) for turn in next_sequence)
                if id in self._cache:
                    continue
                else:
                    self._cache.add(id)

            # make a deep copy to allow all modifiers and extractors to work in-place
            next_sequence = copy.deepcopy(next_sequence)

            # always apply postprocessing
            if self._modifiers:
                next_sequence = TurnSequenceModifier.apply_all(
                    self._modifiers, next_sequence, training=training,
                )

            # check if filters still pass (we modified the sequence)
            if training and (
                not all(_filter(next_sequence) for _filter in self._final_validations)
            ):
                continue

            num_generated += 1
            yield next_sequence


class TurnSequenceValidation(Generic[TurnType]):
    """Filters a given list of turns."""

    def __call__(self, turns: List[TurnType]) -> bool:
        pass


class TurnSequenceModifier(Generic[TurnType], ABC):
    """Returns a modified list of turns.

    Must not modify the given list of turns.
    """

    @abstractmethod
    def __call__(self, turns: List[TurnType], training: bool) -> List[TurnType]:
        pass

    @staticmethod
    def apply_all(
        modifiers: List[TurnSequenceModifier[TurnType]],
        turns: List[TurnType],
        training: bool,
    ) -> List[TurnType]:
        for modifier in modifiers:
            turns = modifier(turns, training=training)
        return turns


class DatasetFromTurns(TurnSubSequenceGenerator[TurnType], Generic[TurnType]):
    """Generates labeled and modified subsequences of turns."""

    def __init__(
        self,
        label_extractors: Dict[Text, LabelFromTurnsExtractor[TurnType, Any]],
        initial_validations: Optional[
            List[Optional[TurnSequenceValidation[TurnType]]]
        ] = None,
        ignore_duplicates: bool = False,
        modifiers: Optional[List[Optional[TurnSequenceModifier[TurnType]]]] = None,
        final_validations: Optional[
            List[Optional[TurnSequenceValidation[TurnType]]]
        ] = None,
    ):
        super().__init__(
            initial_validations=initial_validations,
            ignore_duplicates=ignore_duplicates,
            modifiers=modifiers,
            final_validation=final_validations,
        )
        self._label_extractors = label_extractors

    def generate(
        self, turns: List[TurnType], training: bool,
    ) -> Iterator[Tuple[List[Turn], Optional[Dict[Text, Any]]]]:
        for processed_turns in self.generate_subsequence(turns, training=training):
            processed_turns, outputs = LabelFromTurnsExtractor.apply_all(
                self._label_extractors, turns=processed_turns, training=training,
            )
            yield processed_turns, outputs


class FeaturizedDatasetFromTurns(
    TurnSubSequenceGenerator[TurnType], Generic[TurnType], Trainable
):
    """Generates and encodes labeled and modified subsequences of turns."""

    # FIXME: if we run predict on training data for evaluation, then we'll leak
    # information

    def __init__(
        self,
        turn_featurizer: TurnFeaturizer[TurnType],
        label_handling: List[Tuple[Text, LabelFeaturizationPipeline[TurnType, Any]]],
        initial_validations: Optional[
            List[Optional[TurnSequenceValidation[TurnType]]]
        ] = None,
        ignore_duplicates: bool = False,
        modifiers: Optional[List[Optional[TurnSequenceModifier[TurnType]]]] = None,
        final_validations: Optional[
            List[Optional[TurnSequenceValidation[TurnType]]]
        ] = None,
    ):
        super().__init__(
            initial_validations=initial_validations,
            ignore_duplicates=ignore_duplicates,
            modifiers=modifiers,
            final_validation=final_validations,
        )
        self._turn_featurizer = turn_featurizer
        self._label_handling = label_handling

    def train(
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
        for processed_turns in self.generate_subsequence(turns, training=training):

            # extract and featurize labels
            # Note that we run the output extraction when `training==False` as well
            # because the extractors will take care of removing all labels from the
            # training data.
            collected_features = {}
            collected_indices = {}
            if training:
                for label_name, label_handling in self._label_handling:
                    processed_turns, next_features, next_indices = label_handling(
                        processed_turns, training=training, interpreter=interpreter
                    )
                    if next_features:
                        collected_features[label_name] = next_features
                    if next_indices:
                        collected_indices[label_name] = next_indices

            # featurize the remaining input
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
                features = self.featurizer(extracted_label, interpreter=interpreter)
            if self.indexer and extracted_label:
                indices = self.indexer(extracted_label)
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
    """Featurizes a label."""

    @abstractmethod
    def __call__(
        self, raw_label: RawLabelType, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        pass

    @abstractmethod
    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        pass


class LabelIndexer(Generic[TurnType, RawLabelType], Trainable):
    """Converts a label to an index."""

    @abstractmethod
    def __call__(self, raw_label: Optional[RawLabelType],) -> np.ndarray:
        pass

    @abstractmethod
    def train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        pass
