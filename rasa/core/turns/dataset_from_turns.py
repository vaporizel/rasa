from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Text,
    TypeVar,
    Generic,
    Tuple,
    Optional,
    Union,
)

import numpy as np

from rasa.core.turns.turn import Turn
from rasa.core.turns.turn_featurizer import (
    TurnFeatures,
    TurnFeaturizer,
)
from rasa.core.turns.turn_sequence import (
    TurnSubSequenceGenerator,
    TurnSequenceFilter,
    TurnSequenceModifier,
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features


TurnType = TypeVar("TurnType")
RawLabels = Dict[Text, Any]
FeaturizedLabels = Dict[Text, Union[List[Features], np.ndarray]]


class DatasetFromTurns(TurnSubSequenceGenerator[TurnType], Generic[TurnType]):
    """Generates labeled and modified turns from a sequence of turns."""

    def __init__(
        self,
        initial_filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
        ignore_duplicates: bool = False,
        postprocessing: Optional[List[TurnSequenceModifier[TurnType]]] = None,
        output_extractors: List[LabelsFromTurns[TurnType]] = None,
        final_filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
    ):
        super().__init__(
            initial_filters=initial_filters,
            ignore_duplicates=ignore_duplicates,
            postprocessing=postprocessing,
            final_filters=final_filters,
        )
        self._output_extractors = output_extractors

    def generate(
        self,
        turns: List[TurnType],
        interpreter: NaturalLanguageInterpreter,
        training: bool,
    ) -> Iterator[Tuple[List[Turn], Optional[RawLabels]]]:
        for processed_turns, _, outputs in self.generate_subsequence(
            turns, interpreter, training
        ):
            outputs = LabelsFromTurns.extract_all(
                self._output_extractors,
                turns=processed_turns,
                turns_featurized=None,
                interpreter=interpreter,
                training=training,
            )
            yield processed_turns, outputs


class FeaturizedDatasetFromTurns(TurnSubSequenceGenerator[TurnType], Generic[TurnType]):
    """Generates and featurizes labeled and modified turns from a sequence of turns."""

    def __init__(
        self,
        turn_featurizer: TurnFeaturizer[TurnType],
        output_extractors: List[FeaturizedLabelsFromTurns[TurnType]],
        initial_filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
        ignore_duplicates: bool = False,
        postprocessing: Optional[List[TurnSequenceModifier[TurnType]]] = None,
        final_filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
    ):
        super().__init__(
            turn_featurizer=turn_featurizer,
            initial_filters=initial_filters,
            ignore_duplicates=ignore_duplicates,
            postprocessing=postprocessing,
            final_filters=final_filters,
        )
        self._output_extractors = output_extractors
        self._turn_featurizer = turn_featurizer
        self._validate()

    def _validate(self):
        if self._turn_featurizer:
            turn_featurizer_class = self._turn_featurizer.__class__
            for output_extractor in self._output_extractors:
                if not output_extractor.compatible_with(turn_featurizer_class):
                    raise ValueError("...")

    def generate(
        self,
        turns: List[TurnType],
        interpreter: NaturalLanguageInterpreter,
        training: bool,
    ) -> Iterator[Tuple[List[TurnFeatures], Optional[FeaturizedLabels]]]:
        for processed_turns in self.generate_subsequence(turns, interpreter, training):
            outputs = LabelsFromTurns.extract_all(
                self._output_extractors,
                turns=processed_turns,
                interpreter=interpreter,
                training=training,
            )
            processed_turns_featurized = [
                self._turn_featurizer.featurize(turn, training=training)
                for turn in processed_turns
            ]

            yield processed_turns_featurized, outputs


class LabelsFromTurns(Generic[TurnType]):
    """Extracts label information from a sequence of turns."""

    @abstractmethod
    def extract(self, turns: List[TurnType], training: bool = True,) -> RawLabels:
        pass

    @abstractmethod
    def extract_all(self, domain: Domain) -> RawLabels:
        pass

    @classmethod
    def apply(
        cls,
        label_extractors: List[LabelsFromTurns[TurnType]],
        turns: List[TurnType],
        training: bool = True,
    ) -> RawLabels:

        outputs = {}
        for idx, extractor in enumerate(label_extractors):
            turns, extracted = extractor(turns=turns, training=training,)
            if extracted:
                if not set(outputs.keys()).isdisjoint(extracted.keys()):
                    raise RuntimeError(
                        f"The given extractors are not compatible. "
                        f"Received features for keys {sorted(extracted.keys())} from "
                        f"{extractor}, after already extracting "
                        f"{sorted(outputs.keys())} "
                        f"via {label_extractors[:idx]}."
                    )
                outputs.update(extracted)

        # TODO: skip keys with empty lists ... ?

        return turns, outputs


class FeaturizedLabelsFromTurns(Generic[TurnType]):
    """Extracts and featurizes label information from a sequence of turns."""

    @abstractmethod
    def extract_and_featurize(
        self,
        turns: List[TurnType],
        training: bool = True,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> FeaturizedLabels:
        pass

    @abstractmethod
    def featurize_all(self, domain: Domain,) -> Dict[Text, FeaturizedLabels]:
        return

    @classmethod
    def apply(
        cls,
        label_extractors: List[FeaturizedLabelsFromTurns],
        turns: List[TurnType],
        training: bool = True,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> FeaturizedLabels:

        outputs = {}
        for idx, extractor in enumerate(label_extractors):
            turns, extracted = extractor(
                turns=turns, training=training, interpreter=interpreter,
            )
            if extracted:
                if not set(outputs.keys()).isdisjoint(extracted.keys()):
                    raise RuntimeError(
                        f"The given extractors are not compatible. "
                        f"Received features for keys {sorted(extracted.keys())} from "
                        f"{extractor}, after already extracting "
                        f"{sorted(outputs.keys())} "
                        f"via {label_extractors[:idx]}."
                    )
                outputs.update(extracted)

        # TODO: skip keys with empty lists ... ?

        return turns, outputs
