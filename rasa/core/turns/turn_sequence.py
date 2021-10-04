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
    Set,
)
from dataclasses import dataclass

from rasa.core.turns.turn import Actor
from rasa.core.turns.turn_featurizer import (
    OutputExtractor,
    TurnFeatures,
    TurnFeaturizer,
)
from rasa.shared.nlu.training_data.features import Features


TurnType = TypeVar("TurnType")


class TurnSequencePipeline(Generic[TurnType]):
    """Generates (sub) sequence(s) from a sequence of turns."""

    def __init__(
        self,
        filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
        ignore_duplicates: bool = False,
        postprocessing: Optional[List[TurnSequenceModifier[TurnType]]] = None,
    ) -> None:
        self._pipeline = TurnSequenceFeaturizationPipeline(
            filter=filters,
            ignore_duplicates=ignore_duplicates,
            postprocessing=postprocessing,
            turn_featurizer=None,
            output_extractors=None,
        )

    def iterator(
        self, turns: List[TurnType], training: bool
    ) -> Iterator[List[TurnType]]:
        """
        During training, the whole sequence of turns is processed and returned.
        During validation, we extract multiple (or no) sub-sequences of turns from the
        given sequence of turns: Each subsequence of turns from the 0-th to the i-th
        turn that passes the given sequence filters and is not a duplicate of any
        other subsequence created so far.

        In both cases, the same postprocessing is applied.
        """
        for sequence, _ in self._pipeline.iterator(turns, training):
            yield sequence

    # ???
    def extract_output(self, turns: List[TurnType]) -> Dict[Text, Features]:
        pass


class TurnSequenceFeaturizationPipeline(Generic[TurnType]):
    """Generates featurized (sub) sequence(s) from a sequence of turns."""

    def __init__(
        self,
        filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
        ignore_duplicates: bool = False,
        postprocessing: Optional[List[TurnSequenceModifier[TurnType]]] = None,
        turn_featurizer: TurnFeaturizer[TurnType] = None,
        output_extractors: List[OutputExtractor[TurnType]] = None,
    ) -> None:
        """
        Args:

            filters: only applied during training

        """
        self._filters = filters
        self._ignore_duplicates = ignore_duplicates
        self._cache: Set[Tuple[int, ...]] = set()
        self._postprocessing = postprocessing
        self._turn_featurizer = turn_featurizer
        self._output_extractors = output_extractors or []

        if self._turn_featurizer:
            turn_featurizer_class = self._turn_featurizer.__class__
            for output_extractor in self._output_extractors:
                if not output_extractor.compatible_with(turn_featurizer_class):
                    raise ValueError("...")

    def clear_cache(self):
        self._cache = set()

    def iterator(
        self, turns: List[TurnType], training: bool
    ) -> Iterator[List[TurnType], List[TurnFeatures]]:
        """

        During training, the whole sequence of turns is processed and returned.
        During validation, we extract multiple (or no) sub-sequences of turns from the
        given sequence of turns: Each subsequence of turns from the 0-th to the i-th
        turn that passes the given sequence filters and is not a duplicate of any
        other subsequence created so far.

        In both cases, the same postprocessing is applied.

        Args:
            training: Indicates whether we are in training mode.

        """
        steps = [len(turns)] if not self._training else range(1, len(turns))

        turns_featurized = (
            [self._turn_featurizer.featurize(turn, training=training) for turn in turns]
            if self._turn_featurizer
            else None
        )

        for idx in steps:

            next_sequence = turns[:idx]

            next_sequence_featurized = (
                turns_featurized[:idx] if self._turn_featurizer else None
            )

            # during training - skip if it does not pass filters
            if training and (
                self._filter
                and not all(_filter(next_sequence) for _filter in self._filters)
            ):
                continue

            # during training - skip if it is a duplicate
            if training and self._ignore_duplicates:
                id = tuple(hash(turn) for turn in next_sequence)
                if id in self._cache:
                    continue
                else:
                    self._cache.add(id)

            # always apply postprocessing
            if self._postprocessing:
                (
                    next_sequence,
                    next_sequence_featurized,
                ) = TurnSequenceModifier.apply_all(
                    self._postprocessing, next_sequence, next_sequence_featurized
                )

            # FIXME: return typed dict instead of tuples...

            outputs = OutputExtractor.extract_all(
                self._output_extractors,
                turns=turns,
                turns_as_inputs=turns_as_input,
                interpreter=interpreter,
                training=training,
            )

            yield next_sequence, next_sequence_featurized


class TurnSequenceFilter(Generic[TurnType]):
    def __call__(self, turns: List[TurnType]) -> bool:
        pass


class EndsWithBotTurn(TurnSequenceFilter[TurnType]):
    def __call__(self, turns: List[TurnType]) -> bool:
        return turns[-1].actor == Actor.BOT


@dataclass
class HasMinLength(TurnSequenceFilter[TurnType]):
    min_length: int

    def __call__(self, turns: List[TurnType]) -> bool:
        return len(turns) >= self.min_length


class TurnSequenceModifier(Generic[TurnType], ABC):
    @abstractmethod
    def __call__(
        self,
        turns: List[TurnType],
        turns_as_input: Optional[List[TurnFeatures]],
        training: bool,
        **kwargs: Any,
    ) -> Tuple[List[TurnType], Optional[List[TurnFeatures]]]:
        pass

    @staticmethod
    def apply_all(
        modifiers: List[TurnSequenceModifier[TurnType]],
        turns: List[TurnType],
        turns_as_input: Optional[List[TurnFeatures]],
    ) -> Tuple[List[TurnType], Optional[List[TurnFeatures]]]:
        for modifier in modifiers:
            turn, turns_as_input = modifier(turns, turns_as_input)
        return turns, turns_as_input


@dataclass
class KeepMaxHistory(TurnSequenceModifier[TurnType], Generic[TurnType]):

    max_history: Optional[int] = None

    def __call__(
        self,
        turns: List[TurnType],
        turns_as_input: Optional[List[TurnFeatures]],
        training: bool,
        **kwargs: Any,
    ) -> Tuple[List[TurnType], Optional[List[TurnFeatures]]]:
        if self.max_history:
            last_max_history_steps = slice(-self.max_history, None)
            turns = turns[last_max_history_steps]
            if turns_as_input:
                turns_as_input = turns_as_input[last_max_history_steps]
        return turns, turns_as_input
