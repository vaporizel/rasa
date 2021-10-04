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

from numpy.lib.function_base import copy

from rasa.core.turns.turn import Actor, Turn


TurnType = TypeVar("TurnType")


class TurnSubSequenceGenerator(Generic[TurnType], ABC):
    """Generates sub-sequencesfrom a sequence of turns."""

    def __init__(
        self,
        initial_filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
        ignore_duplicates: bool = False,
        postprocessing: Optional[List[TurnSequenceModifier[TurnType]]] = None,
        final_filters: Optional[List[TurnSequenceFilter[TurnType]]] = None,
        limit: Optional[int] = None,
    ) -> None:
        """
        Args:
            filters: only applied during training
            ...
        """
        self._initial_filters = initial_filters or []
        self._ignore_duplicates = ignore_duplicates
        self._cache: Set[Tuple[int, ...]] = set()
        self._postprocessing = postprocessing
        self._limit = limit
        self._final_filters = final_filters or []

    def generate_subsequence(
        self, turns: List[TurnType], training: bool
    ) -> Iterator[List[Turn]]:
        """

        During training, the whole sequence of turns is processed.

        During validation, we extract multiple (or no) sub-sequences of turns from the
        given sequence of turns: Each subsequence of turns from the 0-th to the i-th
        turn that passes the given sequence filters and is not a duplicate of any
        other subsequence created so far.

        In both cases, the same postprocessing is applied.

        Args:
            training: Indicates whether we are in training mode.

        """
        steps = [len(turns)] if not self._training else range(1, len(turns))
        num_generated = 0

        for idx in steps:

            if self._limit and num_generated >= self._limit:
                return

            # we'll make a copy of this subsequence, once we know we continue with it
            next_sequence = turns[:idx]

            # during training - skip if it does not pass filters
            if training and (
                not all(_filter(next_sequence) for _filter in self._initial_filters)
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
            if self._postprocessing:
                (next_sequence,) = TurnSequenceModifier.apply_all(
                    self._postprocessing, next_sequence,
                )

            # check if filters still pass (we modified the sequence)
            if training and (
                not all(_filter(next_sequence) for _filter in self._final_filters)
            ):
                continue

            num_generated += 1
            yield turns


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


@dataclass
class KeepMaxHistory(TurnSequenceModifier[TurnType], Generic[TurnType]):

    max_history: Optional[int] = None

    def __call__(self, turns: List[TurnType], training: bool,) -> List[TurnType]:
        if self.max_history:
            turns = turns[-self.max_history :]
        return turns
