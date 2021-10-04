from __future__ import annotations
from abc import abstractmethod, ABC
from typing import (
    List,
    Type,
    TypeVar,
    Generic,
    Dict,
    Text,
    Union,
    Text,
    Tuple,
)

import numpy as np

from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features
import rasa.shared.utils.io

TurnType = TypeVar("TurnType")


TurnFeatures = Dict[Text, List[Features]]


class TurnFeaturizer(Generic[TurnType], ABC):
    """Featurize a single turn.

    Note that this isn't the only place where we featurize....
    If you need for output a different featurization for the same attribute - or
    if you only need some feature for output turns but turns are much less often
    used as output (than as input) - then use the output extractor.
    """

    @abstractmethod
    def featurize(
        self,
        turn: TurnType,
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> TurnFeatures:
        pass


class OutputExtractor(Generic[TurnType], ABC):
    """...


    FIXME: training and prediction mode... -> prediction mode makes sure that
    no predictable things are left in from training data.
    """

    def compatible_with(self, turn_featurizer_type: Type[TurnFeaturizer]) -> bool:
        True

    @abstractmethod
    def extract(
        self,
        turns: List[TurnType],
        turns_as_inputs: List[Dict[Text, Features]],
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ) -> Tuple[List[Dict[Text, Features]], Union]:
        pass

    @abstractmethod
    def encode_all(self, domain: Domain) -> Union[List[np.ndarray], List[Features]]:
        pass

    @classmethod
    def extract_all(
        cls,
        extractors: List[OutputExtractor],
        turns: List[TurnType],
        turns_as_inputs: List[Dict[Text, Features]],
        interpreter: NaturalLanguageInterpreter,
        training: bool = True,
    ):
        outputs = {}
        for idx, extractor in enumerate(extractors):
            turns_as_inputs, extracted = extractor(
                turns=turns,
                turns_as_inputs=turns_as_inputs,
                interpreter=interpreter,
                training=training,
            )
            if extracted:
                if not set(outputs.keys()).isdisjoint(extracted.keys()):
                    raise RuntimeError(
                        f"The given extractors are not compatible. "
                        f"Received features for keys {sorted(extracted.keys())} from "
                        f"{extractor}, after already extracting "
                        f"{sorted(outputs.keys())} "
                        f"via {extractors[:idx]}."
                    )
                outputs.update(extracted)

        # TODO: skip keys with empty lists ... ?

        return turns_as_inputs, outputs


class TrainableOutputExtractor(Generic[TurnType], ABC):
    def __init__(self):
        self._trained = False

    def raise_if_not_trained(self, message: Text = ""):
        if not self._trained:
            raise RuntimeError(
                f"Expected this {self.__class__.__name__} to be trained. " f"{message}"
            )

    def warn_if_not_trained(self, message: Text = ""):
        if not self._trained:
            rasa.shared.utils.io.raise_warning(
                f"Expected this {self.__class__.__name__} to be trained. " f"{message}"
            )


class InputPostProcessor(ABC):
    def __call__(
        self, turns_as_inputs: List[Dict[Text, Features]], training: bool
    ) -> List[Dict[Text, Features]]:
        pass


class DropLastTurn(InputPostProcessor):
    def __call__(self, turns_as_inputs: List[Dict[Text, Features]], training: bool):
        if training:
            turns_as_inputs = turns_as_inputs[:-1]
        return turns_as_inputs
