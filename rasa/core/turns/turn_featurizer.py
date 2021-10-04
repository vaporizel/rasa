from __future__ import annotations
from abc import abstractmethod, ABC
from typing import (
    List,
    TypeVar,
    Generic,
    Dict,
    Text,
    Text,
)


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
