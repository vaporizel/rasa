from typing import Generic, Optional, Text, List, Any, Dict, Tuple

import numpy as np

from rasa.core.turns.to_dataset.dataset import (
    LabelFeaturizer,
    LabelFromTurnsExtractor,
    LabelIndexer,
    RawLabelType,
    TurnType,
)
from rasa.core.turns.to_dataset.utils.trainable import Trainable
from rasa.core.turns.to_dataset.utils.entity_tags_encoder import EntityTagsEncoder
from rasa.core.turns.to_dataset.utils.feature_lookup import FeatureLookup
from rasa.core.turns.to_dataset.utils.multihot_encoder import MultiHotEncoder
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message


class LabelFeaturizerViaLookup(LabelFeaturizer[Any]):
    """Featurizes a label via the lookup table."""

    def __init__(self, attribute: Text) -> None:
        self.attribute = attribute

    def __call__(
        self, raw_label: Any, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        return FeatureLookup.lookup_features(
            message_data={self.attribute: raw_label}, interpreter=interpreter
        )

    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        pass


class ApplyEntityTagsEncoder(LabelFeaturizer[Tuple[Text, Dict[Text, Any]]], Trainable):
    """Featurizes entities via an entity tags encoder."""

    def __init__(self, bilou_tagging: bool = False) -> None:
        super().__init__()
        self._bilou_tagging = bilou_tagging

    def train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._entity_tags_encoder = EntityTagsEncoder(
            domain=domain, bilou_tagging=self._bilou_tagging,
        )
        self._trained = True

    def __call__(
        self,
        text_with_entities: Tuple[Text, Dict[Text, Any]],
        interpreter: NaturalLanguageInterpreter,
    ) -> List[Features]:
        self.raise_if_not_trained()
        if interpreter is None:
            raise RuntimeError("Needs interpreter")

        if not isinstance(text_with_entities, Tuple) or len(text_with_entities) != 2:
            # NOTE: typing should help identify issues with this
            raise RuntimeError(
                "Expected text and entities. Use a label extractor that returns "
                "Text and entities"
            )

        text, entities = text_with_entities

        # Don't bother encoding anyting if there are less than 2 entity tags,
        # because we won't train any entity extactor anyway.
        if not entities or self._entity_tags_encoder.entity_tag_spec.num_tags < 2:
            return {}

        # # FIXME: move this to postprocessing ->> if TEXT is removed then we also
        # remove ENTITIES from the output.
        # # train stories support both text and intent,
        # # but if intent is present, the text is ignored
        # if INTENT in turn.state[USER]:
        #     return {}

        message = interpreter.featurize_message(Message(data={TEXT: text}))
        text_tokens = message.get(TOKENS_NAMES[TEXT])
        return self._entity_tags_encoder.encode(
            text_tokens=text_tokens, entities=entities
        )


class IndexerFromLabelExtractor(
    LabelIndexer[TurnType, RawLabelType], Generic[TurnType, RawLabelType], Trainable,
):
    """Converts """

    def train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        self._extractor = extractor
        self._multihot_encoder = MultiHotEncoder(
            dimension_names=self._extractor.from_domain(domain=domain)
        )
        self._trained = True

    def __call__(self, raw_label: RawLabelType,) -> np.ndarray:
        self.raise_if_not_trained()
        return self._multihot_encoder.encode_as_index_array([raw_label])
