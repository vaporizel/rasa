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
from rasa.shared.nlu.constants import ENTITY_TAGS, TEXT
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message


class LabelFeaturizerViaLookup(LabelFeaturizer[Any]):
    """Featurizes a label via the lookup table."""

    def __init__(self, attribute: Text) -> None:
        self.attribute = attribute

    def _train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        pass

    def _featurize(
        self, raw_label: Any, interpreter: NaturalLanguageInterpreter
    ) -> List[Features]:
        if raw_label is None:
            return []
        return FeatureLookup.lookup_features(
            message_data={self.attribute: raw_label}, interpreter=interpreter
        )[self.attribute]


class ApplyEntityTagsEncoder(LabelFeaturizer[Tuple[Text, Dict[Text, Any]]]):
    """Featurizes entities via an entity tags encoder."""

    def __init__(self, bilou_tagging: bool = False) -> None:
        super().__init__()
        self._bilou_tagging = bilou_tagging

    def _train(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self._entity_tags_encoder = EntityTagsEncoder(
            domain=domain, bilou_tagging=self._bilou_tagging,
        )

    def _featurize(
        self,
        raw_label: Tuple[Text, Dict[Text, Any]],
        interpreter: NaturalLanguageInterpreter,
    ) -> List[Features]:
        if interpreter is None:
            raise RuntimeError("Needs interpreter")

        if not isinstance(raw_label, Tuple) or len(raw_label) != 2:
            # NOTE: typing should help identify issues with this
            raise RuntimeError(
                "Expected text and entities. Use a label extractor that returns "
                "Text and entities"
            )

        text, entities = raw_label

        # Don't bother encoding anyting if there are less than 2 entity tags,
        # because we won't train any entity extactor anyway.
        if (
            not text
            or not entities
            or self._entity_tags_encoder.entity_tag_spec.num_tags < 2
        ):
            return []

        message = interpreter.featurize_message(Message(data={TEXT: text}))
        text_tokens = message.get(TOKENS_NAMES[TEXT])
        return self._entity_tags_encoder.encode(
            text_tokens=text_tokens, entities=entities
        )


class IndexerFromLabelExtractor(
    LabelIndexer[TurnType, RawLabelType], Generic[TurnType, RawLabelType], Trainable,
):
    """Converts a label to an index.

    This indexer must be trained and uses a given `LabelFromTurnsExtractor` to
    retrieve all possible labels from the given domain. These labels are then used
    to create a label to index mapping.
    """

    def _train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        all_labels = extractor.from_domain(domain=domain)
        self._multihot_encoder = MultiHotEncoder(dimension_names=all_labels)

    def _index(self, raw_label: RawLabelType,) -> np.ndarray:
        return self._multihot_encoder.encode_as_index_array([raw_label])
