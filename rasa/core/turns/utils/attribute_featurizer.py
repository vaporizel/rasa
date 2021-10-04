from rasa.shared.core.domain import SubState
from typing import Any, List, Optional, Dict, Set, Text, Text


from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message


class FeaturizerUsingInterpreter:
    @staticmethod
    def featurize(
        message_data: SubState,
        interpreter: NaturalLanguageInterpreter,
        exclude_from_results: Optional[Set[Text]] = None,
    ) -> Dict[Text, List[Features]]:
        """

        """
        exclude_from_results = exclude_from_results or {}

        # NOTE: we could exclude skipped attributes here already -- we will fix this
        # when moving to the 3.0 branch and reworking this part because we'll use the
        # lookup table here.
        message = Message(data=message_data)
        parsed_message = interpreter.featurize_message(message)

        if parsed_message is None:
            return {}

        attributes = set(
            attribute
            for attribute in message_data.keys()
            if attribute not in exclude_from_results
        )
        output = FeaturizerUsingInterpreter._get_features_from_parsed_message(
            parsed_message, attributes
        )
        return output

    @staticmethod
    def _get_features_from_parsed_message(
        message: Message, attributes: Set[Text]
    ) -> Dict[Text, List[Features]]:
        """Collects and combine all features for the specified attributes.

        Args:
            message: an arbitrary message
        """
        output = dict()
        for attribute in attributes:
            # Unfortunately, the following lists are not just empty if there are no
            # sparse/dense features but might include `None`s.
            sparse_features = message.get_sparse_features(attribute)
            dense_features = message.get_dense_features(attribute)
            # So we have to filter them:
            features = [f for f in sparse_features + dense_features if f is not None]
            output.setdefault(attribute, []).append(features)
        return output
