from abc import ABC
from typing import Text

import rasa.shared.utils.io


class Trainable(ABC):
    def __init__(self) -> None:
        self._trained = False

    def raise_if_not_trained(self, message: Text = "") -> None:
        if not self._trained:
            raise RuntimeError(
                f"Expected this {self.__class__.__name__} to be trained. " f"{message}"
            )

    def warn_if_not_trained(self, message: Text = "") -> None:
        if not self._trained:
            rasa.shared.utils.io.raise_warning(
                f"Expected this {self.__class__.__name__} to be trained. " f"{message}"
            )
