# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


class BaseQualityEvaluator(ABC, Generic[T]):
    """Base interface for all quality evaluators."""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the evaluator. Returns True if successful."""
        pass

    @abstractmethod
    async def evaluate(self, *args, **kwargs) -> T:
        """Evaluate quality and return results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if evaluator is available for use."""
        pass
