import torch
from abc import ABC, abstractmethod

from typing import Optional


class AlphaScheduler(ABC):
    """Abstract class to handle alpha scheduling."""

    @abstractmethod
    def set_num_steps(self, num_steps: int):
        """Set the number of steps for the scheduler.

        Args:
            num_steps (int): The number of steps to set the scheduler to.
        """
        pass

    @abstractmethod
    def set_step(self, step: int):
        """Set the current step of the scheduler.

        Args:
            step (int): The step to set the scheduler to.
        """
        pass

    @abstractmethod
    def increment_step(self):
        """Increment the current step of the scheduler."""
        pass

    @abstractmethod
    def get_alphas(self, size: int):
        """Get a batch of alpha values.

        Args:
            size (int): The number of alpha values to generate.
        """
        pass


class ConstantAlphaScheduler(AlphaScheduler):
    """Class to handle constant alpha scheduling."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def set_num_steps(self, num_steps: int):
        pass

    def set_step(self, step: int):
        pass

    def increment_step(self):
        pass

    def get_alphas(self, size: int):
        """Get a batch of alpha values.

        Args:
            size (int): The number of alpha values to generate.
        """
        return torch.full((size,), self.alpha)


class RandomAlphaScheduler(AlphaScheduler):
    """Class to handle random alpha scheduling."""

    def __init__(self, start_alpha: float, end_alpha: float):
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha

    def set_num_steps(self, num_steps: int):
        pass

    def set_step(self, step: int):
        pass

    def increment_step(self):
        pass

    def get_alphas(self, size: int):
        """Get a batch of alpha values.

        Args:
            size (int): The number of alpha values to generate.
        """
        alphas = (
            torch.rand(size) * (self.end_alpha - self.start_alpha) + self.start_alpha
        )
        return alphas


class CurriculumAlphaScheduler(AlphaScheduler):
    """Main class to handle curriculum learning via alpha scheduling."""

    def __init__(
        self,
        start_mean_alpha: float,
        end_mean_alpha: float,
        std_dev_alpha: float,
        num_steps: Optional[int] = None,
    ):
        self.start_mean_alpha = start_mean_alpha
        self.mean_alpha = start_mean_alpha
        self.end_mean_alpha = end_mean_alpha
        self.std_dev_alpha = std_dev_alpha
        self.num_steps = num_steps
        
        self.constant_alpha = None

        self.step = 0

    def set_num_steps(self, num_steps: int):
        """Set the number of steps for the scheduler.

        Args:
            num_steps (int): The number of steps to set the scheduler to.
        """
        self.num_steps = num_steps
        self.step = 0
        self._update_alpha()

    def set_step(self, step: int):
        """Set the current step of the scheduler.

        Args:
            step (int): The step to set the scheduler to.
        """
        self.step = step
        if self.step >= self.num_steps:
            self.constant_alpha = self.end_mean_alpha
        self._update_alpha()

    def increment_step(self):
        """Increment the current step of the scheduler."""
        self.step += 1
        if self.step >= self.num_steps:
            self.constant_alpha = self.end_mean_alpha
        self._update_alpha()

    def _update_alpha(self):
        """Update the alpha value based on the current step."""
        assert self.num_steps is not None, "Number of steps is not set."
        self.mean_alpha = self.start_mean_alpha + (
            self.end_mean_alpha - self.start_mean_alpha
        ) * (self.step / (self.num_steps - 1))

    def get_alphas(self, size: int):
        """Get a batch of alpha values.

        Args:
            size (int): The number of alpha values to generate.
        """
        if self.constant_alpha is not None:
            return torch.full((size,), self.constant_alpha)
        # compute alpha using torch.normal
        alphas = torch.normal(self.mean_alpha, self.std_dev_alpha, size=(size,))
        # clip alphas to be in the range [start_mean_alpha, end_mean_alpha]
        alphas = torch.clamp(alphas, self.start_mean_alpha, self.end_mean_alpha)
        return alphas
