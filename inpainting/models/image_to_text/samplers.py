from torch.utils.data import (
    BatchSampler,
    Sampler,
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from typing import List, Iterator, Optional
import torch
from .schedulers import AlphaScheduler


class AlphaScheduleBatchSampler(BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        dataset: Dataset,
        alpha_scheduler: AlphaScheduler,
        num_processes: int,
        gradient_accumulation_steps: int,
        max_concepts: int = 5,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.alpha_scheduler = alpha_scheduler
        self.num_processes = num_processes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_concepts = max_concepts
        if self.max_concepts is None:
            self.max_concepts = 10
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if self.shuffle:
            self.sampler = RandomSampler(dataset, generator=generator)
        else:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            batch_idx = 0
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [
                        (
                            next(sampler_iter),
                            self.alpha_scheduler.get_alphas(size=self.max_concepts),
                        )
                        for _ in range(self.batch_size)
                    ]
                    yield batch
                    # Update the alpha scheduler every num_processes batches
                    batch_idx += 1
                    if (
                        batch_idx
                        % (self.num_processes * self.gradient_accumulation_steps)
                        == 0
                    ):
                        self.alpha_scheduler.increment_step()
                except StopIteration:
                    break
        else:
            batch_idx = 0
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                alphas = self.alpha_scheduler.get_alphas(size=self.max_concepts)
                batch[idx_in_batch] = (idx, alphas)
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    # Update the alpha scheduler every num_processes batches
                    batch_idx += 1
                    if (
                        batch_idx
                        % (self.num_processes * self.gradient_accumulation_steps)
                        == 0
                    ):
                        self.alpha_scheduler.increment_step()
                    # Reset the batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
                # Update the alpha scheduler every num_processes batches
                batch_idx += 1
                if (
                    batch_idx % (self.num_processes * self.gradient_accumulation_steps)
                    == 0
                ):
                    self.alpha_scheduler.increment_step()

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
