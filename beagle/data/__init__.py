"""The dataloader (design-plot "dataloader" box).

Public surface::

    import beagle as bgl
    train_ds, val_ds = bgl.DataMixture.from_config(data_config).split()

* :class:`TaskDataset` — an indexable ``(Task, TaskContext)`` collection.
* :class:`DataMixture` — mix tasks across benchmarks with weights + train/val split.
"""

from __future__ import annotations

from beagle.data.dataset import TaskDataset, TaskItem
from beagle.data.mixture import DataMixture, MixtureComponent
from beagle.data.sampler import ConcatSampler, TaskSampler, WeightedRoundRobinSampler

__all__ = [
    "TaskDataset",
    "TaskItem",
    "DataMixture",
    "MixtureComponent",
    "TaskSampler",
    "ConcatSampler",
    "WeightedRoundRobinSampler",
]
