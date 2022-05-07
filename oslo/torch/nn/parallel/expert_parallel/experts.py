import math
from typing import Type
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed._seed.helper import seed

from .utils import get_current_device


