from typing import Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, Shaped, PRNGKeyArray

Arr = Union[np.ndarray, Array]

AnyShaped = Shaped[Arr, "*"]
AnyFloat = Float[Arr, "*"]
Shape = Union[tuple[int, ...], list[int]]

FloatScalar = float | Float[Arr, ""]
IntScalar = int | Int[Arr, ""]
BoolScalar = bool | Bool[Arr, ""]

BFloat = Float[Arr, "b"]
BInt = Int[Arr, "b"]
BBool = Bool[Arr, "b"]

Vec2 = Float[Arr, "2"]
Vec3 = Float[Arr, "3"]
Vec4 = Float[Arr, "4"]

RotMat2D = Float[Arr, "2 2"]
RotMat3D = Float[Arr, "3 3"]

FloatDict = dict[str, Union[FloatScalar, "FloatDict"]]

BBFloat = Float[Arr, "b1 b2"]
