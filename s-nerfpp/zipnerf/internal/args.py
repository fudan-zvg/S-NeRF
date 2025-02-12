import gin

from internal import math
gin.config.external_configurable(math.safe_exp, module='math')


@gin.configurable
class Args:
    RENDER_N: int = 10
    only_side_cam: bool = False
    only_front_cam: bool = False
