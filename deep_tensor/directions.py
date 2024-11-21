import enum


class Direction(enum.Enum):
    """The direction in which the cores are being iterated over."""

    FORWARD = 1
    BACKWARD = -1


REVERSE_DIRECTIONS = {
    Direction.FORWARD: Direction.BACKWARD,
    Direction.BACKWARD: Direction.FORWARD
}