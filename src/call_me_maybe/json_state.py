from enum import Enum, auto


class JSONState(Enum):
    START = auto()

    EXPECT_KEY_START = auto()
    KEY_CONTENT = auto()
    EXPECT_COLON = auto()

    EXPECT_VALUE_START = auto()
    VALUE_CONTENT = auto()
    EXPECT_OBJECT_END = auto()

    DONE = auto()
