from .json_state import JSONState


class JSONDecoder:
    def __init__(self) -> None:
        self.state = JSONState.START

    def consume_char(self, char: str) -> bool:
        """
        Consume one character.

        Returns True if the character is valid for the current state.
        """
        # if self.state == JSONState.START:
        #     if char == "{":
        #         self.state = JSONState.OBJECT_START
        #         return True

        # elif self.state == JSONState.OBJECT_START:
        #     if char == '"':
        #         self.state = JSONState.KEY
        #         return True

        # elif self.state == JSONState.KEY:
        #     if char == '"':
        #         self.state = JSONState.COLON
        #         return True

        # elif self.state == JSONState.COLON:
        #     if char == ":":
        #         self.state = JSONState.VALUE
        #         return True

        # elif self.state == JSONState.VALUE:
        #     if char == '"':
        #         self.state = JSONState.OBJECT_END
        #         return True

        # elif self.state == JSONState.OBJECT_END:
        #     if char == "}":
        #         self.state = JSONState.START
        #         return True

        if self.state == JSONState.START:
            if char == "{":
                self.state = JSONState.EXPECT_KEY_START
                return True

        elif self.state == JSONState.EXPECT_KEY_START:
            if char == '"':
                self.state = JSONState.KEY_CONTENT
                return True

        elif self.state == JSONState.KEY_CONTENT:
            if char == '"':
                self.state = JSONState.EXPECT_COLON
                return True

        elif self.state == JSONState.EXPECT_COLON:
            if char == ":":
                self.state = JSONState.EXPECT_VALUE_START
                return True

        elif self.state == JSONState.EXPECT_VALUE_START:
            if char == '"':
                self.state = JSONState.VALUE_CONTENT
                return True

        elif self.state == JSONState.VALUE_CONTENT:
            if char == "}":
                self.state = JSONState.EXPECT_OBJECT_END
                return True
        return False

    def consume_token(self, token: str) -> bool:
        """
        Consume a complete tokenizer token.

        Returns True if the complete token is valid.
        """
        original_state = self.state

        for char in token:
            if not self.consume_char(char):
                self.state = original_state
                return False

        return True
