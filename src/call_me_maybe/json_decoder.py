from .json_state import JSONState


class JSONDecoder:
    def __init__(self) -> None:
        self.state = JSONState.START
        self.object_stack: list[str] = []

    def consume_char(self, char: str) -> bool:
        """
        Consume one character.

        Returns True if the character is valid for the current state.
        """

        if self.state == JSONState.START:
            if char == "{":
                self.object_stack.append("object")
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
            return True

        elif self.state == JSONState.EXPECT_COLON:
            if char == ":":
                self.state = JSONState.EXPECT_VALUE_START
                return True

        elif self.state == JSONState.EXPECT_VALUE_START:
            if char == '"':
                self.state = JSONState.VALUE_CONTENT
                return True

            if char == "{":
                self.object_stack.append("object")
                self.state = JSONState.EXPECT_KEY_START
                return True

        elif self.state == JSONState.VALUE_CONTENT:
            if char == '"':
                self.state = JSONState.EXPECT_COMMA
                return True
            return True

        elif self.state == JSONState.EXPECT_COMMA:
            if char == ",":
                self.state = JSONState.EXPECT_KEY_START
                return True

            if char == "}":
                if not self.object_stack:
                    return False

                self.object_stack.pop()

                if not self.object_stack:
                    self.state = JSONState.DONE
                else:
                    self.state = JSONState.EXPECT_COMMA
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
