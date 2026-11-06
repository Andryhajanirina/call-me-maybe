#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_decoder.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:40:38 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/21 14:48:25 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from .json_state import JSONState


class JSONDecoder:
    def __init__(self) -> None:
        self.state = JSONState.START
        self.object_stack: list[str] = []
        self.escape_next = False
        self.unicode_escape_remaining = 0

    def consume_char(self, char: str) -> bool:
        """
        Consume one character.

        Returns True if the character is valid for the current state.
        """

        if self.state == JSONState.DONE:
            return self.is_whitespace(char)

        if self.state == JSONState.START:
            if self.is_whitespace(char):
                return True

            if char == "{":
                self.object_stack.append("object")
                self.state = JSONState.EXPECT_KEY_OR_END
                return True

        elif self.state == JSONState.EXPECT_KEY_OR_END:
            if char == '"':
                self.state = JSONState.KEY_CONTENT
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

        elif self.state == JSONState.EXPECT_KEY_START:
            if self.is_whitespace(char):
                return True

            if char == '"':
                self.state = JSONState.KEY_CONTENT
                return True

        elif self.state == JSONState.KEY_CONTENT:
            if self.unicode_escape_remaining > 0:
                if char not in "0123456789abcdefABCDEF":
                    return False
                self.unicode_escape_remaining -= 1
                return True

            if self.escape_next:
                self.escape_next = False
                if char == "u":
                    self.unicode_escape_remaining = 4
                    return True
                return char in '"\\/bfnrt'

            if char == "\\":
                self.escape_next = True
                return True

            if char == '"':
                self.state = JSONState.EXPECT_COLON
                return True
            return True

        elif self.state == JSONState.EXPECT_COLON:
            if self.is_whitespace(char):
                return True

            if char == ":":
                self.state = JSONState.EXPECT_VALUE_START
                return True

        elif self.state == JSONState.EXPECT_VALUE_START:
            if self.is_whitespace(char):
                return True

            if char == '"':
                self.state = JSONState.VALUE_CONTENT
                return True

            if char == "{":
                self.object_stack.append("object")
                self.state = JSONState.EXPECT_KEY_OR_END
                return True

            if char == "-":
                self.state = JSONState.EXPECT_NUMBER_DIGIT
                return True

            if char == "0":
                self.state = JSONState.EXPECT_ZERO_END
                return True

            if char.isdigit():
                self.state = JSONState.VALUE_NUMBER
                return True

        elif self.state == JSONState.VALUE_CONTENT:
            if self.unicode_escape_remaining > 0:
                if char not in "0123456789abcdefABCDEF":
                    return False
                self.unicode_escape_remaining -= 1
                return True

            if self.escape_next:
                self.escape_next = False
                if char == "u":
                    self.unicode_escape_remaining = 4
                    return True
                return char in '"\\/bfnrt'

            if char == "\\":
                self.escape_next = True
                return True

            if char == '"':
                self.state = JSONState.EXPECT_COMMA
                return True
            return True

        elif self.state == JSONState.EXPECT_NUMBER_DIGIT:
            if char == "0":
                self.state = JSONState.EXPECT_ZERO_END
                return True

            if char in "123456789":
                self.state = JSONState.VALUE_NUMBER
                return True

        elif self.state == JSONState.EXPECT_ZERO_END:
            if self.is_whitespace(char):
                self.state = JSONState.EXPECT_COMMA
                return True

            if char == ".":
                self.state = JSONState.EXPECT_FRACTION_DIGIT
                return True

            if char in "eE":
                self.state = JSONState.EXPECT_EXPONENT
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

            if char == ",":
                self.state = JSONState.EXPECT_KEY_START
                return True

        elif self.state == JSONState.VALUE_NUMBER:
            if char.isdigit():
                return True

            if char == ".":
                self.state = JSONState.EXPECT_FRACTION_DIGIT
                return True

            if char in "eE":
                self.state = JSONState.EXPECT_EXPONENT
                return True

            if self.is_whitespace(char):
                self.state = JSONState.EXPECT_COMMA
                return True

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

        elif self.state == JSONState.EXPECT_FRACTION_DIGIT:
            if char.isdigit():
                self.state = JSONState.VALUE_NUMBER
                return True

        elif self.state == JSONState.EXPECT_EXPONENT:
            if char in "+-":
                self.state = JSONState.EXPECT_EXPONENT_DIGIT
                return True

            if char.isdigit():
                self.state = JSONState.EXPECT_EXPONENT_NUMBER
                return True

        elif self.state == JSONState.EXPECT_EXPONENT_DIGIT:
            if char.isdigit():
                self.state = JSONState.EXPECT_EXPONENT_NUMBER
                return True

        elif self.state == JSONState.EXPECT_EXPONENT_NUMBER:
            if char.isdigit():
                return True

            if self.is_whitespace(char):
                self.state = JSONState.EXPECT_COMMA
                return True

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

        # ==================================================

        elif self.state == JSONState.EXPECT_COMMA:
            if self.is_whitespace(char):
                return True

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

    def consume_text(self, token: str) -> bool:
        """Consume text and restore the state if it is invalid."""
        original_state = self.state
        original_stack = self.object_stack.copy()
        original_escape_next = self.escape_next
        original_unicode_escape_remaining = (
            self.unicode_escape_remaining
        )

        for char in token:
            if not self.consume_char(char):
                self.state = original_state
                self.object_stack = original_stack
                self.escape_next = original_escape_next
                self.unicode_escape_remaining = (
                    original_unicode_escape_remaining
                )
                return False

        return True

    def consume_token(self, token: str) -> bool:
        """Consume a complete token, but roll back incomplete escapes."""
        original_state = self.state
        original_stack = self.object_stack.copy()
        original_escape_next = self.escape_next
        original_unicode_escape_remaining = (
            self.unicode_escape_remaining
        )

        if not self.consume_text(token):
            return False

        if self.escape_next or self.unicode_escape_remaining:
            self.state = original_state
            self.object_stack = original_stack
            self.escape_next = original_escape_next
            self.unicode_escape_remaining = (
                original_unicode_escape_remaining
            )

        return True

    def is_token_valid(self, token: str) -> bool:
        """Check a token without changing the decoder state."""
        original_state = self.state
        original_stack = self.object_stack.copy()
        original_escape_next = self.escape_next
        original_unicode_escape_remaining = (
            self.unicode_escape_remaining
        )

        try:
            return self.consume_text(token)
        finally:
            self.state = original_state
            self.object_stack = original_stack
            self.escape_next = original_escape_next
            self.unicode_escape_remaining = (
                original_unicode_escape_remaining
            )

    def is_whitespace(self, char: str) -> bool:
        return char in (" ", "\t", "\n", "\r")
