#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_decoder.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:40:38 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/21 14:28:21 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from .json_state import JSONState


class JSONDecoder:
    def __init__(self) -> None:
        self.state = JSONState.START
        self.object_stack: list[str] = []

    def my_isdigit(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def consume_char(self, char: str) -> bool:
        """
        Consume one character.

        Returns True if the character is valid for the current state.
        """

        if self.state == JSONState.DONE:
            return False

        # if self.is_whitespace(char):
        #     if self.state == JSONState.VALUE_NUMBER:
        #         return False
        #     return True

        if self.state == JSONState.START:
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
            # if char == "}":
            #     if not self.object_stack:
            #         return False

            #     self.object_stack.pop()

            #     if not self.object_stack:
            #         self.state = JSONState.DONE
            #     else:
            #         self.state = JSONState.EXPECT_COMMA

            #     return True

        elif self.state == JSONState.KEY_CONTENT:
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
            if char == '"':
                self.state = JSONState.EXPECT_COMMA
                return True
            return True

        elif self.state == JSONState.EXPECT_NUMBER_DIGIT:
            if char.isdigit():
                self.state = JSONState.VALUE_NUMBER
                return True

        elif self.state == JSONState.EXPECT_ZERO_END:
            if char == ".":
                self.state = JSONState.EXPECT_FRACTION_DIGIT
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

    def get_allowed_number_characters(self) -> set[str]:
        if self.state == JSONState.EXPECT_VALUE_START:
            return {
                "-",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            }

        if self.state == JSONState.EXPECT_NUMBER_DIGIT:
            return {
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            }

        if self.state == JSONState.EXPECT_ZERO_END:
            return {
                ".",
                ",",
                "}",
            }

        if self.state == JSONState.VALUE_NUMBER:
            return {
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                ".",
                ",",
                "}",
            }

        if self.state == JSONState.EXPECT_FRACTION_DIGIT:
            return {
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            }

        return set()

    def consume_token(self, token: str) -> bool:
        """
        Consume a complete tokenizer token.

        Returns True if the complete token is valid.
        """
        original_state = self.state
        original_stack = self.object_stack.copy()

        for char in token:
            if not self.consume_char(char):
                self.state = original_state
                self.object_stack = original_stack
                return False

        return True

    def is_whitespace(self, char: str) -> bool:
        return char in (" ", "\t", "\n", "\r")
