#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_decoder.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:40:38 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/13 11:10:51 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

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

        if self.is_whitespace(char):
            return True

        if self.state == JSONState.START:
            if char == "{":
                self.object_stack.append("object")
                self.state = JSONState.EXPECT_KEY_START
                return True

        elif self.state == JSONState.EXPECT_KEY_START:
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

            if char.isdigit():
                self.state = JSONState.VALUE_NUMBER
                return True

        elif self.state == JSONState.VALUE_CONTENT:
            if char == '"':
                self.state = JSONState.EXPECT_COMMA
                return True
            return True

        elif self.state == JSONState.VALUE_NUMBER:
            if char.isdigit():
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
        original_stack = self.object_stack.copy()

        for char in token:
            if not self.consume_char(char):
                self.state = original_state
                self.object_stack = original_stack
                return False

        return True

    def is_whitespace(self, char: str) -> bool:
        return char in (" ", "\t", "\n", "\r")
