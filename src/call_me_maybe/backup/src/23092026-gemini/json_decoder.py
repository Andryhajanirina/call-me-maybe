#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_decoder.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:40:38 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/22 11:20:26 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""Finite State Machine for incremental JSON syntax validation."""


from .json_state import JSONState


class JSONDecoder:

    def __init__(self) -> None:
        """Initialize decoder at START state."""
        self.state: JSONState = JSONState.START
        self.stack: list[str] = []
        self.escape_next: bool = False

    def clone(self) -> "JSONDecoder":
        """Create a deep copy of the current decoder state for simulation.

        Returns:
            A cloned JSONDecoder instance in the exact same state.
        """
        new_decoder = JSONDecoder()
        new_decoder.state = self.state
        new_decoder.stack = list(self.stack)
        new_decoder.escape_next = self.escape_next
        return new_decoder

    def consume_char(self, char: str) -> bool:
        """Process a single character and transition to the next state.

        Args:
            char: Character to process.

        Returns:
            True if the character is syntactically valid in the current state,
            False otherwise.
        """
        if self.state == JSONState.DONE:
            return False

        # Gestion de l'échappement dans les chaînes
        if self.escape_next:
            self.escape_next = False
            return True

        if self.state in (JSONState.KEY_CONTENT, JSONState.VALUE_CONTENT):
            if char == "\\":
                self.escape_next = True
                return True
            if char == '"':
                if self.state == JSONState.KEY_CONTENT:
                    self.state = JSONState.EXPECT_COLON
                else:
                    self.state = JSONState.EXPECT_COMMA
                return True
            # Tout autre caractère est valide à l'intérieur des guillemets
            return True

        # Gestion des espaces/blancs
        if char in " \t\n\r":
            # Les espaces sont tolérés dans les états de transition
            if self.state in (
                JSONState.START,
                JSONState.EXPECT_KEY_OR_END,
                JSONState.EXPECT_KEY_START,
                JSONState.EXPECT_COLON,
                JSONState.EXPECT_VALUE_START,
                JSONState.EXPECT_COMMA,
                JSONState.EXPECT_OBJECT_END,
            ):
                return True

        # Transitions selon l'état actuel
        if self.state == JSONState.START:
            if char == "{":
                self.stack.append("}")
                self.state = JSONState.EXPECT_KEY_OR_END
                return True
            return False

        elif self.state == JSONState.EXPECT_KEY_OR_END:
            if char == "}":
                if self.stack and self.stack[-1] == "}":
                    self.stack.pop()
                    self.state = JSONState.DONE if not self.stack else JSONState.EXPECT_COMMA
                    return True
                return False
            if char == '"':
                self.state = JSONState.KEY_CONTENT
                return True
            return False

        elif self.state == JSONState.EXPECT_KEY_START:
            if char == '"':
                self.state = JSONState.KEY_CONTENT
                return True
            return False

        elif self.state == JSONState.EXPECT_COLON:
            if char == ":":
                self.state = JSONState.EXPECT_VALUE_START
                return True
            return False

        elif self.state == JSONState.EXPECT_VALUE_START:
            if char == '"':
                self.state = JSONState.VALUE_CONTENT
                return True
            if char.isdigit() or char == "-":
                self.state = JSONState.VALUE_NUMBER
                return True
            if char in ("t", "f", "n"):  # true, false, null
                self.state = JSONState.VALUE_CONTENT  # ou état dédié bool
                return True
            return False

        elif self.state == JSONState.VALUE_NUMBER:
            if char.isdigit() or char in ".eE+-":
                return True
            if char == ",":
                self.state = JSONState.EXPECT_KEY_START
                return True
            if char == "}":
                if self.stack and self.stack[-1] == "}":
                    self.stack.pop()
                    self.state = JSONState.DONE if not self.stack else JSONState.EXPECT_COMMA
                    return True
                return False
            return False

        elif self.state == JSONState.EXPECT_COMMA:
            if char == ",":
                self.state = JSONState.EXPECT_KEY_START
                return True
            if char == "}":
                if self.stack and self.stack[-1] == "}":
                    self.stack.pop()
                    self.state = JSONState.DONE if not self.stack else JSONState.EXPECT_COMMA
                    return True
                return False
            return False

        return False
