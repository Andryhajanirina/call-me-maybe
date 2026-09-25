#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_context.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/09/15 14:36:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/22 11:23:15 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #


"""Semantic tracker for keys and value constraints during JSON generation."""

from typing import Optional, Any


class JSONContext:
    """Tracks current JSON key, argument path, and schema constraints."""

    def __init__(self) -> None:
        """Initialize empty context state."""
        self.current_key: str = ""
        self.building_key: list[str] = []
        self.parsed_object: dict[str, Any] = {}
        self.current_function_name: Optional[str] = None
        self.expected_value_type: Optional[str] = None

    def clone(self) -> "JSONContext":
        """Create a copy of current semantic context for simulation.

        Returns:
            A cloned instance of JSONContext.
        """
        new_ctx = JSONContext()
        new_ctx.current_key = self.current_key
        new_ctx.building_key = list(self.building_key)
        new_ctx.current_function_name = self.current_function_name
        new_ctx.expected_value_type = self.expected_value_type
        return new_ctx

    def start_key(self) -> None:
        """Reset key buffer when a quote opening a key is encountered."""
        self.building_key = []

    def add_key_character(self, char: str) -> None:
        """Append character to the active key buffer.

        Args:
            char: Character inside key string.
        """
        self.building_key.append(char)

    def finish_key(self) -> str:
        """Finalize and store the current key.

        Returns:
            The complete key string.
        """
        self.current_key = "".join(self.building_key)
        self.building_key = []
        return self.current_key
