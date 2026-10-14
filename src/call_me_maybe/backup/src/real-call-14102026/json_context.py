#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_context.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/09/15 14:36:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/10/04 10:10:36 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #


class JSONContext:
    def __init__(self) -> None:
        self.current_key = ""

    def start_key(self) -> None:
        self.current_key = ""

    def add_key_character(self, char: str) -> None:
        self.current_key += char

    def finish_key(self) -> str:
        return self.current_key

    def is_key(self, key: str) -> bool:
        return self.current_key == key

    def is_function_name_key(self) -> bool:
        return self.current_key == "name"
