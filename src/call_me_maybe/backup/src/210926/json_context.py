#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_context.py                                      :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/09/15 14:36:22 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/16 15:38:32 by andry-ha           ###   ########.fr      #
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
