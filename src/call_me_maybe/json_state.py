#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_state.py                                        :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:41:02 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/09 11:41:12 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from enum import Enum, auto


class JSONState(Enum):
    START = auto()

    EXPECT_KEY_START = auto()
    KEY_CONTENT = auto()
    EXPECT_COLON = auto()

    EXPECT_VALUE_START = auto()
    VALUE_CONTENT = auto()
    VALUE_NUMBER = auto()
    EXPECT_COMMA = auto()
    EXPECT_OBJECT_END = auto()

    DONE = auto()
