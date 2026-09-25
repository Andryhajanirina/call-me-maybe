#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   __init__.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/08/19 11:44:03 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/22 13:15:15 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from .font_logo import LogoGenerator
from .main import main
# # from .explore_vocab import main
# # from .test_schema import main

# # from .test_token_constraints import main
# # from .test_json_context import main
# from .test_json_decoder import main


LogoGenerator.print_two_tone_logo(
    text1="Call me", couleur1="yellow",
    text2=" Maybe", couleur2="cyan",
    font="slant",
    margin_top=0
)
__all__ = ["main"]
