#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   font_logo.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: andry-ha <andry-ha@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/09/07 11:41:32 by andry-ha            #+#    #+#            #
#   Updated: 2026/09/13 10:55:52 by andry-ha           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import pyfiglet
import subprocess


class LogoGenerator:
    # ANSI Color Dictionary
    COLORS = {
        'red': "\033[31m",
        'green': "\033[32m",
        'yellow': "\033[33m",
        'blue': "\033[34m",
        'magenta': "\033[35m",
        'cyan': "\033[36m",
        'white': "\033[37m",
        'default': "\033[0m"
    }

    @classmethod
    def print_logo(cls,
                   text: str,
                   couleur: str = 'cyan',
                   font: str = 'standard') -> None:
        "Generates a dynamic logo with Pyfiglet and a custom color"

        # Automatic generation of ASCII art text
        try:
            ascii_art = pyfiglet.figlet_format(
                text, font=font, width=100, justify="center")
        except Exception:
            # Fallback to the default font if the requested font does not exist
            ascii_art = pyfiglet.figlet_format(
                text, font='standard', width=100, justify="center")

        color_code = cls.COLORS.get(couleur.lower(), "\033[37m")
        reset_code = cls.COLORS['default']

        print(color_code + ascii_art + reset_code)

    @classmethod
    def _get_cleaned_lines(cls, text: str, font: str) -> list:
        """Generates the text and cleans up empty lines at the top/bottom"""
        try:
            ascii_art = pyfiglet.figlet_format(text, font=font)
        except Exception:
            ascii_art = pyfiglet.figlet_format(text, font='standard')

        lines = ascii_art.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)
        return lines

    @classmethod
    def print_two_tone_logo(cls,
                            text1: str, couleur1: str,
                            text2: str, couleur2: str,
                            font: str = 'standard',
                            margin_top: int = 0,
                            width: int = 50) -> None:
        """Displays logo composed of two text segments in different colors."""

        command = 'cls' if subprocess.os.name == 'nt' else 'clear'
        subprocess.run(command, shell=True)

        lines1 = cls._get_cleaned_lines(text1, font)
        lines2 = cls._get_cleaned_lines(text2, font)

        max_height = max(len(lines1), len(lines2))
        width1 = max(len(length) for length in lines1) if lines1 else 0
        width2 = max(len(length) for length in lines2) if lines2 else 0

        while len(lines1) < max_height:
            lines1.append(" " * width1)
        while len(lines2) < max_height:
            lines2.append(" " * width2)

        c1 = cls.COLORS.get(couleur1.lower(), cls.COLORS['white'])
        c2 = cls.COLORS.get(couleur2.lower(), cls.COLORS['cyan'])
        reset = cls.COLORS['default']

        merged_lines = []
        total_content_width = width1 + width2
        padding_left_size = max(0, (width - total_content_width) // 2)
        global_padding_left = " " * padding_left_size

        for l1, l2 in zip(lines1, lines2):
            l1_padded = l1.ljust(width1)
            full_line = f"{global_padding_left}{c1}{l1_padded}{c2}{l2}{reset}"
            merged_lines.append(full_line)

        top_spacing = "\n" * margin_top
        print(top_spacing + "\n".join(merged_lines))


if __name__ == '__main__':
    # You can change the text, color and font style in
    # ('standard', 'slant', 'block', 'doom', etc.)
    LogoGenerator.print_logo("PYTHON_v3", couleur="green", font="slant")
