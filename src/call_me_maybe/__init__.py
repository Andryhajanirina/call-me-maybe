# from .main import main
# from .explore_vocab import main
# from .test_schema import main

# from .test_token_constraints import main
from .font_logo import LogoGenerator
from .test_json_decoder import main


LogoGenerator.print_two_tone_logo(
    text1="Call me", couleur1="yellow",
    text2=" Maybe", couleur2="cyan",
    font="slant",
    margin_top=0
)
__all__ = ["main"]
