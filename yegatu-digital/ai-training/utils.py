'''
Utils that can be imported from this file:

-> 'Unicode' class: to ensure the use of valid unicode representation of chars;

-> 'SpecialCharacters' class: lists the valid chars specific to each spelling
    variant of nheengatu;

-> 'VALID_SPECIAL_CHARS': a set of all valid special chars for nheengatu,
    including all the spelling variants;

-> 'NHEENGATU_TO_REPLACE': a dict with characters to replace in nheengatu text,
    and their correspondent valid character;

-> 'normalize_nheengatu': a function to normalize nheengatu text (replaces
    invalid characters).
'''

from enum import Enum
from enum import StrEnum


class Unicode(StrEnum):
    '''
    These are constants to ensure we're refering
    only to unicode characters.
    '''

    # These are valid special characters used for nheengatu
    A_ACUTE = chr(225)
    E_ACUTE = chr(233)
    I_ACUTE = chr(237)
    O_ACUTE = chr(243)
    U_ACUTE = chr(250)
    Y_ACUTE = chr(253)
    A_TILDE = chr(227)
    E_TILDE = chr(7869)
    I_TILDE = chr(297)
    O_TILDE = chr(245)
    U_TILDE = chr(361)
    Y_TILDE = chr(7929)
    C_CEDILLA = chr(231)

    # These are 'combining marks' that shouldn't be used
    # (codes between 768 and 879, inclusive)
    COMBINING_ACUTE = chr(769)
    COMBINING_TILDE = chr(771)


class SpecialCharacters(Enum):
    '''
    These are the valid special characters for
    each spelling variant of nheengatu.
    '''

    NAVARRO = [
        Unicode.A_ACUTE,
        Unicode.E_ACUTE,
        Unicode.I_ACUTE,
        Unicode.U_ACUTE,
        Unicode.A_TILDE,
        Unicode.E_TILDE,
        Unicode.I_TILDE,
        Unicode.U_TILDE,
    ]
    
    CASASNOVAS = [
        Unicode.A_ACUTE,
        Unicode.E_ACUTE,
        Unicode.I_ACUTE,
        Unicode.U_ACUTE,
        Unicode.A_TILDE,
        Unicode.E_TILDE,
        Unicode.I_TILDE,
        Unicode.U_TILDE,
    ]
    
    ALTO_RIO_NEGRO = [
        Unicode.A_TILDE,
        Unicode.E_TILDE,
        Unicode.I_TILDE,
        Unicode.U_TILDE,
    ]
    
    BAIXO_AMAZONAS = [
        Unicode.A_ACUTE,
        Unicode.E_ACUTE,
        Unicode.I_ACUTE,
        Unicode.O_ACUTE,
        Unicode.U_ACUTE,
        Unicode.Y_ACUTE,
        Unicode.A_TILDE,
        Unicode.E_TILDE,
        Unicode.I_TILDE,
        Unicode.O_TILDE,
        Unicode.U_TILDE,
        Unicode.Y_TILDE,
        Unicode.C_CEDILLA,
    ]


'''
A set of all valid special chars for nheengatu,
including all the spelling variants.
'''
VALID_SPECIAL_CHARS = set()
for variant in SpecialCharacters:
    VALID_SPECIAL_CHARS.update(variant.value)


'''
Every invalid character or combination of characters
and their correspondent unicode representation for
nheengatu.
'''
NHEENGATU_TO_REPLACE = {
    # Encoding inconsistencies:
    'a' + Unicode.COMBINING_ACUTE: Unicode.A_ACUTE,
    'e' + Unicode.COMBINING_ACUTE: Unicode.E_ACUTE,
    'i' + Unicode.COMBINING_ACUTE: Unicode.I_ACUTE,
    'o' + Unicode.COMBINING_ACUTE: Unicode.O_ACUTE,
    'u' + Unicode.COMBINING_ACUTE: Unicode.U_ACUTE,
    'y' + Unicode.COMBINING_ACUTE: Unicode.Y_ACUTE,
    'a' + Unicode.COMBINING_TILDE: Unicode.A_TILDE,
    'e' + Unicode.COMBINING_TILDE: Unicode.E_TILDE,
    'i' + Unicode.COMBINING_TILDE: Unicode.I_TILDE,
    'o' + Unicode.COMBINING_TILDE: Unicode.O_TILDE,
    'u' + Unicode.COMBINING_TILDE: Unicode.U_TILDE,
    'y' + Unicode.COMBINING_TILDE: Unicode.Y_TILDE,
    # Character equivalence specific to nheengatu:
    'ä': Unicode.A_TILDE,
    'ā': Unicode.A_TILDE,
    'ă': Unicode.A_TILDE,
    'ë': Unicode.E_TILDE,
    'ē': Unicode.E_TILDE,
    'ï': Unicode.I_TILDE,
    'ī': Unicode.I_TILDE,
    'ῖ': Unicode.I_TILDE,
    'ö': Unicode.O_TILDE,
    'ü': Unicode.U_TILDE,
    'ū': Unicode.U_TILDE,
    'ữ': Unicode.U_TILDE,
    'ñ': 'nh',
    # Other invalid characters found in the dataset:
    'ŕ': 'r',
    'ķ': 'k',
    # TODO - Other characters that might need to be replaced, but we need to
    #        check by which characters (character : count)
    # à : 104
    # â : 28
    # è : 59
    # ê : 970
    # ì : 7
    # ò : 61
    # ô : 54
    # ù : 50
    # û : 1
}


def normalize_nheengatu(text: str) -> str:
    '''
    Replaces every instance of invalid nheengatu
    characters in a given string.
    '''

    for s in NHEENGATU_TO_REPLACE.keys():
        text = text.replace(s, NHEENGATU_TO_REPLACE[s])
    
    return text
