#!/usr/bin/env python3
"""Morse code <-> text converter.

Usage:
    python morse.py "... --- ..."            # decode Morse to text
    python morse.py --encode "hello world"   # encode text to Morse
    echo "... --- ..." | python morse.py     # decode from stdin

As a module:
    from morse import decode, encode
    decode("... --- ...")   -> "SOS"
    encode("SOS")           -> "... --- ..."

Conventions (International Morse):
    '.' = dit, '-' = dah, single space between letters, '/' (or 3+ spaces)
    between words. Unknown symbols decode to '?'.
"""

import sys
import argparse

MORSE_TO_CHAR = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!",
    "-..-.": "/", "-.--.": "(", "-.--.-": ")", ".-...": "&", "---...": ":",
    "-.-.-.": ";", "-...-": "=", ".-.-.": "+", "-....-": "-", "..--.-": "_",
    ".-..-.": '"', "...-..-": "$", ".--.-.": "@",
}
CHAR_TO_MORSE = {c: m for m, c in MORSE_TO_CHAR.items()}


def _normalize(morse: str) -> str:
    """Accept common variants: underscores for dahs, bullets/middle dots for dits,
    '|' as a word separator."""
    return (morse.replace("_", "-").replace("•", ".").replace("·", ".")
                 .replace("|", "/").strip())


def decode(morse: str) -> str:
    """Decode a Morse string to text. Letters separated by spaces; words by '/'
    or by three or more spaces."""
    morse = _normalize(morse)
    if not morse:
        return ""
    # Treat runs of 3+ spaces as word gaps too.
    import re
    morse = re.sub(r" {3,}", " / ", morse)
    words = []
    for word in morse.split("/"):
        letters = [MORSE_TO_CHAR.get(sym, "?") for sym in word.split()]
        words.append("".join(letters))
    return " ".join(w for w in words).strip()


def encode(text: str) -> str:
    """Encode text to Morse. Letters separated by one space, words by ' / '.
    Characters with no Morse equivalent are skipped."""
    words = []
    for word in text.upper().split():
        syms = [CHAR_TO_MORSE[c] for c in word if c in CHAR_TO_MORSE]
        if syms:
            words.append(" ".join(syms))
    return " / ".join(words)


def main(argv=None):
    p = argparse.ArgumentParser(description="Morse code <-> text converter")
    p.add_argument("input", nargs="?", help="string to convert (reads stdin if omitted)")
    p.add_argument("-e", "--encode", action="store_true", help="text -> Morse (default is Morse -> text)")
    args = p.parse_args(argv)
    data = args.input if args.input is not None else sys.stdin.read()
    print(encode(data) if args.encode else decode(data))


if __name__ == "__main__":
    main()
