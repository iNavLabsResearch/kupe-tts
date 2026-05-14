"""
Digit-to-word conversion for TTS preprocessing (kupe-tts).

**Cardinal** (natural phrasing for whole numbers): ``en``, ``hi``, ``gu``, ``kn``.

**Digit-by-digit** (native 0–9 + minus + decimal word, good for every script and
long codes): Bengali, Assamese, Tamil, Telugu, Malayalam, Gurmukhi, Odia, Nepali,
Sinhala, Urdu, Arabic, Myanmar, Thai, Lao, Khmer, Spanish, French, German,
Portuguese, Italian, Dutch, Polish, Russian, Ukrainian, Turkish, Vietnamese,
Chinese, Japanese, Korean, Czech, Slovak, Hungarian, Romanian, Swedish, Danish,
Finnish, Greek, Hebrew, Indonesian, Malay, Tagalog — plus **aliases** for many
more ISO 639 codes (Indian and foreign) that route to the nearest locale above.

Auto-detect picks a default pronunciation from the dominant Unicode script.
Optional ``digit_pronunciation`` overrides everything; ``digit_words_hint`` =
*hinglish* forces English digit words in Indic/SEA scripts when no explicit
pronunciation was set.

**Grouping commas / spaces:** numerals like ``1,50,00,000`` (Indian lakhs/crores),
``15,000,000`` (Western thousands), and thin-space / underscore groupings are
normalised to plain digits before cardinal conversion. Use a **full stop** for
decimals (e.g. ``1,50,000.5``).

No third-party libraries (pure tables + regex).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

# ──────────────────────────────────────────────────────────────────────────────
# English number words
# ──────────────────────────────────────────────────────────────────────────────

_EN_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _en_below_1000(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        rem = n % 10
        return _EN_TENS[n // 10] + ("-" + _EN_ONES[rem] if rem else "")
    rem = n % 100
    return _EN_ONES[n // 100] + " hundred" + (" " + _en_below_1000(rem) if rem else "")


def _en_integer(n: int) -> str:
    if n < 0:
        return "minus " + _en_integer(-n)
    if n == 0:
        return "zero"
    parts = []
    scales = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
    for scale, name in scales:
        if n >= scale:
            parts.append(_en_below_1000(n // scale) + " " + name)
            n %= scale
    if n:
        parts.append(_en_below_1000(n))
    return " ".join(parts)


def _en_decimal_part(dec_str: str) -> str:
    return " ".join(_EN_ONES[int(d)] for d in dec_str)


def _en_number(num_str: str) -> str:
    negative = num_str.startswith("-")
    num_str = num_str.lstrip("-")
    if "." in num_str:
        int_part, dec_part = num_str.split(".", 1)
        result = _en_integer(int(int_part)) + " point " + _en_decimal_part(dec_part)
    else:
        result = _en_integer(int(num_str))
    return ("minus " + result) if negative else result


# ──────────────────────────────────────────────────────────────────────────────
# Hindi / Marathi (Devanagari)
# ──────────────────────────────────────────────────────────────────────────────

_HI_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह",
    "सत्रह", "अठारह", "उन्नीस",
]
_HI_TENS = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]
_HI_SPECIALS = {
    21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस",
    27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस", 31: "इकतीस", 32: "बत्तीस",
    33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस",
    38: "अड़तीस", 39: "उनचालीस", 41: "इकतालीस", 42: "बयालीस", 43: "तैंतालीस",
    44: "चवालीस", 45: "पैंतालीस", 46: "छियालीस", 47: "सैंतालीस", 48: "अड़तालीस",
    49: "उनचास", 51: "इक्यावन", 52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन",
    56: "छप्पन", 57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 61: "इकसठ", 62: "बासठ",
    63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ", 66: "छियासठ", 67: "सड़सठ", 68: "अड़सठ",
    69: "उनहत्तर", 71: "इकहत्तर", 72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर",
    75: "पचहत्तर", 76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्यासी",
    81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 91: "इक्यानवे",
    92: "बानवे", 93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे", 96: "छियानवे",
    97: "सत्तानवे", 98: "अट्ठानवे", 99: "निन्यानवे",
}


def _hi_below_100(n: int) -> str:
    if n < 20:
        return _HI_ONES[n]
    if n in _HI_SPECIALS:
        return _HI_SPECIALS[n]
    tens = _HI_TENS[n // 10]
    ones = _HI_ONES[n % 10]
    return (tens + " " + ones).strip() if ones else tens


def _hi_below_1000(n: int) -> str:
    if n < 100:
        return _hi_below_100(n)
    h = n // 100
    rem = n % 100
    hundreds = _HI_ONES[h] + " सौ"
    return (hundreds + " " + _hi_below_100(rem)).strip() if rem else hundreds


def _hi_integer(n: int) -> str:
    if n < 0:
        return "ऋण " + _hi_integer(-n)
    if n == 0:
        return "शून्य"
    parts = []
    if n >= 10**9:
        parts.append(_hi_below_1000(n // 10**9) + " अरब")
        n %= 10**9
    if n >= 10**7:
        parts.append(_hi_below_100(n // 10**7) + " करोड़")
        n %= 10**7
    if n >= 10**5:
        parts.append(_hi_below_100(n // 10**5) + " लाख")
        n %= 10**5
    if n >= 1000:
        parts.append(_hi_below_100(n // 1000) + " हज़ार")
        n %= 1000
    if n:
        parts.append(_hi_below_1000(n))
    return " ".join(parts)


def _hi_number(num_str: str) -> str:
    negative = num_str.startswith("-")
    num_str = num_str.lstrip("-")
    if "." in num_str:
        int_part, dec_part = num_str.split(".", 1)
        dec_words = " ".join("शून्य" if d == "0" else _HI_ONES[int(d)] for d in dec_part)
        result = _hi_integer(int(int_part)) + " दशमलव " + dec_words
    else:
        result = _hi_integer(int(num_str))
    return ("ऋण " + result) if negative else result


# ──────────────────────────────────────────────────────────────────────────────
# Gujarati
# ──────────────────────────────────────────────────────────────────────────────

_GU_ONES = [
    "", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ",
    "દસ", "અગ્યાર", "બાર", "તેર", "ચૌદ", "પંદર", "સોળ",
    "સત્તર", "અઢાર", "ઓગણીસ",
]
_GU_TENS = ["", "", "વીસ", "ત્રીસ", "ચાળીસ", "પચાસ", "સાઠ", "સિત્તેર", "એંસી", "નેવું"]
_GU_SPECIALS = {
    21: "એકવીસ", 22: "બાવીસ", 23: "તેવીસ", 24: "ચોવીસ", 25: "પચ્ચીસ",
    26: "છવ્વીસ", 27: "સત્તાવીસ", 28: "અઠ્ઠાવીસ", 29: "ઓગણત્રીસ",
    31: "એકત્રીસ", 32: "બત્રીસ", 33: "તેત્રીસ", 34: "ચૌત્રીસ", 35: "પાંત્રીસ",
    36: "છત્રીસ", 37: "સાડત્રીસ", 38: "અડત્રીસ", 39: "ઓગણચાળીસ",
    41: "એકતાળીસ", 42: "બેતાળીસ", 43: "ત્રેતાળીસ", 44: "ચુંમાળીસ", 45: "પિસ્તાળીસ",
    46: "છેતાળીસ", 47: "સુડતાળીસ", 48: "અડતાળીસ", 49: "ઓગણપચાસ",
    51: "એકાવન", 52: "બાવન", 53: "ત્રેપન", 54: "ચોપન", 55: "પંચાવન",
    56: "છપ્પન", 57: "સત્તાવન", 58: "અઠ્ઠાવન", 59: "ઓગણસાઠ",
    61: "એકસઠ", 62: "બાસઠ", 63: "ત્રેસઠ", 64: "ચોસઠ", 65: "પાંસઠ",
    66: "છાસઠ", 67: "સડસઠ", 68: "અડસઠ", 69: "અગણોસિત્તેર",
    71: "એકોત્તેર", 72: "બોત્તેર", 73: "ત્રોત્તેર", 74: "ચોત્તેર", 75: "પંચોત્તેર",
    76: "છોત્તેર", 77: "સત્યોત્તેર", 78: "અઠ્યોત્તેર", 79: "ઓગણએંસી",
    81: "એક્યાશી", 82: "બ્યાશી", 83: "ત્ર્યાશી", 84: "ચોર્યાશી", 85: "પંચ્યાશી",
    86: "છ્યાશી", 87: "સત્યાશી", 88: "અઠ્યાશી", 89: "નેવ્યાશી",
    91: "એક્યાણું", 92: "બ્યાણું", 93: "ત્ર્યાણું", 94: "ચોર્યાણું", 95: "પંચ્યાણું",
    96: "છ્યાણું", 97: "સત્યાણું", 98: "અઠ્યાણું", 99: "નેવ્યાણું",
}


def _gu_below_100(n: int) -> str:
    if n < 20:
        return _GU_ONES[n]
    if n in _GU_SPECIALS:
        return _GU_SPECIALS[n]
    return (_GU_TENS[n // 10] + " " + _GU_ONES[n % 10]).strip()


def _gu_below_1000(n: int) -> str:
    if n < 100:
        return _gu_below_100(n)
    h = n // 100
    rem = n % 100
    hundreds = _GU_ONES[h] + " સો"
    return (hundreds + " " + _gu_below_100(rem)).strip() if rem else hundreds


def _gu_integer(n: int) -> str:
    if n < 0:
        return "ઋણ " + _gu_integer(-n)
    if n == 0:
        return "શૂન્ય"
    parts = []
    if n >= 10**7:
        parts.append(_gu_below_100(n // 10**7) + " કરોડ")
        n %= 10**7
    if n >= 10**5:
        parts.append(_gu_below_100(n // 10**5) + " લાખ")
        n %= 10**5
    if n >= 1000:
        parts.append(_gu_below_100(n // 1000) + " હજાર")
        n %= 1000
    if n:
        parts.append(_gu_below_1000(n))
    return " ".join(parts)


def _gu_number(num_str: str) -> str:
    negative = num_str.startswith("-")
    num_str = num_str.lstrip("-")
    if "." in num_str:
        int_part, dec_part = num_str.split(".", 1)
        dec_words = " ".join("શૂન્ય" if d == "0" else _GU_ONES[int(d)] for d in dec_part)
        result = _gu_integer(int(int_part)) + " દશાંશ " + dec_words
    else:
        result = _gu_integer(int(num_str))
    return ("ઋણ " + result) if negative else result


# ──────────────────────────────────────────────────────────────────────────────
# Kannada
# ──────────────────────────────────────────────────────────────────────────────

_KN_ONES = [
    "", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು",
    "ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು", "ಹದಿನೈದು",
    "ಹದಿನಾರು", "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು",
]
_KN_TENS = ["", "", "ಇಪ್ಪತ್ತು", "ಮೂವತ್ತು", "ನಲವತ್ತು", "ಐವತ್ತು", "ಅರವತ್ತು", "ಎಪ್ಪತ್ತು", "ಎಂಬತ್ತು", "ತೊಂಬತ್ತು"]


def _kn_below_100(n: int) -> str:
    if n < 20:
        return _KN_ONES[n]
    ones = _KN_ONES[n % 10]
    return (_KN_TENS[n // 10] + (" " + ones if ones else "")).strip()


def _kn_below_1000(n: int) -> str:
    if n < 100:
        return _kn_below_100(n)
    h = _KN_ONES[n // 100]
    rem = n % 100
    hundreds = h + " ನೂರು"
    return (hundreds + " " + _kn_below_100(rem)).strip() if rem else hundreds


def _kn_integer(n: int) -> str:
    if n < 0:
        return "ಋಣ " + _kn_integer(-n)
    if n == 0:
        return "ಸೊನ್ನೆ"
    parts = []
    if n >= 10**7:
        parts.append(_kn_below_100(n // 10**7) + " ಕೋಟಿ")
        n %= 10**7
    if n >= 10**5:
        parts.append(_kn_below_100(n // 10**5) + " ಲಕ್ಷ")
        n %= 10**5
    if n >= 1000:
        parts.append(_kn_below_100(n // 1000) + " ಸಾವಿರ")
        n %= 1000
    if n:
        parts.append(_kn_below_1000(n))
    return " ".join(parts)


def _kn_number(num_str: str) -> str:
    negative = num_str.startswith("-")
    num_str = num_str.lstrip("-")
    if "." in num_str:
        int_part, dec_part = num_str.split(".", 1)
        dec_words = " ".join("ಸೊನ್ನೆ" if d == "0" else _KN_ONES[int(d)] for d in dec_part)
        result = _kn_integer(int(int_part)) + " ದಶಮಾಂಶ " + dec_words
    else:
        result = _kn_integer(int(num_str))
    return ("ಋಣ " + result) if negative else result


# ──────────────────────────────────────────────────────────────────────────────
# Digit-by-digit lexicons (native 0–9 + minus + decimal word) — no third-party libs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SpokenLex:
    """Spoken labels for each ASCII digit 0–9, minus, and the decimal separator."""

    digits: tuple[str, ...]  # length 10 — index = digit value
    minus: str
    point: str


def _lexicon_number(num_str: str, lex: _SpokenLex) -> str:
    negative = num_str.startswith("-")
    body = num_str.lstrip("-")
    if "." in body:
        int_part, dec_part = body.split(".", 1)
    else:
        int_part, dec_part = body, ""
    parts: list[str] = []
    if negative:
        parts.append(lex.minus)
    if int_part:
        parts.append(" ".join(lex.digits[int(c)] for c in int_part if c.isdigit()))
    if dec_part:
        parts.append(lex.point)
        parts.append(" ".join(lex.digits[int(c)] for c in dec_part if c.isdigit()))
    return " ".join(p for p in parts if p)


def _lexicon_conv(lex: _SpokenLex) -> Callable[[str], str]:
    def _f(num_str: str) -> str:
        return _lexicon_number(num_str, lex)

    return _f


# Major Indic + SEA + widely used foreign locales (digit-by-digit where noted).
_SPOKEN_LEXICONS: dict[str, _SpokenLex] = {
    "bn": _SpokenLex(
        ("শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়"),
        "ঋণ", "দশমিক",
    ),
    "ta": _SpokenLex(
        ("பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"),
        "கழிவு", "புள்ளி",
    ),
    "te": _SpokenLex(
        ("సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"),
        "ఋణ", "బిందువు",
    ),
    "ml": _SpokenLex(
        ("പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "എഴ്", "എട്ട്", "ഒമ്പത്"),
        "ഋണ", "പുള്ളി",
    ),
    "pa": _SpokenLex(
        ("ਸਿਫਰ", "ਇੱਕ", "ਦੋ", "ਤਿੰਨ", "ਚਾਰ", "ਪੰਜ", "ਛੇ", "ਸੱਤ", "ਅੱਠ", "ਨੌਂ"),
        "਋ਣ", "ਦਸ਼ਮਲਵ",
    ),
    "or": _SpokenLex(
        ("ଶୂନ୍ୟ", "ଏକ", "ଦୁଇ", "ତିନି", "ଚାରି", "ପାଞ୍ଚ", "ଛଅ", "ସାତ", "ଆଠ", "ନଅ"),
        "ଋଣ", "ଦଶମିକ",
    ),
    "ne": _SpokenLex(
        ("शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ"),
        "ऋण", "दशमलव",
    ),
    "si": _SpokenLex(
        ("ශුන්‍ය", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නමය"),
        "ඍණ", "දශම",
    ),
    "ur": _SpokenLex(
        ("صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو"),
        "منفی", "اعشاریہ",
    ),
    "ar": _SpokenLex(
        ("صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"),
        "سالب", "فاصلة",
    ),
    "my": _SpokenLex(
        ("သုည", "တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ရှစ်", "ကိုး"),
        "အနုတ်", "မှတ်",
    ),
    "th": _SpokenLex(
        ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"),
        "ลบ", "จุด",
    ),
    "lo": _SpokenLex(
        ("ສູນ", "ໜຶ່ງ", "ສອງ", "ສາມ", "ສີ່", "ຫ້າ", "ຫົກ", "ເຈັດ", "ແປດ", "ເກົ້າ"),
        "ລົບ", "ຈຸດ",
    ),
    "km": _SpokenLex(
        ("សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"),
        "អវិជ្ជមាន", "ចុច",
    ),
    "es": _SpokenLex(
        ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"),
        "menos", "punto",
    ),
    "fr": _SpokenLex(
        ("zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"),
        "moins", "virgule",
    ),
    "de": _SpokenLex(
        ("null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"),
        "minus", "komma",
    ),
    "pt": _SpokenLex(
        ("zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"),
        "menos", "vírgula",
    ),
    "it": _SpokenLex(
        ("zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"),
        "meno", "virgola",
    ),
    "nl": _SpokenLex(
        ("nul", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"),
        "min", "komma",
    ),
    "pl": _SpokenLex(
        ("zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć"),
        "minus", "przecinek",
    ),
    "ru": _SpokenLex(
        ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
        "минус", "запятая",
    ),
    "uk": _SpokenLex(
        ("нуль", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять"),
        "мінус", "кома",
    ),
    "tr": _SpokenLex(
        ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"),
        "eksi", "virgül",
    ),
    "vi": _SpokenLex(
        ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"),
        "âm", "phẩy",
    ),
    "zh": _SpokenLex(
        ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九"),
        "负", "点",
    ),
    "ja": _SpokenLex(
        ("ゼロ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"),
        "マイナス", "てん",
    ),
    "ko": _SpokenLex(
        ("영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"),
        "마이너스", "점",
    ),
    "cs": _SpokenLex(
        ("nula", "jedna", "dva", "tři", "čtyři", "pět", "šest", "sedm", "osm", "devět"),
        "mínus", "čárka",
    ),
    "sk": _SpokenLex(
        ("nula", "jeden", "dva", "tri", "štyri", "päť", "šesť", "sedem", "osem", "deväť"),
        "mínus", "čiarka",
    ),
    "hu": _SpokenLex(
        ("nulla", "egy", "kettő", "három", "négy", "öt", "hat", "hét", "nyolc", "kilenc"),
        "mínusz", "vessző",
    ),
    "ro": _SpokenLex(
        ("zero", "unu", "doi", "trei", "patru", "cinci", "șase", "șapte", "opt", "nouă"),
        "minus", "virgulă",
    ),
    "sv": _SpokenLex(
        ("noll", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio"),
        "minus", "komma",
    ),
    "da": _SpokenLex(
        ("nul", "et", "to", "tre", "fire", "fem", "seks", "syv", "otte", "ni"),
        "minus", "komma",
    ),
    "fi": _SpokenLex(
        ("nolla", "yksi", "kaksi", "kolme", "neljä", "viisi", "kuusi", "seitsemän", "kahdeksan", "yhdeksän"),
        "miinus", "pilkku",
    ),
    "el": _SpokenLex(
        ("μηδέν", "ένα", "δύο", "τρία", "τέσσερα", "πέντε", "έξι", "επτά", "οκτώ", "εννέα"),
        "πλην", "υποδιαστολή",
    ),
    "he": _SpokenLex(
        ("אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"),
        "מינוס", "נקודה",
    ),
    "id": _SpokenLex(
        ("nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"),
        "minus", "koma",
    ),
    "ms": _SpokenLex(
        ("sifar", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan"),
        "negatif", "perpuluhan",
    ),
    "tl": _SpokenLex(
        ("sero", "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam"),
        "negatibo", "punto",
    ),
}

# Fix Assamese tuple (avoid duplicate bug in construction above)
_SPOKEN_LEXICONS["as"] = _SpokenLex(
    ("শূন্য", "এক", "দুই", "তিন", "চাৰি", "পাঁচ", "ছয়", "সাত", "আঠ", "নৱ"),
    "ঋণ", "দশমিক",
)

_CARDINAL_FUNCS: dict[str, Callable[[str], str]] = {
    "en": _en_number,
    "hi": _hi_number,
    "gu": _gu_number,
    "kn": _kn_number,
}

_LEXICON_FUNCS: dict[str, Callable[[str], str]] = {
    loc: _lexicon_conv(lex) for loc, lex in _SPOKEN_LEXICONS.items()
}

# ──────────────────────────────────────────────────────────────────────────────
# Script detection → default digit pronunciation locale
# ──────────────────────────────────────────────────────────────────────────────

def _script_of_char(ch: str) -> str:
    try:
        name = unicodedata.name(ch, "")
        for script in (
            "DEVANAGARI", "GUJARATI", "KANNADA", "TELUGU", "TAMIL",
            "BENGALI", "GURMUKHI", "MALAYALAM", "ORIYA", "MYANMAR",
            "THAI", "LAO", "KHMER", "SINHALA", "TIBETAN", "TAI VIET",
            "MEETEI MAYEK", "OL CHIKI", "WARANG CITI",
        ):
            if script in name:
                return script
        if "ARABIC" in name:
            return "ARABIC"
        if ch.isascii() and ch.isalpha():
            return "LATIN"
    except Exception:
        pass
    return "UNKNOWN"


def detect_script(text: str) -> str:
    """Return the dominant script token found in *text* (skips digits/punct)."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isspace() or ch.isdigit() or ch in ".,!?-":
            continue
        s = _script_of_char(ch)
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "LATIN"
    return max(counts, key=counts.__getitem__)


_SCRIPT_DEFAULT_PRONUNCIATION: dict[str, str] = {
    "LATIN": "en",
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "GURMUKHI": "pa",
    "GUJARATI": "gu",
    "ORIYA": "or",
    "TELUGU": "te",
    "KANNADA": "kn",
    "TAMIL": "ta",
    "MALAYALAM": "ml",
    "MYANMAR": "my",
    "THAI": "th",
    "LAO": "lo",
    "KHMER": "km",
    "SINHALA": "si",
    "TIBETAN": "hi",
    "TAI VIET": "vi",
    "MEETEI MAYEK": "hi",
    "OL CHIKI": "hi",
    "WARANG CITI": "hi",
    "ARABIC": "ar",
}

# Scripts where “hinglish” / english_digits hint switches digit words to English.
_INDIC_HINT_SCRIPTS: frozenset[str] = frozenset({
    "DEVANAGARI", "GUJARATI", "KANNADA", "BENGALI", "GURMUKHI", "ORIYA",
    "TELUGU", "TAMIL", "MALAYALAM", "MYANMAR", "THAI", "LAO", "KHMER",
    "SINHALA", "TIBETAN", "MEETEI MAYEK", "OL CHIKI", "WARANG CITI",
})

_NUMBER_GROUP_SEP = r"[,،٬_\u202f\u00a0\u2009\s]"
# Indian-style (…,00,000), Western thousands (…,000,000), or plain integers / decimals.
_NUMBER_RE = re.compile(
    r"-?(?:"
    r"\d{1,3}(?:" + _NUMBER_GROUP_SEP + r"\d{2,3})+"
    r"|\d{1,3}(?:" + _NUMBER_GROUP_SEP + r"\d{3})+"
    r"|\d+)(?:\.\d+)?",
    re.UNICODE,
)


def _normalize_numeric_token(raw: str) -> Optional[str]:
    """Strip lakh/crore/Western-style grouping; return canonical ``-?digits(.digits)?`` or ``None``."""
    t = raw.replace("\u202f", "").replace("\u00a0", "").replace("\u2009", "")
    t = re.sub(r"\s+", "", t)
    t = t.replace("_", "").replace(",", "").replace("،", "").replace("٬", "")
    if not t or t in ("-", ".", "-."):
        return None
    neg = t.startswith("-")
    if neg:
        t = t[1:]
    if "." in t:
        int_part, frac_part = t.split(".", 1)
        if frac_part.count("."):
            return None
        if int_part == "":
            int_part = "0"
        if not int_part.isdigit() or not frac_part.isdigit():
            return None
        out = int_part + "." + frac_part
    else:
        if not t.isdigit():
            return None
        out = t
    return ("-" + out) if neg else out

# ISO-like keys + English names → canonical pronunciation locale id.
_PRONUNCIATION_ALIASES: dict[str, str] = {
    "english": "en", "eng": "en", "latin": "en", "en-us": "en", "en-gb": "en",
    "hindi": "hi", "hin": "hi", "devanagari": "hi",
    "marathi": "hi", "mr": "hi", "san": "hi", "sanskrit": "hi", "sa": "hi",
    "bhojpuri": "hi", "bho": "hi", "magahi": "hi", "mag": "hi", "mai": "hi",
    "awadhi": "hi", "awa": "hi", "rajasthani": "hi", "raj": "hi", "hne": "hi",
    "dogri": "hi", "doi": "hi", "konkani": "hi", "kok": "hi", "gom": "hi",
    "nepali": "ne", "nep": "ne", "ne": "ne",
    "bengali": "bn", "ben": "bn", "bn": "bn",
    "assamese": "as", "asm": "as", "as": "as",
    "tamil": "ta", "tam": "ta", "ta": "ta",
    "telugu": "te", "tel": "te", "te": "te",
    "malayalam": "ml", "mal": "ml", "ml": "ml",
    "panjabi": "pa", "punjabi": "pa", "pan": "pa", "pa": "pa",
    "odia": "or", "orya": "or", "ori": "or", "or": "or",
    "gujarati": "gu", "guj": "gu", "gu": "gu",
    "kannada": "kn", "kan": "kn", "kn": "kn",
    "sinhala": "si", "sin": "si", "si": "si",
    "urdu": "ur", "urd": "ur", "ur": "ur", "ks": "ur", "snd": "ur", "sd": "ur",
    "arabic": "ar", "ara": "ar", "ar": "ar",
    "burmese": "my", "myanmar": "my", "my": "my",
    "thai": "th", "tha": "th", "th": "th",
    "lao": "lo", "lo": "lo",
    "khmer": "km", "khm": "km", "km": "km",
    "spanish": "es", "spa": "es", "es": "es",
    "french": "fr", "fra": "fr", "fre": "fr", "fr": "fr",
    "german": "de", "deu": "de", "ger": "de", "de": "de",
    "portuguese": "pt", "por": "pt", "pt": "pt",
    "italian": "it", "ita": "it", "it": "it",
    "dutch": "nl", "nld": "nl", "nl": "nl",
    "polish": "pl", "pol": "pl", "pl": "pl",
    "russian": "ru", "rus": "ru", "ru": "ru",
    "ukrainian": "uk", "ukr": "uk", "uk": "uk",
    "turkish": "tr", "tur": "tr", "tr": "tr",
    "vietnamese": "vi", "vie": "vi", "vi": "vi",
    "chinese": "zh", "chi": "zh", "zh": "zh", "cmn": "zh",
    "japanese": "ja", "jpn": "ja", "ja": "ja",
    "korean": "ko", "kor": "ko", "ko": "ko",
    "czech": "cs", "ces": "cs", "cze": "cs", "cs": "cs",
    "slovak": "sk", "slk": "sk", "sk": "sk",
    "hungarian": "hu", "hun": "hu", "hu": "hu",
    "romanian": "ro", "ron": "ro", "ro": "ro",
    "swedish": "sv", "swe": "sv", "sv": "sv",
    "danish": "da", "dan": "da", "da": "da",
    "finnish": "fi", "fin": "fi", "fi": "fi",
    "greek": "el", "ell": "el", "el": "el",
    "hebrew": "he", "heb": "he", "he": "he",
    "indonesian": "id", "ind": "id", "id": "id",
    "malay": "ms", "msa": "ms", "ms": "ms",
    "tagalog": "tl", "tgl": "tl", "tl": "tl", "fil": "tl",
    "bodo": "hi", "brx": "hi", "santali": "hi", "sat": "hi",
    "kashmiri": "ur", "kas": "ur",
    "manipuri": "hi", "mni": "hi", "meitei": "hi",
    "khasi": "en", "kha": "en", "garo": "en", "grt": "en",
    "mizo": "en", "lus": "en",
}

# Extra ISO 639-3 codes (India / South Asia) → nearest supported locale.
for _code in (
    "bgq", "bhb", "bjj", "bpy", "btv", "dcc", "hoc", "kfr", "gbm", "gju", "hif",
    "jns", "awa", "bho", "bhr", "bgc", "srx", "sck", "wbr", "xnr", "syl", "hno",
    "bgd", "rkt", "mup", "wtm", "bgw", "sdr", "hnd", "bfy", "kfm", "bgv", "bsh",
    "cwd", "dgo", "dhd", "dml", "gbk", "gda", "gom", "skt", "hne", "bns", "bus",
):
    _PRONUNCIATION_ALIASES.setdefault(_code, "hi")

_KNOWN_PRONUNCIATION_LOCALES: frozenset[str] = frozenset(
    set(_CARDINAL_FUNCS) | set(_LEXICON_FUNCS),
)

_ENGLISH_DIGIT_HINTS = frozenset({
    "en", "english", "hinglish", "english_digits", "latin_digits", "roman_digits",
})


def supported_digit_pronunciations() -> list[str]:
    """Sorted locale ids with built-in digit pronunciation (cardinal or digit-by-digit)."""
    return sorted(_KNOWN_PRONUNCIATION_LOCALES)


def normalize_digit_pronunciation(raw: Optional[str]) -> Optional[str]:
    """Map any alias / ISO-ish tag to a built-in pronunciation locale, or ``None``."""
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("_", "-")
    if not s:
        return None
    if "-" in s:
        s = s.split("-", 1)[0]
    s = _PRONUNCIATION_ALIASES.get(s, s)
    if s in _KNOWN_PRONUNCIATION_LOCALES:
        return s
    return None


def normalize_digit_lang_code(raw: Optional[str]) -> Optional[str]:
    """Backward-compatible: same as :func:`normalize_digit_pronunciation`."""
    return normalize_digit_pronunciation(raw)


def normalize_digit_hint(raw: Optional[str]) -> Optional[str]:
    """Normalize ``digit_words_hint``; returns ``'en'`` for English digit words."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if s in _ENGLISH_DIGIT_HINTS:
        return "en"
    return s


def _converter_for_locale(loc: str) -> Callable[[str], str]:
    if loc in _CARDINAL_FUNCS:
        return _CARDINAL_FUNCS[loc]
    if loc in _LEXICON_FUNCS:
        return _LEXICON_FUNCS[loc]
    return _en_number


@dataclass
class DigitToWordsConverter:
    """Converts digit tokens using *pronunciation* / *lang* / *hint* / script auto-detect."""

    def convert(
        self,
        text: str,
        *,
        pronunciation: Optional[str] = None,
        lang: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> str:
        pron = normalize_digit_pronunciation(pronunciation)
        lang_c = normalize_digit_pronunciation(lang)
        hint_c = normalize_digit_hint(hint)

        if pron:
            conv = _converter_for_locale(pron)
        elif lang_c:
            conv = _converter_for_locale(lang_c)
        else:
            script = detect_script(text)
            loc = _SCRIPT_DEFAULT_PRONUNCIATION.get(script, "en")
            if hint_c == "en" and script in _INDIC_HINT_SCRIPTS:
                conv = _en_number
            else:
                conv = _converter_for_locale(loc)

        def replace_match(m: re.Match[str]) -> str:
            raw = m.group(0)
            canon = _normalize_numeric_token(raw)
            if canon is None:
                return raw
            try:
                return conv(canon)
            except (ValueError, OverflowError):
                return raw

        return _NUMBER_RE.sub(replace_match, text)

    @staticmethod
    def default() -> "DigitToWordsConverter":
        return DigitToWordsConverter()


class DigitToWordService:
    """Process-local digit → spoken-word normaliser for TTS pipelines."""

    __slots__ = ("_converter",)

    def __init__(self, converter: Optional[DigitToWordsConverter] = None) -> None:
        self._converter = converter or DigitToWordsConverter()

    def normalize_for_tts(
        self,
        text: str,
        *,
        digit_pronunciation: Optional[str] = None,
        digit_words_lang: Optional[str] = None,
        digit_words_hint: Optional[str] = None,
    ) -> str:
        """Expand digits to words before synthesis (no-op on empty *text*).

        Precedence: ``digit_pronunciation`` → ``digit_words_lang`` (legacy) →
        script auto-detect, with ``digit_words_hint`` (e.g. *hinglish*) only
        when no explicit pronunciation/locale was given.
        """
        if not text:
            return text
        return self._converter.convert(
            text,
            pronunciation=digit_pronunciation,
            lang=digit_words_lang,
            hint=digit_words_hint,
        )

    def convert_numbers(
        self,
        text: str,
        lang: Optional[str] = None,
        *,
        hint: Optional[str] = None,
        pronunciation: Optional[str] = None,
    ) -> str:
        return self._converter.convert(
            text, pronunciation=pronunciation, lang=lang, hint=hint,
        )


_digit_service: Optional[DigitToWordService] = None


def get_digit_to_word_service() -> DigitToWordService:
    """Shared :class:`DigitToWordService` for this worker process."""
    global _digit_service
    if _digit_service is None:
        _digit_service = DigitToWordService()
    return _digit_service


def normalize_user_text(
    text: str,
    *,
    digit_pronunciation: Optional[str] = None,
    digit_words_lang: Optional[str] = None,
    digit_words_hint: Optional[str] = None,
) -> str:
    """Module-level helper — same as :meth:`DigitToWordService.normalize_for_tts`."""
    return get_digit_to_word_service().normalize_for_tts(
        text,
        digit_pronunciation=digit_pronunciation,
        digit_words_lang=digit_words_lang,
        digit_words_hint=digit_words_hint,
    )


def convert_numbers_to_words(
    text: str,
    lang: Optional[str] = None,
    *,
    hint: Optional[str] = None,
    pronunciation: Optional[str] = None,
) -> str:
    """Replace numbers with words (optional ``hint`` / ``pronunciation``)."""
    return get_digit_to_word_service().convert_numbers(
        text, lang=lang, hint=hint, pronunciation=pronunciation,
    )
