"""Trilingual (English / Hindi / Gujarati) strings for the forensic export report.

Two deliberate rules keep the translated report defensible as evidence:

1. Only *fixed* text is translated — section headings, field labels, and the standing
   integrity statements. Every translation here is a static, reviewable constant. No
   machine translation ever runs during an export, so the same input always produces
   byte-identical report text.
2. *Recorded facts* are reproduced verbatim in every language section: case/export IDs,
   the operator's original query, filters, hashes, camera ids, and timestamps. Translating
   an officer's query would misstate what was actually searched.

Detector vocabulary (`entity_type`, `subtype`, `color`) is the one exception to rule 2: it
is a closed set produced by our own models, so it is shown as "<translated> (<original>)"
— readable in the local language while keeping the machine-emitted token auditable.
"""

from __future__ import annotations

LANGUAGES: tuple[str, ...] = ("en", "hi", "gu")

# Endonyms — a Gujarati reader looks for "ગુજરાતી", not "Gujarati".
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "gu": "ગુજરાતી (Gujarati)",
}

# Script per language, so the report picks a font that can actually shape the glyphs.
LANGUAGE_SCRIPT: dict[str, str] = {
    "en": "latin",
    "hi": "devanagari",
    "gu": "gujarati",
}

_STRINGS: dict[str, dict[str, str]] = {
    "report_title": {
        "en": "CCTV FORENSIC EXPORT REPORT",
        "hi": "सीसीटीवी फ़ॉरेंसिक निर्यात रिपोर्ट",
        "gu": "સીસીટીવી ફોરેન્સિક નિકાસ અહેવાલ",
    },
    "report_subtitle": {
        "en": "Tamper-evident evidence package",
        "hi": "छेड़छाड़-प्रकट साक्ष्य पैकेज",
        "gu": "ચેડાં-પ્રગટ પુરાવા પૅકેજ",
    },
    "translation_note": {
        "en": (
            "Translations cover fixed labels only. Identifiers, hashes, filters, timestamps "
            "and the operator's original query are reproduced verbatim in every language."
        ),
        "hi": (
            "अनुवाद केवल निश्चित लेबल तक सीमित है। पहचानकर्ता, हैश, फ़िल्टर, समय-चिह्न और "
            "संचालक की मूल क्वेरी प्रत्येक भाषा में अक्षरशः दोहराए गए हैं।"
        ),
        "gu": (
            "અનુવાદ ફક્ત નિશ્ચિત લેબલ પૂરતો છે. ઓળખકર્તા, હૅશ, ફિલ્ટર, સમય-ચિહ્ન અને "
            "સંચાલકની મૂળ ક્વેરી દરેક ભાષામાં અક્ષરશઃ પુનરાવર્તિત છે."
        ),
    },
    # ---- header block -------------------------------------------------------
    "case_id": {"en": "Case ID", "hi": "केस आईडी", "gu": "કેસ આઈડી"},
    "export_id": {"en": "Export ID", "hi": "निर्यात आईडी", "gu": "નિકાસ આઈડી"},
    "officer": {
        "en": "Officer / user",
        "hi": "अधिकारी / उपयोगकर्ता",
        "gu": "અધિકારી / વપરાશકર્તા",
    },
    "report_generated": {
        "en": "Report generated",
        "hi": "रिपोर्ट बनाई गई",
        "gu": "અહેવાલ બનાવ્યો",
    },
    # ---- search block -------------------------------------------------------
    "section_search": {"en": "SEARCH", "hi": "खोज", "gu": "શોધ"},
    "query": {"en": "Query", "hi": "खोज क्वेरी", "gu": "શોધ ક્વેરી"},
    "query_absent": {
        "en": "[not supplied]",
        "hi": "[उपलब्ध नहीं]",
        "gu": "[ઉપલબ્ધ નથી]",
    },
    "filters": {"en": "Filters", "hi": "फ़िल्टर", "gu": "ફિલ્ટર"},
    "retrieval_method": {
        "en": "Retrieval method",
        "hi": "पुनर्प्राप्ति विधि",
        "gu": "પુનઃપ્રાપ્તિ પદ્ધતિ",
    },
    "searched_at": {
        "en": "Search performed",
        "hi": "खोज की गई",
        "gu": "શોધ કરવામાં આવી",
    },
    # ---- evidence block -----------------------------------------------------
    "section_evidence": {"en": "EVIDENCE", "hi": "साक्ष्य", "gu": "પુરાવા"},
    "tracklet": {"en": "Tracklet", "hi": "ट्रैकलेट", "gu": "ટ્રૅકલેટ"},
    "camera": {"en": "Camera", "hi": "कैमरा", "gu": "કૅમેરા"},
    "scene": {"en": "Scene", "hi": "दृश्य", "gu": "દૃશ્ય"},
    "sighting_start": {
        "en": "Sighting start",
        "hi": "दृश्यांकन प्रारंभ",
        "gu": "દૃશ્યાંકન પ્રારંભ",
    },
    "sighting_end": {
        "en": "Sighting end",
        "hi": "दृश्यांकन समाप्त",
        "gu": "દૃશ્યાંકન સમાપ્ત",
    },
    "duration": {"en": "Duration", "hi": "अवधि", "gu": "અવધિ"},
    "seconds_short": {"en": "s", "hi": "से", "gu": "સે"},
    "video_offset": {
        "en": "Position in source file",
        "hi": "स्रोत फ़ाइल में स्थिति",
        "gu": "સ્ત્રોત ફાઇલમાં સ્થાન",
    },
    "entity": {"en": "Entity", "hi": "वस्तु", "gu": "વસ્તુ"},
    "plate": {
        "en": "Number plate",
        "hi": "नंबर प्लेट",
        "gu": "નંબર પ્લેટ",
    },
    "source_sha256": {
        "en": "Source SHA-256",
        "hi": "स्रोत SHA-256",
        "gu": "સ્ત્રોત SHA-256",
    },
    # ---- integrity block ----------------------------------------------------
    "section_integrity": {"en": "INTEGRITY", "hi": "अखंडता", "gu": "અખંડિતતા"},
    "integrity_source": {
        "en": "original_or_source_clip.mp4 is an unannotated byte-for-byte copy.",
        "hi": "original_or_source_clip.mp4 बिना किसी चिह्न के बाइट-दर-बाइट प्रतिलिपि है।",
        "gu": "original_or_source_clip.mp4 કોઈ ચિહ્ન વગરની બાઇટ-દર-બાઇટ નકલ છે.",
    },
    "integrity_derived": {
        "en": "selected_clip.mp4 and annotated_frame.jpg are derived review artifacts.",
        "hi": "selected_clip.mp4 और annotated_frame.jpg व्युत्पन्न समीक्षा कलाकृतियाँ हैं।",
        "gu": "selected_clip.mp4 અને annotated_frame.jpg વ્યુત્પન્ન સમીક્ષા કલાકૃતિઓ છે.",
    },
    "integrity_verify": {
        "en": (
            "Run the verifier against this ZIP. A valid Ed25519 signature and matching "
            "SHA-256 digests are required before the package is reported as VALID."
        ),
        "hi": (
            "इस ZIP पर सत्यापनकर्ता चलाएँ। पैकेज को VALID बताने से पहले वैध Ed25519 "
            "हस्ताक्षर और मेल खाते SHA-256 डाइजेस्ट आवश्यक हैं।"
        ),
        "gu": (
            "આ ZIP પર ચકાસણીકર્તા ચલાવો. પૅકેજને VALID જાહેર કરતાં પહેલાં માન્ય Ed25519 "
            "સહી અને મેળ ખાતા SHA-256 ડાયજેસ્ટ જરૂરી છે."
        ),
    },
    "signing_key": {
        "en": "Signing key",
        "hi": "हस्ताक्षर कुंजी",
        "gu": "સહી કી",
    },
    # ---- timestamp provenance ----------------------------------------------
    "clock_note": {
        "en": (
            "Times are the camera wall clock in {tz}. The recording date is a deployment "
            "setting, not metadata read from the source file."
        ),
        "hi": (
            "समय {tz} में कैमरे की दीवार-घड़ी के अनुसार है। रिकॉर्डिंग की तारीख़ एक "
            "परिनियोजन सेटिंग है, स्रोत फ़ाइल से पढ़ा गया मेटाडेटा नहीं।"
        ),
        "gu": (
            "સમય {tz} માં કૅમેરાની દીવાલ-ઘડિયાળ મુજબ છે. રેકોર્ડિંગની તારીખ એ ડિપ્લોયમેન્ટ "
            "સેટિંગ છે, સ્ત્રોત ફાઇલમાંથી વાંચેલો મેટાડેટા નથી."
        ),
    },
    "page_of": {"en": "Page", "hi": "पृष्ठ", "gu": "પૃષ્ઠ"},
}

# Closed detector vocabulary. Missing keys fall back to the raw token, which is the safe
# behaviour: a new model class shows as-is rather than silently disappearing.
_VOCAB: dict[str, dict[str, str]] = {
    # entity_type
    "vehicle": {"hi": "वाहन", "gu": "વાહન"},
    "person": {"hi": "व्यक्ति", "gu": "વ્યક્તિ"},
    # vehicle subtypes (UVH-26 / COCO classes seen in this deployment)
    "bicycle": {"hi": "साइकिल", "gu": "સાયકલ"},
    "bus": {"hi": "बस", "gu": "બસ"},
    "car": {"hi": "कार", "gu": "કાર"},
    "hatchback": {"hi": "हैचबैक", "gu": "હૅચબૅક"},
    "lcv": {"hi": "हल्का वाणिज्यिक वाहन", "gu": "હલકું વાણિજ્યિક વાહન"},
    "muv": {"hi": "बहुउपयोगी वाहन", "gu": "બહુઉપયોગી વાહન"},
    "sedan": {"hi": "सेडान", "gu": "સેડાન"},
    "suv": {"hi": "एसयूवी", "gu": "એસયુવી"},
    "tempo_traveller": {"hi": "टेम्पो ट्रैवलर", "gu": "ટેમ્પો ટ્રાવેલર"},
    "three_wheeler": {"hi": "तिपहिया / ऑटो रिक्शा", "gu": "ત્રિચક્રી / ઑટો રિક્ષા"},
    "truck": {"hi": "ट्रक", "gu": "ટ્રક"},
    "two_wheeler": {"hi": "दोपहिया", "gu": "દ્વિચક્રી"},
    "van": {"hi": "वैन", "gu": "વૅન"},
    # colors
    "black": {"hi": "काला", "gu": "કાળો"},
    "blue": {"hi": "नीला", "gu": "વાદળી"},
    "brown": {"hi": "भूरा", "gu": "કથ્થઈ"},
    "gray": {"hi": "स्लेटी", "gu": "ભૂખરો"},
    "green": {"hi": "हरा", "gu": "લીલો"},
    "orange": {"hi": "नारंगी", "gu": "નારંગી"},
    "red": {"hi": "लाल", "gu": "લાલ"},
    "silver": {"hi": "चाँदी", "gu": "ચાંદી"},
    "white": {"hi": "सफ़ेद", "gu": "સફેદ"},
    "yellow": {"hi": "पीला", "gu": "પીળો"},
}


def t(key: str, lang: str) -> str:
    """Fixed label in `lang`, falling back to English rather than raising."""
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry["en"]


def term(token: str | None, lang: str) -> str:
    """One detector token in `lang`, falling back to the raw token.

    Falling back to the raw token means an untranslated new model class degrades to plain
    English instead of vanishing from the report.
    """
    if not token:
        return ""
    raw = str(token)
    if lang == "en":
        return raw
    return _VOCAB.get(raw.lower(), {}).get(lang) or raw


def entity_phrase(
    color: str | None, subtype: str | None, entity_type: str | None, lang: str,
) -> str:
    """'<colour> <subtype> (<entity type>)', with the raw tokens appended once.

    The raw English tokens are always shown so the report never hides what the model
    actually emitted, while avoiding a per-word '(token)' suffix that nests badly.
    """
    def phrase(language: str) -> str:
        head = " ".join(p for p in (term(color, language), term(subtype, language)) if p)
        tail = term(entity_type, language)
        return f"{head} ({tail})".strip() if tail else head

    localized = phrase(lang)
    original = phrase("en")
    return localized if lang == "en" or localized == original else f"{localized}  ·  {original}"
