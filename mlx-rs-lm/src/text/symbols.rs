//! Phoneme symbols for GPT-SoVITS
//!
//! This module defines the exact phoneme vocabulary used by GPT-SoVITS.
//! The symbols MUST match the Python implementation exactly for correct encoding.

use std::collections::HashMap;
use std::sync::LazyLock;

/// GPT-SoVITS symbol table (322 symbols)
/// Generated from dora_primespeech.moyoyo_tts.text.symbols
pub const GPT_SOVITS_SYMBOLS: &[&str] = &[
    "!",
    ",",
    "-",
    ".",
    "?",
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "EE",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "En1",
    "En2",
    "En3",
    "En4",
    "En5",
    "F",
    "G",
    "HH",
    "I",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OO",
    "OW0",
    "OW1",
    "OW2",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "SP",
    "SP2",
    "SP3",
    "T",
    "TH",
    "U",
    "UH0",
    "UH1",
    "UH2",
    "UNK",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "_",
    "a",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "ai1",
    "ai2",
    "ai3",
    "ai4",
    "ai5",
    "an1",
    "an2",
    "an3",
    "an4",
    "an5",
    "ang1",
    "ang2",
    "ang3",
    "ang4",
    "ang5",
    "ao1",
    "ao2",
    "ao3",
    "ao4",
    "ao5",
    "b",
    "by",
    "c",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "ei1",
    "ei2",
    "ei3",
    "ei4",
    "ei5",
    "en1",
    "en2",
    "en3",
    "en4",
    "en5",
    "eng1",
    "eng2",
    "eng3",
    "eng4",
    "eng5",
    "er1",
    "er2",
    "er3",
    "er4",
    "er5",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i01",
    "i02",
    "i03",
    "i04",
    "i05",
    "i1",
    "i2",
    "i3",
    "i4",
    "i5",
    "ia1",
    "ia2",
    "ia3",
    "ia4",
    "ia5",
    "ian1",
    "ian2",
    "ian3",
    "ian4",
    "ian5",
    "iang1",
    "iang2",
    "iang3",
    "iang4",
    "iang5",
    "iao1",
    "iao2",
    "iao3",
    "iao4",
    "iao5",
    "ie1",
    "ie2",
    "ie3",
    "ie4",
    "ie5",
    "in1",
    "in2",
    "in3",
    "in4",
    "in5",
    "ing1",
    "ing2",
    "ing3",
    "ing4",
    "ing5",
    "iong1",
    "iong2",
    "iong3",
    "iong4",
    "iong5",
    "ir1",
    "ir2",
    "ir3",
    "ir4",
    "ir5",
    "iu1",
    "iu2",
    "iu3",
    "iu4",
    "iu5",
    "j",
    "k",
    "ky",
    "l",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o1",
    "o2",
    "o3",
    "o4",
    "o5",
    "ong1",
    "ong2",
    "ong3",
    "ong4",
    "ong5",
    "ou1",
    "ou2",
    "ou3",
    "ou4",
    "ou5",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "u",
    "u1",
    "u2",
    "u3",
    "u4",
    "u5",
    "ua1",
    "ua2",
    "ua3",
    "ua4",
    "ua5",
    "uai1",
    "uai2",
    "uai3",
    "uai4",
    "uai5",
    "uan1",
    "uan2",
    "uan3",
    "uan4",
    "uan5",
    "uang1",
    "uang2",
    "uang3",
    "uang4",
    "uang5",
    "ui1",
    "ui2",
    "ui3",
    "ui4",
    "ui5",
    "un1",
    "un2",
    "un3",
    "un4",
    "un5",
    "uo1",
    "uo2",
    "uo3",
    "uo4",
    "uo5",
    "v",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "van1",
    "van2",
    "van3",
    "van4",
    "van5",
    "ve1",
    "ve2",
    "ve3",
    "ve4",
    "ve5",
    "vn1",
    "vn2",
    "vn3",
    "vn4",
    "vn5",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "…"
];

/// Symbol to ID mapping
static SYMBOL_TO_ID: LazyLock<HashMap<&'static str, i32>> = LazyLock::new(|| {
    GPT_SOVITS_SYMBOLS
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as i32))
        .collect()
});

/// ID to symbol mapping  
static ID_TO_SYMBOL: LazyLock<HashMap<i32, &'static str>> = LazyLock::new(|| {
    GPT_SOVITS_SYMBOLS
        .iter()
        .enumerate()
        .map(|(i, &s)| (i as i32, s))
        .collect()
});

/// Get vocabulary size
pub fn vocab_size() -> usize {
    GPT_SOVITS_SYMBOLS.len()
}

/// Get all symbols
pub fn all_symbols() -> &'static [&'static str] {
    GPT_SOVITS_SYMBOLS
}

/// Convert symbol to ID
pub fn symbol_to_id(symbol: &str) -> i32 {
    SYMBOL_TO_ID.get(symbol).copied().unwrap_or(0)  // Return 0 (!) for unknown
}

/// Convert ID to symbol
pub fn id_to_symbol(id: i32) -> &'static str {
    ID_TO_SYMBOL.get(&id).copied().unwrap_or("!")
}

/// Convert list of symbols to IDs
pub fn symbols_to_ids(symbols: &[&str]) -> Vec<i32> {
    symbols.iter().map(|s| symbol_to_id(s)).collect()
}

/// Convert list of IDs to symbols
pub fn ids_to_symbols(ids: &[i32]) -> Vec<&'static str> {
    ids.iter().map(|&id| id_to_symbol(id)).collect()
}

/// Check if symbol exists in vocabulary
pub fn has_symbol(symbol: &str) -> bool {
    SYMBOL_TO_ID.contains_key(symbol)
}

// Special token constants (from GPT-SoVITS symbol table)
pub const PAD: &str = "_";      // Index 95
pub const UNK: &str = "UNK";    // Index 86
pub const SP: &str = "SP";      // Index 77 (short pause)
pub const SP2: &str = "SP2";    // Index 78 (medium pause)
pub const SP3: &str = "SP3";    // Index 79 (long pause)

// BOS/EOS are not in GPT-SoVITS - use SP as boundaries
pub const BOS: &str = "SP";
pub const EOS: &str = "SP";

// Special token ID functions
pub fn pad_id() -> i32 { symbol_to_id(PAD) }  // 95
pub fn unk_id() -> i32 { symbol_to_id(UNK) }  // 86
pub fn bos_id() -> i32 { symbol_to_id(BOS) }  // 77
pub fn eos_id() -> i32 { symbol_to_id(EOS) }  // 77
pub fn sp_id() -> i32 { symbol_to_id(SP) }    // 77

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        assert_eq!(vocab_size(), 322);
    }

    #[test]
    fn test_specific_symbols() {
        // These are the IDs for "你好" phonemes from Python
        assert_eq!(symbol_to_id("n"), 227);
        assert_eq!(symbol_to_id("i3"), 168);
        assert_eq!(symbol_to_id("h"), 158);
        assert_eq!(symbol_to_id("ao3"), 119);
    }

    #[test]
    fn test_roundtrip() {
        let symbols = &["n", "i3", "h", "ao3"];
        let ids = symbols_to_ids(symbols);
        let recovered: Vec<&str> = ids_to_symbols(&ids);
        assert_eq!(symbols.to_vec(), recovered);
    }
}

