//! CMU Pronouncing Dictionary for English G2P
//!
//! Loads the full CMU dictionary (134K+ words) from cmudict.rep

use std::collections::HashMap;
use std::sync::LazyLock;

/// CMU dictionary loaded from cmudict.rep file
static CMU_DICT: LazyLock<HashMap<String, Vec<Vec<String>>>> = LazyLock::new(|| {
    let dict_content = include_str!("cmudict.rep");
    parse_cmudict(dict_content)
});

/// Parse CMU dictionary format: "WORD  PH1 PH2 PH3"
fn parse_cmudict(content: &str) -> HashMap<String, Vec<Vec<String>>> {
    let mut dict: HashMap<String, Vec<Vec<String>>> = HashMap::new();

    for line in content.lines() {
        // Skip comments
        if line.starts_with(";;;") || line.is_empty() {
            continue;
        }

        // Format: "WORD  PH1 PH2 PH3" (two spaces between word and phonemes)
        if let Some(idx) = line.find("  ") {
            let word = line[..idx].to_lowercase();
            let phonemes: Vec<String> = line[idx+2..]
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            if !phonemes.is_empty() {
                // Handle alternate pronunciations: WORD(1), WORD(2), etc.
                let base_word = if let Some(paren_idx) = word.find('(') {
                    word[..paren_idx].to_string()
                } else {
                    word.clone()
                };

                dict.entry(base_word).or_insert_with(Vec::new).push(phonemes);
            }
        }
    }

    dict
}

/// Look up a word in the CMU dictionary
/// Returns the first pronunciation if found
pub fn lookup(word: &str) -> Option<Vec<String>> {
    let word_lower = word.to_lowercase();
    CMU_DICT.get(&word_lower).and_then(|prons| prons.first().cloned())
}

/// Convert English word to ARPAbet phonemes
/// Falls back to rule-based G2P if word not in dictionary
pub fn word_to_phonemes(word: &str) -> Vec<String> {
    if let Some(phonemes) = lookup(word) {
        phonemes
    } else {
        // Try rule-based G2P for unknown words
        rule_based_g2p(word)
    }
}

/// Simple rule-based G2P for unknown English words
/// Handles common letter patterns and produces reasonable phonemes
fn rule_based_g2p(word: &str) -> Vec<String> {
    let word = word.to_lowercase();
    let chars: Vec<char> = word.chars().collect();
    let mut phonemes = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let next2 = chars.get(i + 2).copied();

        // Multi-character patterns first
        match (c, next, next2) {
            // Three-character patterns
            ('t', Some('c'), Some('h')) => { phonemes.push("CH".to_string()); i += 3; continue; }
            ('s', Some('c'), Some('h')) => { phonemes.push("SH".to_string()); i += 3; continue; }
            ('t', Some('i'), Some('o')) => { phonemes.push("SH".to_string()); phonemes.push("AH0".to_string()); i += 3; continue; }
            ('o', Some('u'), Some('s')) if i + 3 == chars.len() => { phonemes.push("AH0".to_string()); phonemes.push("S".to_string()); i += 3; continue; }
            _ => {}
        }

        match (c, next) {
            // Two-character consonant patterns
            ('c', Some('h')) => { phonemes.push("CH".to_string()); i += 2; continue; }
            ('s', Some('h')) => { phonemes.push("SH".to_string()); i += 2; continue; }
            ('t', Some('h')) => { phonemes.push("TH".to_string()); i += 2; continue; }
            ('p', Some('h')) => { phonemes.push("F".to_string()); i += 2; continue; }
            ('w', Some('h')) => { phonemes.push("W".to_string()); i += 2; continue; }
            ('c', Some('k')) => { phonemes.push("K".to_string()); i += 2; continue; }
            ('n', Some('g')) => { phonemes.push("NG".to_string()); i += 2; continue; }
            ('g', Some('h')) => { i += 2; continue; } // silent gh
            ('k', Some('n')) => { phonemes.push("N".to_string()); i += 2; continue; } // silent k in kn
            ('w', Some('r')) => { phonemes.push("R".to_string()); i += 2; continue; } // silent w in wr

            // Two-character vowel patterns
            ('a', Some('i')) | ('a', Some('y')) => { phonemes.push("EY1".to_string()); i += 2; continue; }
            ('e', Some('a')) => { phonemes.push("IY1".to_string()); i += 2; continue; }
            ('e', Some('e')) => { phonemes.push("IY1".to_string()); i += 2; continue; }
            ('o', Some('o')) => { phonemes.push("UW1".to_string()); i += 2; continue; }
            ('o', Some('u')) => { phonemes.push("AW1".to_string()); i += 2; continue; }
            ('o', Some('w')) => { phonemes.push("OW1".to_string()); i += 2; continue; }
            ('o', Some('i')) | ('o', Some('y')) => { phonemes.push("OY1".to_string()); i += 2; continue; }
            ('a', Some('u')) | ('a', Some('w')) => { phonemes.push("AO1".to_string()); i += 2; continue; }
            ('e', Some('w')) => { phonemes.push("UW1".to_string()); i += 2; continue; }
            ('i', Some('e')) => { phonemes.push("IY1".to_string()); i += 2; continue; }
            ('e', Some('i')) | ('e', Some('y')) => { phonemes.push("EY1".to_string()); i += 2; continue; }

            // Double consonants - just use single sound
            ('s', Some('s')) => { phonemes.push("S".to_string()); i += 2; continue; }
            ('t', Some('t')) => { phonemes.push("T".to_string()); i += 2; continue; }
            ('l', Some('l')) => { phonemes.push("L".to_string()); i += 2; continue; }
            ('f', Some('f')) => { phonemes.push("F".to_string()); i += 2; continue; }
            ('r', Some('r')) => { phonemes.push("R".to_string()); i += 2; continue; }
            ('n', Some('n')) => { phonemes.push("N".to_string()); i += 2; continue; }
            ('m', Some('m')) => { phonemes.push("M".to_string()); i += 2; continue; }
            ('p', Some('p')) => { phonemes.push("P".to_string()); i += 2; continue; }
            ('b', Some('b')) => { phonemes.push("B".to_string()); i += 2; continue; }
            ('d', Some('d')) => { phonemes.push("D".to_string()); i += 2; continue; }
            ('g', Some('g')) => { phonemes.push("G".to_string()); i += 2; continue; }

            _ => {}
        }

        // Single character patterns
        match c {
            // Consonants
            'b' => phonemes.push("B".to_string()),
            'd' => phonemes.push("D".to_string()),
            'f' => phonemes.push("F".to_string()),
            'g' => phonemes.push("G".to_string()),
            'h' => phonemes.push("HH".to_string()),
            'j' => phonemes.push("JH".to_string()),
            'k' => phonemes.push("K".to_string()),
            'l' => phonemes.push("L".to_string()),
            'm' => phonemes.push("M".to_string()),
            'n' => phonemes.push("N".to_string()),
            'p' => phonemes.push("P".to_string()),
            'q' => phonemes.push("K".to_string()),
            'r' => phonemes.push("R".to_string()),
            's' => phonemes.push("S".to_string()),
            't' => phonemes.push("T".to_string()),
            'v' => phonemes.push("V".to_string()),
            'w' => phonemes.push("W".to_string()),
            'x' => { phonemes.push("K".to_string()); phonemes.push("S".to_string()); }
            'z' => phonemes.push("Z".to_string()),

            // C depends on following vowel
            'c' => {
                if matches!(next, Some('e') | Some('i') | Some('y')) {
                    phonemes.push("S".to_string());
                } else {
                    phonemes.push("K".to_string());
                }
            }

            // Vowels - context dependent
            'a' => {
                // Check for magic-e pattern (a_e)
                if next.map(|n| n.is_ascii_alphabetic() && n != 'e').unwrap_or(false)
                    && next2 == Some('e')
                    && i + 3 >= chars.len()
                {
                    phonemes.push("EY1".to_string());
                } else {
                    phonemes.push("AE1".to_string());
                }
            }
            'e' => {
                // Silent e at end
                if i + 1 == chars.len() && !phonemes.is_empty() {
                    // skip silent e
                } else {
                    phonemes.push("EH1".to_string());
                }
            }
            'i' => {
                // Check for magic-e pattern (i_e)
                if next.map(|n| n.is_ascii_alphabetic() && n != 'e').unwrap_or(false)
                    && next2 == Some('e')
                    && i + 3 >= chars.len()
                {
                    phonemes.push("AY1".to_string());
                } else {
                    phonemes.push("IH1".to_string());
                }
            }
            'o' => {
                // Check for magic-e pattern (o_e)
                if next.map(|n| n.is_ascii_alphabetic() && n != 'e').unwrap_or(false)
                    && next2 == Some('e')
                    && i + 3 >= chars.len()
                {
                    phonemes.push("OW1".to_string());
                } else {
                    phonemes.push("AA1".to_string());
                }
            }
            'u' => {
                phonemes.push("AH1".to_string());
            }
            'y' => {
                // Y at start is consonant, otherwise vowel
                if i == 0 {
                    phonemes.push("Y".to_string());
                } else {
                    phonemes.push("IY1".to_string());
                }
            }
            _ => {} // Skip non-alphabetic
        }
        i += 1;
    }

    if phonemes.is_empty() {
        // Ultimate fallback: spell out letters
        word.chars()
            .filter(|c| c.is_ascii_alphabetic())
            .filter_map(|c| lookup(&c.to_string()))
            .flatten()
            .collect()
    } else {
        phonemes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_words() {
        assert!(lookup("hello").is_some());
        assert!(lookup("world").is_some());
        assert!(lookup("economist").is_some());
        assert!(lookup("commercial").is_some());
        assert!(lookup("agricultural").is_some());
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(lookup("HELLO"), lookup("hello"));
        assert_eq!(lookup("Hello"), lookup("hello"));
    }

    #[test]
    fn test_word_to_phonemes() {
        let phonemes = word_to_phonemes("economist");
        assert!(!phonemes.is_empty());
        assert!(phonemes.contains(&"K".to_string()) || phonemes.contains(&"IH0".to_string()));
    }
}
