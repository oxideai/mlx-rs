//! CMU Pronouncing Dictionary for English G2P
//!
//! This module provides ARPAbet phoneme conversion for English words
//! using a subset of the CMU Pronouncing Dictionary.

use std::collections::HashMap;
use std::sync::LazyLock;

/// CMU dictionary mapping words to ARPAbet phonemes
static CMU_DICT: LazyLock<HashMap<&'static str, &'static [&'static str]>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    // Common words used in mixed Chinese/English text
    // Format: word -> [phonemes]

    // A
    m.insert("a", &["AH0"][..]);
    m.insert("about", &["AH0", "B", "AW1", "T"][..]);
    m.insert("after", &["AE1", "F", "T", "ER0"][..]);
    m.insert("again", &["AH0", "G", "EH1", "N"][..]);
    m.insert("all", &["AO1", "L"][..]);
    m.insert("also", &["AO1", "L", "S", "OW0"][..]);
    m.insert("am", &["AE1", "M"][..]);
    m.insert("an", &["AH0", "N"][..]);
    m.insert("and", &["AH0", "N", "D"][..]);
    m.insert("any", &["EH1", "N", "IY0"][..]);
    m.insert("app", &["AE1", "P"][..]);
    m.insert("apple", &["AE1", "P", "AH0", "L"][..]);
    m.insert("are", &["AA1", "R"][..]);
    m.insert("as", &["AE1", "Z"][..]);
    m.insert("at", &["AE1", "T"][..]);

    // B
    m.insert("baby", &["B", "EY1", "B", "IY0"][..]);
    m.insert("back", &["B", "AE1", "K"][..]);
    m.insert("bad", &["B", "AE1", "D"][..]);
    m.insert("be", &["B", "IY1"][..]);
    m.insert("beautiful", &["B", "Y", "UW1", "T", "AH0", "F", "AH0", "L"][..]);
    m.insert("because", &["B", "IH0", "K", "AO1", "Z"][..]);
    m.insert("been", &["B", "IH1", "N"][..]);
    m.insert("before", &["B", "IH0", "F", "AO1", "R"][..]);
    m.insert("best", &["B", "EH1", "S", "T"][..]);
    m.insert("better", &["B", "EH1", "T", "ER0"][..]);
    m.insert("big", &["B", "IH1", "G"][..]);
    m.insert("book", &["B", "UH1", "K"][..]);
    m.insert("boy", &["B", "OY1"][..]);
    m.insert("bring", &["B", "R", "IH1", "NG"][..]);
    m.insert("but", &["B", "AH1", "T"][..]);
    m.insert("buy", &["B", "AY1"][..]);
    m.insert("by", &["B", "AY1"][..]);

    // C
    m.insert("call", &["K", "AO1", "L"][..]);
    m.insert("can", &["K", "AE1", "N"][..]);
    m.insert("car", &["K", "AA1", "R"][..]);
    m.insert("check", &["CH", "EH1", "K"][..]);
    m.insert("china", &["CH", "AY1", "N", "AH0"][..]);
    m.insert("chinese", &["CH", "AY0", "N", "IY1", "Z"][..]);
    m.insert("city", &["S", "IH1", "T", "IY0"][..]);
    m.insert("close", &["K", "L", "OW1", "Z"][..]);
    m.insert("code", &["K", "OW1", "D"][..]);
    m.insert("coffee", &["K", "AO1", "F", "IY0"][..]);
    m.insert("come", &["K", "AH1", "M"][..]);
    m.insert("computer", &["K", "AH0", "M", "P", "Y", "UW1", "T", "ER0"][..]);
    m.insert("cool", &["K", "UW1", "L"][..]);
    m.insert("could", &["K", "UH1", "D"][..]);

    // D
    m.insert("day", &["D", "EY1"][..]);
    m.insert("did", &["D", "IH1", "D"][..]);
    m.insert("do", &["D", "UW1"][..]);
    m.insert("does", &["D", "AH1", "Z"][..]);
    m.insert("don't", &["D", "OW1", "N", "T"][..]);
    m.insert("down", &["D", "AW1", "N"][..]);

    // E
    m.insert("eat", &["IY1", "T"][..]);
    m.insert("email", &["IY1", "M", "EY2", "L"][..]);
    m.insert("english", &["IH1", "NG", "G", "L", "IH0", "SH"][..]);
    m.insert("even", &["IY1", "V", "AH0", "N"][..]);
    m.insert("every", &["EH1", "V", "R", "IY0"][..]);

    // F
    m.insert("feel", &["F", "IY1", "L"][..]);
    m.insert("find", &["F", "AY1", "N", "D"][..]);
    m.insert("fine", &["F", "AY1", "N"][..]);
    m.insert("first", &["F", "ER1", "S", "T"][..]);
    m.insert("food", &["F", "UW1", "D"][..]);
    m.insert("for", &["F", "AO1", "R"][..]);
    m.insert("friend", &["F", "R", "EH1", "N", "D"][..]);
    m.insert("from", &["F", "R", "AH1", "M"][..]);
    m.insert("fun", &["F", "AH1", "N"][..]);
    m.insert("funny", &["F", "AH1", "N", "IY0"][..]);

    // G
    m.insert("game", &["G", "EY1", "M"][..]);
    m.insert("get", &["G", "EH1", "T"][..]);
    m.insert("girl", &["G", "ER1", "L"][..]);
    m.insert("give", &["G", "IH1", "V"][..]);
    m.insert("go", &["G", "OW1"][..]);
    m.insert("going", &["G", "OW1", "IH0", "NG"][..]);
    m.insert("good", &["G", "UH1", "D"][..]);
    m.insert("got", &["G", "AA1", "T"][..]);
    m.insert("great", &["G", "R", "EY1", "T"][..]);

    // H
    m.insert("had", &["HH", "AE1", "D"][..]);
    m.insert("happy", &["HH", "AE1", "P", "IY0"][..]);
    m.insert("has", &["HH", "AE1", "Z"][..]);
    m.insert("have", &["HH", "AE1", "V"][..]);
    m.insert("he", &["HH", "IY1"][..]);
    m.insert("hello", &["HH", "AH0", "L", "OW1"][..]);
    m.insert("help", &["HH", "EH1", "L", "P"][..]);
    m.insert("her", &["HH", "ER1"][..]);
    m.insert("here", &["HH", "IY1", "R"][..]);
    m.insert("hey", &["HH", "EY1"][..]);
    m.insert("hi", &["HH", "AY1"][..]);
    m.insert("him", &["HH", "IH1", "M"][..]);
    m.insert("his", &["HH", "IH1", "Z"][..]);
    m.insert("home", &["HH", "OW1", "M"][..]);
    m.insert("hot", &["HH", "AA1", "T"][..]);
    m.insert("hotel", &["HH", "OW0", "T", "EH1", "L"][..]);
    m.insert("hour", &["AW1", "ER0"][..]);
    m.insert("house", &["HH", "AW1", "S"][..]);
    m.insert("how", &["HH", "AW1"][..]);

    // I
    m.insert("i", &["AY1"][..]);
    m.insert("idea", &["AY0", "D", "IY1", "AH0"][..]);
    m.insert("if", &["IH1", "F"][..]);
    m.insert("in", &["IH1", "N"][..]);
    m.insert("internet", &["IH1", "N", "T", "ER0", "N", "EH2", "T"][..]);
    m.insert("is", &["IH1", "Z"][..]);
    m.insert("it", &["IH1", "T"][..]);
    m.insert("its", &["IH1", "T", "S"][..]);

    // J
    m.insert("job", &["JH", "AA1", "B"][..]);
    m.insert("just", &["JH", "AH1", "S", "T"][..]);

    // K
    m.insert("kind", &["K", "AY1", "N", "D"][..]);
    m.insert("know", &["N", "OW1"][..]);

    // L
    m.insert("last", &["L", "AE1", "S", "T"][..]);
    m.insert("late", &["L", "EY1", "T"][..]);
    m.insert("let", &["L", "EH1", "T"][..]);
    m.insert("life", &["L", "AY1", "F"][..]);
    m.insert("like", &["L", "AY1", "K"][..]);
    m.insert("little", &["L", "IH1", "T", "AH0", "L"][..]);
    m.insert("live", &["L", "IH1", "V"][..]);
    m.insert("long", &["L", "AO1", "NG"][..]);
    m.insert("look", &["L", "UH1", "K"][..]);
    m.insert("lot", &["L", "AA1", "T"][..]);
    m.insert("love", &["L", "AH1", "V"][..]);

    // M
    m.insert("make", &["M", "EY1", "K"][..]);
    m.insert("man", &["M", "AE1", "N"][..]);
    m.insert("many", &["M", "EH1", "N", "IY0"][..]);
    m.insert("may", &["M", "EY1"][..]);
    m.insert("maybe", &["M", "EY1", "B", "IY0"][..]);
    m.insert("me", &["M", "IY1"][..]);
    m.insert("meet", &["M", "IY1", "T"][..]);
    m.insert("message", &["M", "EH1", "S", "AH0", "JH"][..]);
    m.insert("money", &["M", "AH1", "N", "IY0"][..]);
    m.insert("more", &["M", "AO1", "R"][..]);
    m.insert("morning", &["M", "AO1", "R", "N", "IH0", "NG"][..]);
    m.insert("most", &["M", "OW1", "S", "T"][..]);
    m.insert("movie", &["M", "UW1", "V", "IY0"][..]);
    m.insert("much", &["M", "AH1", "CH"][..]);
    m.insert("music", &["M", "Y", "UW1", "Z", "IH0", "K"][..]);
    m.insert("must", &["M", "AH1", "S", "T"][..]);
    m.insert("my", &["M", "AY1"][..]);

    // N
    m.insert("name", &["N", "EY1", "M"][..]);
    m.insert("need", &["N", "IY1", "D"][..]);
    m.insert("never", &["N", "EH1", "V", "ER0"][..]);
    m.insert("new", &["N", "UW1"][..]);
    m.insert("next", &["N", "EH1", "K", "S", "T"][..]);
    m.insert("nice", &["N", "AY1", "S"][..]);
    m.insert("night", &["N", "AY1", "T"][..]);
    m.insert("no", &["N", "OW1"][..]);
    m.insert("not", &["N", "AA1", "T"][..]);
    m.insert("nothing", &["N", "AH1", "TH", "IH0", "NG"][..]);
    m.insert("now", &["N", "AW1"][..]);
    m.insert("number", &["N", "AH1", "M", "B", "ER0"][..]);

    // O
    m.insert("of", &["AH1", "V"][..]);
    m.insert("off", &["AO1", "F"][..]);
    m.insert("office", &["AO1", "F", "AH0", "S"][..]);
    m.insert("oh", &["OW1"][..]);
    m.insert("ok", &["OW2", "K", "EY1"][..]);
    m.insert("okay", &["OW2", "K", "EY1"][..]);
    m.insert("old", &["OW1", "L", "D"][..]);
    m.insert("on", &["AA1", "N"][..]);
    m.insert("one", &["W", "AH1", "N"][..]);
    m.insert("only", &["OW1", "N", "L", "IY0"][..]);
    m.insert("open", &["OW1", "P", "AH0", "N"][..]);
    m.insert("or", &["AO1", "R"][..]);
    m.insert("other", &["AH1", "DH", "ER0"][..]);
    m.insert("our", &["AW1", "ER0"][..]);
    m.insert("out", &["AW1", "T"][..]);
    m.insert("over", &["OW1", "V", "ER0"][..]);
    m.insert("own", &["OW1", "N"][..]);

    // P
    m.insert("party", &["P", "AA1", "R", "T", "IY0"][..]);
    m.insert("people", &["P", "IY1", "P", "AH0", "L"][..]);
    m.insert("phone", &["F", "OW1", "N"][..]);
    m.insert("photo", &["F", "OW1", "T", "OW0"][..]);
    m.insert("picture", &["P", "IH1", "K", "CH", "ER0"][..]);
    m.insert("place", &["P", "L", "EY1", "S"][..]);
    m.insert("play", &["P", "L", "EY1"][..]);
    m.insert("please", &["P", "L", "IY1", "Z"][..]);
    m.insert("point", &["P", "OY1", "N", "T"][..]);
    m.insert("price", &["P", "R", "AY1", "S"][..]);
    m.insert("problem", &["P", "R", "AA1", "B", "L", "AH0", "M"][..]);
    m.insert("put", &["P", "UH1", "T"][..]);

    // Q
    m.insert("question", &["K", "W", "EH1", "S", "CH", "AH0", "N"][..]);
    m.insert("quite", &["K", "W", "AY1", "T"][..]);

    // R
    m.insert("read", &["R", "IY1", "D"][..]);
    m.insert("ready", &["R", "EH1", "D", "IY0"][..]);
    m.insert("real", &["R", "IY1", "L"][..]);
    m.insert("really", &["R", "IY1", "L", "IY0"][..]);
    m.insert("restaurant", &["R", "EH1", "S", "T", "ER0", "AA2", "N", "T"][..]);
    m.insert("resturant", &["R", "EH1", "S", "T", "ER0", "AA2", "N", "T"][..]);  // Common misspelling
    m.insert("right", &["R", "AY1", "T"][..]);
    m.insert("run", &["R", "AH1", "N"][..]);

    // S
    m.insert("sad", &["S", "AE1", "D"][..]);
    m.insert("said", &["S", "EH1", "D"][..]);
    m.insert("salad", &["S", "AE1", "L", "AH0", "D"][..]);
    m.insert("same", &["S", "EY1", "M"][..]);
    m.insert("say", &["S", "EY1"][..]);
    m.insert("school", &["S", "K", "UW1", "L"][..]);
    m.insert("see", &["S", "IY1"][..]);
    m.insert("she", &["SH", "IY1"][..]);
    m.insert("shop", &["SH", "AA1", "P"][..]);
    m.insert("shopping", &["SH", "AA1", "P", "IH0", "NG"][..]);
    m.insert("short", &["SH", "AO1", "R", "T"][..]);
    m.insert("should", &["SH", "UH1", "D"][..]);
    m.insert("show", &["SH", "OW1"][..]);
    m.insert("small", &["S", "M", "AO1", "L"][..]);
    m.insert("so", &["S", "OW1"][..]);
    m.insert("some", &["S", "AH1", "M"][..]);
    m.insert("something", &["S", "AH1", "M", "TH", "IH0", "NG"][..]);
    m.insert("sorry", &["S", "AA1", "R", "IY0"][..]);
    m.insert("sound", &["S", "AW1", "N", "D"][..]);
    m.insert("speak", &["S", "P", "IY1", "K"][..]);
    m.insert("steak", &["S", "T", "EY1", "K"][..]);
    m.insert("still", &["S", "T", "IH1", "L"][..]);
    m.insert("stop", &["S", "T", "AA1", "P"][..]);
    m.insert("store", &["S", "T", "AO1", "R"][..]);
    m.insert("story", &["S", "T", "AO1", "R", "IY0"][..]);
    m.insert("student", &["S", "T", "UW1", "D", "AH0", "N", "T"][..]);
    m.insert("study", &["S", "T", "AH1", "D", "IY0"][..]);
    m.insert("such", &["S", "AH1", "CH"][..]);
    m.insert("super", &["S", "UW1", "P", "ER0"][..]);
    m.insert("sure", &["SH", "UH1", "R"][..]);

    // T
    m.insert("take", &["T", "EY1", "K"][..]);
    m.insert("talk", &["T", "AO1", "K"][..]);
    m.insert("tell", &["T", "EH1", "L"][..]);
    m.insert("test", &["T", "EH1", "S", "T"][..]);
    m.insert("than", &["DH", "AE1", "N"][..]);
    m.insert("thank", &["TH", "AE1", "NG", "K"][..]);
    m.insert("thanks", &["TH", "AE1", "NG", "K", "S"][..]);
    m.insert("that", &["DH", "AE1", "T"][..]);
    m.insert("the", &["DH", "AH0"][..]);
    m.insert("their", &["DH", "EH1", "R"][..]);
    m.insert("them", &["DH", "EH1", "M"][..]);
    m.insert("then", &["DH", "EH1", "N"][..]);
    m.insert("there", &["DH", "EH1", "R"][..]);
    m.insert("these", &["DH", "IY1", "Z"][..]);
    m.insert("they", &["DH", "EY1"][..]);
    m.insert("thing", &["TH", "IH1", "NG"][..]);
    m.insert("think", &["TH", "IH1", "NG", "K"][..]);
    m.insert("this", &["DH", "IH1", "S"][..]);
    m.insert("those", &["DH", "OW1", "Z"][..]);
    m.insert("through", &["TH", "R", "UW1"][..]);
    m.insert("time", &["T", "AY1", "M"][..]);
    m.insert("to", &["T", "UW1"][..]);
    m.insert("today", &["T", "AH0", "D", "EY1"][..]);
    m.insert("together", &["T", "AH0", "G", "EH1", "DH", "ER0"][..]);
    m.insert("tomorrow", &["T", "AH0", "M", "AA1", "R", "OW0"][..]);
    m.insert("tonight", &["T", "AH0", "N", "AY1", "T"][..]);
    m.insert("too", &["T", "UW1"][..]);
    m.insert("top", &["T", "AA1", "P"][..]);
    m.insert("try", &["T", "R", "AY1"][..]);
    m.insert("turn", &["T", "ER1", "N"][..]);
    m.insert("tv", &["T", "IY1", "V", "IY1"][..]);
    m.insert("two", &["T", "UW1"][..]);

    // U
    m.insert("understand", &["AH2", "N", "D", "ER0", "S", "T", "AE1", "N", "D"][..]);
    m.insert("up", &["AH1", "P"][..]);
    m.insert("us", &["AH1", "S"][..]);
    m.insert("use", &["Y", "UW1", "Z"][..]);

    // V
    m.insert("vegetable", &["V", "EH1", "JH", "T", "AH0", "B", "AH0", "L"][..]);
    m.insert("very", &["V", "EH1", "R", "IY0"][..]);
    m.insert("video", &["V", "IH1", "D", "IY0", "OW0"][..]);

    // W
    m.insert("wait", &["W", "EY1", "T"][..]);
    m.insert("walk", &["W", "AO1", "K"][..]);
    m.insert("want", &["W", "AA1", "N", "T"][..]);
    m.insert("was", &["W", "AA1", "Z"][..]);
    m.insert("watch", &["W", "AA1", "CH"][..]);
    m.insert("water", &["W", "AO1", "T", "ER0"][..]);
    m.insert("way", &["W", "EY1"][..]);
    m.insert("we", &["W", "IY1"][..]);
    m.insert("week", &["W", "IY1", "K"][..]);
    m.insert("weekend", &["W", "IY1", "K", "EH2", "N", "D"][..]);
    m.insert("well", &["W", "EH1", "L"][..]);
    m.insert("were", &["W", "ER1"][..]);
    m.insert("what", &["W", "AH1", "T"][..]);
    m.insert("when", &["W", "EH1", "N"][..]);
    m.insert("where", &["W", "EH1", "R"][..]);
    m.insert("which", &["W", "IH1", "CH"][..]);
    m.insert("while", &["W", "AY1", "L"][..]);
    m.insert("who", &["HH", "UW1"][..]);
    m.insert("why", &["W", "AY1"][..]);
    m.insert("will", &["W", "IH1", "L"][..]);
    m.insert("with", &["W", "IH1", "DH"][..]);
    m.insert("without", &["W", "IH0", "TH", "AW1", "T"][..]);
    m.insert("woman", &["W", "UH1", "M", "AH0", "N"][..]);
    m.insert("women", &["W", "IH1", "M", "AH0", "N"][..]);
    m.insert("word", &["W", "ER1", "D"][..]);
    m.insert("work", &["W", "ER1", "K"][..]);
    m.insert("world", &["W", "ER1", "L", "D"][..]);
    m.insert("would", &["W", "UH1", "D"][..]);
    m.insert("wow", &["W", "AW1"][..]);
    m.insert("write", &["R", "AY1", "T"][..]);
    m.insert("wrong", &["R", "AO1", "NG"][..]);

    // X (limited)

    // Y
    m.insert("yeah", &["Y", "AE1"][..]);
    m.insert("year", &["Y", "IH1", "R"][..]);
    m.insert("yes", &["Y", "EH1", "S"][..]);
    m.insert("yesterday", &["Y", "EH1", "S", "T", "ER0", "D", "EY2"][..]);
    m.insert("yet", &["Y", "EH1", "T"][..]);
    m.insert("you", &["Y", "UW1"][..]);
    m.insert("young", &["Y", "AH1", "NG"][..]);
    m.insert("your", &["Y", "AO1", "R"][..]);

    // Z
    m.insert("zero", &["Z", "IY1", "R", "OW0"][..]);

    m
});

/// Look up word in CMU dictionary
pub fn lookup(word: &str) -> Option<&'static [&'static str]> {
    CMU_DICT.get(word.to_lowercase().as_str()).copied()
}

/// Convert English word to ARPAbet phonemes
/// Falls back to letter spelling if word not in dictionary
pub fn word_to_phonemes(word: &str) -> Vec<String> {
    if let Some(phonemes) = lookup(word) {
        phonemes.iter().map(|s| s.to_string()).collect()
    } else {
        // Fallback: spell out letters
        word.chars()
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_uppercase().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_words() {
        assert_eq!(lookup("movie"), Some(&["M", "UW1", "V", "IY0"][..]));
        assert_eq!(lookup("get"), Some(&["G", "EH1", "T"][..]));
        assert_eq!(lookup("point"), Some(&["P", "OY1", "N", "T"][..]));
        assert_eq!(lookup("hello"), Some(&["HH", "AH0", "L", "OW1"][..]));
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(lookup("MOVIE"), Some(&["M", "UW1", "V", "IY0"][..]));
        assert_eq!(lookup("Movie"), Some(&["M", "UW1", "V", "IY0"][..]));
    }

    #[test]
    fn test_unknown_word() {
        assert_eq!(lookup("asdfghjkl"), None);
    }
}
