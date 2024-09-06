use std::{collections::HashMap, fmt::Display, rc::Rc};

const DELIMITER: char = '.';

#[derive(Debug)]
pub enum NestedValue<K, T> {
    Value(T),
    Map(HashMap<K, NestedValue<K, T>>),
}

impl<K, V> NestedValue<K, V> {
    pub fn flatten(self, prefix: &str) -> HashMap<Rc<str>, V> 
    where 
        K: Display,
    {
        match self {
            NestedValue::Value(array) => {
                let mut map = HashMap::new();
                map.insert(prefix.into(), array);
                map
            }
            NestedValue::Map(entries) => {
                entries
                    .into_iter()
                    .flat_map(|(key, value)| value.flatten(&format!("{}{}{}", prefix, DELIMITER, key)))
                    .collect()
            }
        }
    }
}

#[derive(Debug)]
pub struct NestedHashMap<K, V> {
    pub entries: HashMap<K, NestedValue<K, V>>,
}

impl<K, V> NestedHashMap<K, V> {
    pub fn flatten(self) -> HashMap<Rc<str>, V>
    where 
        K: AsRef<str> + Display,
    {
        self.entries
            .into_iter()
            .flat_map(|(key, value)| value.flatten(key.as_ref()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::array;

    use super::*;

    #[test]
    fn test_flatten_nested_hash_map_of_owned_arrays() {
        let first_entry = NestedValue::Value(array!([1, 2, 3]));
        let second_entry = NestedValue::Map({
            let mut map = HashMap::new();
            map.insert("a", NestedValue::Value(array!([4, 5, 6])));
            map.insert("b", NestedValue::Value(array!([7, 8, 9])));
            map
        });

        let map = NestedHashMap {
            entries: {
                let mut map = HashMap::new();
                map.insert("first", first_entry);
                map.insert("second", second_entry);
                map
            },
        };

        let flattened = map.flatten();

        assert_eq!(flattened.len(), 3);
        assert_eq!(flattened["first"], array!([1, 2, 3]));
        assert_eq!(flattened["second.a"], array!([4, 5, 6]));
        assert_eq!(flattened["second.b"], array!([7, 8, 9]));
    }

    #[test]   
    fn test_flatten_nested_hash_map_of_borrowed_arrays() {
        let first_entry_content = array!([1, 2, 3]);
        let first_entry = NestedValue::Value(&first_entry_content);

        let second_entry_content_a = array!([4, 5, 6]);
        let second_entry_content_b = array!([7, 8, 9]);
        let second_entry = NestedValue::Map({
            let mut map = HashMap::new();
            map.insert("a", NestedValue::Value(&second_entry_content_a));
            map.insert("b", NestedValue::Value(&second_entry_content_b));
            map
        });

        let map = NestedHashMap {
            entries: {
                let mut map = HashMap::new();
                map.insert("first", first_entry);
                map.insert("second", second_entry);
                map
            },
        };

        let flattened = map.flatten();

        assert_eq!(flattened.len(), 3);
        assert_eq!(flattened["first"], &first_entry_content);
        assert_eq!(flattened["second.a"], &second_entry_content_a);
        assert_eq!(flattened["second.b"], &second_entry_content_b);
    }

    #[test]
    fn test_flatten_nested_hash_map_of_mut_borrowed_arrays() {
        let mut first_entry_content = array!([1, 2, 3]);
        let first_entry = NestedValue::Value(&mut first_entry_content);

        let mut second_entry_content_a = array!([4, 5, 6]);
        let mut second_entry_content_b = array!([7, 8, 9]);
        let second_entry = NestedValue::Map({
            let mut map = HashMap::new();
            map.insert("a", NestedValue::Value(&mut second_entry_content_a));
            map.insert("b", NestedValue::Value(&mut second_entry_content_b));
            map
        });

        let map = NestedHashMap {
            entries: {
                let mut map = HashMap::new();
                map.insert("first", first_entry);
                map.insert("second", second_entry);
                map
            },
        };

        let flattened = map.flatten();

        assert_eq!(flattened.len(), 3);
        assert_eq!(flattened["first"], &mut array!([1, 2, 3]));
        assert_eq!(flattened["second.a"], &mut array!([4, 5, 6]));
        assert_eq!(flattened["second.b"], &mut array!([7, 8, 9]));
    }
}