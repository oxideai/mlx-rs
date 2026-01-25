use mlx_rs_lm::text::g2pw::get_pinyin_with_g2pw;

fn main() {
    let test_cases = [
        "几行代码",
        "银行存款",
        "行走江湖",
        "一行人",
        "举行会议",
    ];

    for text in test_cases {
        println!("\nText: {}", text);
        let pinyin = get_pinyin_with_g2pw(text);
        for (c, p) in text.chars().zip(pinyin.iter()) {
            match p {
                Some(py) => println!("  {} -> {}", c, py),
                None => println!("  {} -> (none)", c),
            }
        }
    }
}
