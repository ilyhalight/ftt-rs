use ftt_rs::Model;

fn main() {
    let m = Model::load("tests/data/models/dense_softmax.bin").unwrap();
    println!("eos_id = {}", m.dictionary().eos_id());
    println!("good true  = {:?}", m.feature_ids("good", true));
    println!("good false = {:?}", m.feature_ids("good", false));
    println!("line true  = {:?}", m.feature_ids("good\nignored", true));
    println!("line false = {:?}", m.feature_ids("good\nignored", false));
}
