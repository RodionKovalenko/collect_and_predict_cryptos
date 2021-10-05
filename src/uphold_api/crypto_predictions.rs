use std::process::Command;

pub fn make_prediction_daily() {
    println!("prediction are being made...");
    if let Ok(mut c) = Command::new("cmd")
        .args(&["/C", "python src/lstm-conv-dense.py"])
        .spawn() {
        println!("Spawned successfully");
        println!("Exit with: {:?}", c.wait());
    } else {
        panic!("panic");
    }

    println!("predictions completed");
}