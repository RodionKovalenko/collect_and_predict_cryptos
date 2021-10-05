extern crate serde_derive;
extern crate serde;
extern crate serde_json;

use std::env;

#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_assignments)]
pub mod uphold_api;

#[allow(unused_imports)]
use uphold_api::*;
#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
use rand::Rng;

pub enum ARGUMENTS {
    UPHOLD,
    NETWORK,
}

/**
* start with cargo run
 */

fn main() {
    println!("Test beginns");

    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        let arg1: &str = &*args[1].clone();

        match arg1 {
            "uphold" => collect_data_task::update_json_data_from_uphold_api(),
            _ => println!(" no argument recognized"),
        }
    } else {
        // default
        collect_data_task::update_json_data_from_uphold_api();
    }
}