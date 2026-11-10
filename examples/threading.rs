use std::thread;

fn main() { 
    println!("available_parallelism: {0}", thread::available_parallelism().unwrap().get());
}