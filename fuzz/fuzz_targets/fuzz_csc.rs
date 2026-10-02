#![no_main]

use libfuzzer_sys::fuzz_target;
use mixedlm_fuzz::{CscInput, check_csc};

fuzz_target!(|input: CscInput| {
    check_csc(input);
});
