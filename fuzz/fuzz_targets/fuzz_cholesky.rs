#![no_main]

use libfuzzer_sys::fuzz_target;
use mixedlm_fuzz::{CholeskyInput, check_cholesky};

fuzz_target!(|input: CholeskyInput| check_cholesky(input));
