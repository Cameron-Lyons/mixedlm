#![no_main]

use arbitrary::Arbitrary;
use faer::linalg::solvers::Llt;
use faer::{Mat, Side};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MatrixInput {
    n: u8,
    data: Vec<f64>,
}

fuzz_target!(|input: MatrixInput| {
    let n = ((input.n % 8) + 1) as usize;

    if input.data.len() < n * n {
        return;
    }

    let mut matrix = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let val = input.data[i * n + j];
            if val.is_finite() {
                matrix[(i, j)] = val;
            }
        }
    }

    for i in 0..n {
        matrix[(i, i)] = matrix[(i, i)].abs() + (n as f64);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let val = (matrix[(i, j)] + matrix[(j, i)]) / 2.0;
            matrix[(i, j)] = val;
            matrix[(j, i)] = val;
        }
    }

    let _ = Llt::new(matrix.as_ref(), Side::Lower);
});
