//! Fuzz the production sparse boundary and factorization with independent oracles.

use arbitrary::Arbitrary;
use numpy::ndarray::Array2;

// Compile the actual project modules, rather than a separate solver example.
// The Python-facing helpers are included to preserve their shared error type;
// neither fuzz target calls the Python interpreter.
#[path = "../../src/csc.rs"]
pub mod csc;
#[path = "../../src/linalg.rs"]
pub mod linalg;
#[path = "../../src/sparse_chol.rs"]
pub mod sparse_chol;

#[derive(Arbitrary, Debug)]
pub struct CholeskyInput {
    pub dimension: u8,
    pub sparse: bool,
    pub noncanonical: bool,
    pub amd: bool,
    pub coefficients: Vec<i16>,
    pub solution: Vec<i16>,
}

fn value_at(values: &[i16], position: usize) -> f64 {
    f64::from(values.get(position).copied().unwrap_or(1)) / 32768.0
}

pub fn check_cholesky(input: CholeskyInput) {
    let n = usize::from(input.dimension % 12) + 1;
    let mut matrix: Vec<Vec<f64>> = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| {
                    if row == column || (input.sparse && (row + column) % 3 == 0) {
                        0.0
                    } else {
                        value_at(&input.coefficients, row.max(column) * n + row.min(column))
                    }
                })
                .collect()
        })
        .collect();
    // Strict diagonal dominance guarantees positive definiteness, including
    // empty/zero fuzz inputs. Every invocation exercises factorization.
    for (row, entries) in matrix.iter_mut().enumerate() {
        entries[row] = 1.0 + entries.iter().map(|value| value.abs()).sum::<f64>();
    }
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];
    for (column, entries) in matrix.iter().enumerate() {
        for (row, &value) in entries.iter().enumerate().skip(column).rev() {
            if value == 0.0 {
                continue;
            }
            indices.push(row);
            if input.noncanonical {
                indices.push(row);
                data.extend([0.25 * value, 0.75 * value]);
            } else {
                data.push(value);
            }
        }
        indptr.push(data.len());
    }
    let symbolic = if input.amd {
        sparse_chol::SymbolicCholeskyCache::new_amd(&indices, &indptr, n)
    } else {
        sparse_chol::SymbolicCholeskyCache::new(&indices, &indptr, n)
    }
    .expect("a valid positive definite sparse matrix must have a symbolic factor");
    let factor = symbolic
        .factor(&data, &indices, &indptr)
        .expect("a strictly diagonally dominant matrix must factor");
    let expected = Array2::from_shape_fn((n, 2), |(row, column)| {
        value_at(&input.solution, row * 2 + column)
    });
    let rhs = Array2::from_shape_fn((n, 2), |(row, column)| {
        (0..n)
            .map(|position| matrix[row][position] * expected[(position, column)])
            .sum()
    });
    let actual = factor.solve(rhs.view()).expect("valid right hand side");
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-10,
            "sparse solve changed the known solution"
        );
    }
    // A scalar Cholesky recurrence gives an independent log-determinant oracle.
    let mut lower = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in 0..=row {
            let product: f64 = (0..column)
                .map(|position| lower[row][position] * lower[column][position])
                .sum();
            lower[row][column] = if row == column {
                (matrix[row][row] - product).sqrt()
            } else {
                (matrix[row][column] - product) / lower[column][column]
            };
        }
    }
    let expected_logdet = 2.0 * (0..n).map(|row| lower[row][row].ln()).sum::<f64>();
    assert!(
        (factor.logdet() - expected_logdet).abs() <= 1e-10,
        "sparse logdet disagrees with scalar Cholesky"
    );
}

#[derive(Arbitrary, Debug)]
pub struct CscInput {
    pub valid: bool,
    pub rows: u8,
    pub columns: u8,
    pub values: Vec<i16>,
    pub indices: Vec<i16>,
    pub indptr: Vec<i16>,
}

pub fn check_csc(input: CscInput) -> bool {
    let rows = usize::from(input.rows % 33);
    let columns = usize::from(input.columns % 65);
    // Half the input space produces valid, frequently noncanonical matrices.
    // Independent arbitrary pointers would almost always stop at validation,
    // leaving numerical crossproducts and canonicalization scarcely exercised.
    let (data, indices, indptr) = if input.valid {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        for column in 0..columns {
            let count = if rows == 0 {
                0
            } else {
                usize::from(
                    input
                        .indptr
                        .get(column)
                        .copied()
                        .unwrap_or(3)
                        .unsigned_abs(),
                ) % (2 * rows + 1)
            };
            for _ in 0..count {
                let position = data.len();
                let value = input.values.get(position).copied().unwrap_or(1);
                let row = input
                    .indices
                    .get(position)
                    .copied()
                    .unwrap_or((rows - 1 - position % rows) as i16);
                data.push(f64::from(value) / 256.0);
                indices.push((usize::from(row.unsigned_abs()) % rows) as i64);
            }
            indptr.push(data.len() as i64);
        }
        (data, indices, indptr)
    } else {
        (
            input
                .values
                .iter()
                .map(|&value| f64::from(value) / 256.0)
                .collect(),
            input
                .indices
                .iter()
                .map(|&value| i64::from(value))
                .collect(),
            input.indptr.iter().map(|&value| i64::from(value)).collect(),
        )
    };
    let expected_valid = data.len() == indices.len()
        && indptr.len() == columns + 1
        && indptr.first() == Some(&0)
        && indptr.last().copied() == Some(indices.len() as i64)
        && indptr
            .windows(2)
            .all(|pair| pair[0] >= 0 && pair[0] <= pair[1])
        && indices.iter().all(|&row| row >= 0 && row < rows as i64);
    let actual = csc::CscMatrix::try_from_i64(&data, &indices, &indptr, (rows, columns));
    assert_eq!(
        actual.is_ok(),
        expected_valid,
        "CSC accepted or rejected the wrong structure"
    );
    let Ok(matrix) = actual else { return false };
    let mut dense = vec![vec![0.0; columns]; rows];
    for column in 0..columns {
        for position in indptr[column] as usize..indptr[column + 1] as usize {
            dense[indices[position] as usize][column] += data[position];
        }
        let start = matrix.col_offsets()[column];
        let stop = matrix.col_offsets()[column + 1];
        assert!(
            matrix.row_indices()[start..stop]
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        for position in start..stop {
            assert_eq!(
                matrix.values()[position],
                dense[matrix.row_indices()[position]][column]
            );
        }
    }
    let weights: Vec<_> = (0..rows).map(|row| 1.0 / (row + 1) as f64).collect();
    let actual = matrix.weighted_crossproduct(&weights);
    for left in 0..columns {
        for right in 0..columns {
            let expected: f64 = dense
                .iter()
                .zip(&weights)
                .map(|(row, weight)| row[left] * weight * row[right])
                .sum();
            let absolute_sum: f64 = dense
                .iter()
                .zip(&weights)
                .map(|(row, weight)| (row[left] * weight * row[right]).abs())
                .sum();
            // Scale the rounding bound by the accumulated products, including
            // cases whose positive and negative terms nearly cancel.
            let tolerance = 64.0 * f64::EPSILON * (1.0 + absolute_sum);
            assert!((actual[(left, right)] - expected).abs() <= tolerance);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_factorization_oracles_cover_storage_and_ordering() {
        for dimension in [0, 1, 3, 7, 11] {
            for sparse in [false, true] {
                for noncanonical in [false, true] {
                    for amd in [false, true] {
                        check_cholesky(CholeskyInput {
                            dimension,
                            sparse,
                            noncanonical,
                            amd,
                            coefficients: (0..144)
                                .map(|i| ((i * 701) % 65536 - 32768) as i16)
                                .collect(),
                            solution: (0..24)
                                .map(|i| ((i * 7901) % 65536 - 32768) as i16)
                                .collect(),
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_boundary_oracle_covers_empty_valid_and_malformed_inputs() {
        let (mut valid, mut malformed) = (0, 0);
        for columns in 0..=16 {
            assert!(check_csc(CscInput {
                valid: false,
                rows: 0,
                columns,
                values: vec![],
                indices: vec![],
                indptr: vec![0; usize::from(columns) + 1],
            }));
            valid += 1;
        }
        for indptr in [
            vec![0, 3, 5],
            vec![-1, 3, 5],
            vec![0, 5, 3],
            vec![0, 3, 6],
            vec![0],
        ] {
            if check_csc(CscInput {
                valid: false,
                rows: 3,
                columns: 2,
                values: vec![32, 64, 16, 64, -32],
                indices: vec![2, 0, 2, 1, 0],
                indptr,
            }) {
                valid += 1;
            } else {
                malformed += 1;
            }
        }
        assert_eq!((valid, malformed), (18, 4));
        for indices in [vec![2, -1, 2, 1, 0], vec![3, 0, 2, 1, 0], vec![2, 0, 2, 1]] {
            assert!(!check_csc(CscInput {
                valid: false,
                rows: 3,
                columns: 2,
                values: vec![32, 64, 16, 64, -32],
                indices,
                indptr: vec![0, 3, 5],
            }));
            malformed += 1;
        }
        for rows in 0..=16 {
            for columns in 0..=16 {
                assert!(check_csc(CscInput {
                    valid: true,
                    rows,
                    columns,
                    values: vec![32, 64, 16, 64, -32],
                    indices: vec![2, 0, 2, 1, 0],
                    indptr: vec![3, 5],
                }));
                valid += 1;
            }
        }
        assert_eq!((valid, malformed), (307, 7));
    }
}
