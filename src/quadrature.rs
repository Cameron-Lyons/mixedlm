use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn gauss_hermite_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    compute_gauss_hermite(n)
}

fn compute_gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Newton iteration is substantially faster for the small rules commonly
    // used in adaptive quadrature. The implicit QL method is more robust for
    // high orders, where root-to-root extrapolation becomes unreliable.
    if n <= 128 {
        return compute_gauss_hermite_newton(n);
    }

    compute_gauss_hermite_ql(n)
}

fn compute_gauss_hermite_newton(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let mut root = 0.0;

    for index in 0..n.div_ceil(2) {
        root = match index {
            0 => (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-1.0 / 6.0),
            1 => root - 1.14 * (n as f64).powf(0.426) / root,
            2 => 1.86 * root - 0.86 * nodes[0],
            3 => 1.91 * root - 0.91 * nodes[1],
            _ => 2.0 * root - nodes[index - 2],
        };

        let mut previous_polynomial = 0.0;
        for _ in 0..100 {
            let mut polynomial = std::f64::consts::PI.powf(-0.25);
            previous_polynomial = 0.0;

            for degree in 1..=n {
                let previous_previous = previous_polynomial;
                previous_polynomial = polynomial;
                polynomial = root * (2.0 / degree as f64).sqrt() * previous_polynomial
                    - ((degree - 1) as f64 / degree as f64).sqrt() * previous_previous;
            }

            let old_root = root;
            root -= polynomial / ((2.0 * n as f64).sqrt() * previous_polynomial);
            if (root - old_root).abs() < 1e-15 {
                break;
            }
        }

        nodes[index] = root;
        nodes[n - 1 - index] = -root;
        let weight = 1.0 / (n as f64 * previous_polynomial * previous_polynomial);
        weights[index] = weight;
        weights[n - 1 - index] = weight;
    }

    let mut paired: Vec<(f64, f64)> = nodes.into_iter().zip(weights).collect();
    paired.sort_by(|left, right| left.0.total_cmp(&right.0));
    paired.into_iter().unzip()
}

fn compute_gauss_hermite_ql(n: usize) -> (Vec<f64>, Vec<f64>) {
    // The Golub-Welsch Jacobi matrix for the physicists' Hermite
    // polynomials has a zero diagonal and sqrt(i / 2) off-diagonal.
    // Its eigenvalues are the quadrature nodes. Applying the same
    // transformations to the first basis vector gives the weights without
    // materializing the dense eigenvector matrix.
    let mut nodes = vec![0.0; n];
    let mut off_diagonal: Vec<f64> = (1..n).map(|i| (i as f64 / 2.0).sqrt()).collect();
    off_diagonal.push(0.0);

    let mut transformed = vec![0.0; n];
    transformed[0] = std::f64::consts::PI.powf(0.25);

    implicit_ql(&mut nodes, &mut off_diagonal, &mut transformed);

    let mut paired: Vec<(f64, f64)> = nodes
        .into_iter()
        .zip(transformed.into_iter().map(|value| value * value))
        .collect();
    paired.sort_by(|left, right| left.0.total_cmp(&right.0));
    paired.into_iter().unzip()
}

fn implicit_ql(diagonal: &mut [f64], off_diagonal: &mut [f64], transformed: &mut [f64]) {
    if diagonal.len() == 1 {
        return;
    }

    let n = diagonal.len();
    for left in 0..n {
        let mut iterations = 0;
        loop {
            let mut right = left;
            while right < n - 1
                && off_diagonal[right].abs()
                    > f64::EPSILON * (diagonal[right].abs() + diagonal[right + 1].abs())
            {
                right += 1;
            }

            if right == left {
                break;
            }

            iterations += 1;
            assert!(
                iterations <= 100,
                "Gauss-Hermite eigensolver did not converge"
            );

            let pivot = diagonal[left];
            let mut shift = (diagonal[left + 1] - pivot) / (2.0 * off_diagonal[left]);
            let radius = shift.hypot(1.0);
            shift = diagonal[right] - pivot + off_diagonal[left] / (shift + radius.copysign(shift));

            let mut sine = 1.0;
            let mut cosine = 1.0;
            let mut correction = 0.0;

            for offset in 1..=(right - left) {
                let index = right - offset;
                let scaled_off_diagonal = sine * off_diagonal[index];
                let projected_off_diagonal = cosine * off_diagonal[index];

                if shift.abs() <= scaled_off_diagonal.abs() {
                    cosine = shift / scaled_off_diagonal;
                    let norm = cosine.hypot(1.0);
                    off_diagonal[index + 1] = scaled_off_diagonal * norm;
                    sine = norm.recip();
                    cosine *= sine;
                } else {
                    sine = scaled_off_diagonal / shift;
                    let norm = sine.hypot(1.0);
                    off_diagonal[index + 1] = shift * norm;
                    cosine = norm.recip();
                    sine *= cosine;
                }

                shift = diagonal[index + 1] - correction;
                let rotation =
                    (diagonal[index] - shift) * sine + 2.0 * cosine * projected_off_diagonal;
                correction = sine * rotation;
                diagonal[index + 1] = shift + correction;
                shift = cosine * rotation - projected_off_diagonal;

                let transformed_next = transformed[index + 1];
                transformed[index + 1] = sine * transformed[index] + cosine * transformed_next;
                transformed[index] = cosine * transformed[index] - sine * transformed_next;
            }

            diagonal[left] -= correction;
            off_diagonal[left] = shift;
            off_diagonal[right] = 0.0;
        }
    }
}

#[pyfunction]
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    gauss_hermite_nodes_weights(n)
}

#[pyfunction]
pub fn adaptive_gauss_hermite_1d(
    nodes: Vec<f64>,
    weights: Vec<f64>,
    mode: f64,
    scale: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if nodes.len() != weights.len() {
        return Err(PyValueError::new_err(
            "nodes and weights must have the same length",
        ));
    }
    if !mode.is_finite() {
        return Err(PyValueError::new_err("mode must be finite"));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PyValueError::new_err(
            "scale must be finite and greater than zero",
        ));
    }
    if nodes.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "nodes must contain only finite values",
        ));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "weights must contain only finite values",
        ));
    }

    let sqrt2 = std::f64::consts::SQRT_2;
    let mut adapted_nodes = Vec::with_capacity(nodes.len());
    let mut adapted_weights = Vec::with_capacity(weights.len());

    for (node, weight) in nodes.into_iter().zip(weights) {
        adapted_nodes.push(mode + sqrt2 * scale * node);
        adapted_weights.push(weight * (node * node).exp());
    }

    Ok((adapted_nodes, adapted_weights))
}

#[cfg(test)]
mod tests {
    use super::compute_gauss_hermite;

    #[test]
    fn high_order_rule_integrates_low_order_moments() {
        let (nodes, weights) = compute_gauss_hermite(200);
        let total_weight: f64 = weights.iter().sum();
        let second_moment: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(node, weight)| node * node * weight)
            .sum();

        assert!((total_weight - std::f64::consts::PI.sqrt()).abs() < 1e-13);
        assert!((second_moment - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-13);
        assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
        );
    }
}
