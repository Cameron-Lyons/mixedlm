use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

const CACHE_SIZE: usize = 32;
const MAX_CACHED_ORDER: usize = 1024;

pub struct HermiteRule {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

#[derive(Default)]
struct RuleCache {
    rules: VecDeque<(usize, Arc<HermiteRule>)>,
}

impl RuleCache {
    fn get(&mut self, n: usize) -> Option<Arc<HermiteRule>> {
        let index = self.rules.iter().position(|(order, _)| *order == n)?;
        let entry = self.rules.remove(index)?;
        let rule = Arc::clone(&entry.1);
        self.rules.push_front(entry);
        Some(rule)
    }

    fn insert(&mut self, n: usize, rule: Arc<HermiteRule>) -> Arc<HermiteRule> {
        if n == 0 || n > MAX_CACHED_ORDER {
            return rule;
        }
        if let Some(existing) = self.get(n) {
            return existing;
        }
        if self.rules.len() == CACHE_SIZE {
            self.rules.pop_back();
        }
        self.rules.push_front((n, Arc::clone(&rule)));
        rule
    }
}

pub fn gauss_hermite_nodes_weights(n: usize) -> Arc<HermiteRule> {
    static CACHE: OnceLock<Mutex<RuleCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(RuleCache::default()));
    if n > 0 && n <= MAX_CACHED_ORDER {
        let mut guard = cache.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(rule) = guard.get(n) {
            return rule;
        }
    }

    // Compute outside the lock so unrelated cache hits are not delayed.
    let (nodes, weights) = compute_gauss_hermite(n);
    let rule = Arc::new(HermiteRule { nodes, weights });
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(n, rule)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_requests_share_rule_storage() {
        let first = gauss_hermite_nodes_weights(67);
        let second = gauss_hermite_nodes_weights(67);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn cache_bounds_storage_and_evicts_least_recently_used() {
        let mut cache = RuleCache::default();
        let rule = Arc::new(HermiteRule {
            nodes: vec![0.0],
            weights: vec![std::f64::consts::PI.sqrt()],
        });
        for order in 1..=CACHE_SIZE {
            cache.insert(order, Arc::clone(&rule));
        }
        cache.get(2);
        cache.get(1);
        cache.insert(CACHE_SIZE + 1, Arc::clone(&rule));
        assert!(cache.get(3).is_none());
        assert!(cache.get(1).is_some());
        assert_eq!(cache.rules.len(), CACHE_SIZE);
        cache.insert(0, Arc::clone(&rule));
        cache.insert(MAX_CACHED_ORDER + 1, Arc::clone(&rule));
        assert!(cache.get(0).is_none());
        assert!(cache.get(MAX_CACHED_ORDER + 1).is_none());
        assert_eq!(cache.rules.len(), CACHE_SIZE);
    }

    #[test]
    fn concurrent_misses_publish_one_shared_rule() {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| gauss_hermite_nodes_weights(71)))
                .collect();
            let rules: Vec<_> = handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();
            assert!(rules.iter().all(|rule| Arc::ptr_eq(rule, &rules[0])));
        });
    }

    #[test]
    fn rules_are_sorted_symmetric_and_exact_for_low_degree_polynomials() {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert!(compute_gauss_hermite(0).0.is_empty());
        let (nodes, weights) = compute_gauss_hermite(1);
        assert!(nodes == [0.0] && (weights[0] - sqrt_pi).abs() < 1e-15);
        let (nodes, weights) = compute_gauss_hermite(2);
        let node = std::f64::consts::FRAC_1_SQRT_2;
        assert!((nodes[0] + node).abs() < 1e-15 && (nodes[1] - node).abs() < 1e-15);
        assert!(
            weights
                .iter()
                .all(|weight| (weight - sqrt_pi / 2.0).abs() < 1e-15)
        );
        // Orders on both sides of the switch from Newton iteration to the QL method.
        for n in [5, 10, 50, 128, 129, 200] {
            let (nodes, weights) = compute_gauss_hermite(n);
            assert_eq!((nodes.len(), weights.len()), (n, n));
            assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(
                weights
                    .iter()
                    .all(|weight| weight.is_finite() && *weight > 0.0)
            );
            for (index, (node, weight)) in nodes.iter().zip(&weights).enumerate() {
                let (mirror, mirror_weight) = (nodes[n - 1 - index], weights[n - 1 - index]);
                assert!(
                    (node + mirror).abs() <= 1e-12 * node.abs().max(1.0),
                    "{n}: {node}"
                );
                assert!(
                    (weight - mirror_weight).abs() <= 1e-12 * weight,
                    "{n}: {weight}"
                );
            }
            // An n-point rule integrates x^(2k) exp(-x^2) exactly, to Gamma(k + 1/2).
            let mut expected = sqrt_pi;
            for k in 0..5 {
                let moment: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(node, weight)| weight * node.powi(2 * k))
                    .sum();
                assert!(
                    (moment - expected).abs() <= 1e-12 * expected,
                    "{n}, {k}: {moment}"
                );
                expected *= f64::from(k) + 0.5;
            }
        }
    }
}
