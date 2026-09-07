const EPS: f64 = 1e-12;

pub fn pearson_corr(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    if lhs.len() != rhs.len() {
        return None;
    }

    let pairs: Vec<(f64, f64)> = lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .collect();

    if pairs.len() < 2 {
        return None;
    }

    let n = pairs.len() as f64;
    let mean_lhs = pairs.iter().map(|(a, _)| *a).sum::<f64>() / n;
    let mean_rhs = pairs.iter().map(|(_, b)| *b).sum::<f64>() / n;

    let mut covariance = 0.0;
    let mut lhs_var = 0.0;
    let mut rhs_var = 0.0;

    for (a, b) in pairs {
        let da = a - mean_lhs;
        let db = b - mean_rhs;
        covariance += da * db;
        lhs_var += da * da;
        rhs_var += db * db;
    }

    let denom = (lhs_var * rhs_var).sqrt();
    if denom <= EPS {
        None
    } else {
        Some(covariance / denom)
    }
}

pub fn rank_ic(signal: &[f64], forward_return: &[f64]) -> Option<f64> {
    if signal.len() != forward_return.len() {
        return None;
    }

    let pairs: Vec<(f64, f64)> = signal
        .iter()
        .copied()
        .zip(forward_return.iter().copied())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .collect();

    if pairs.len() < 2 {
        return None;
    }

    let signal_values: Vec<f64> = pairs.iter().map(|(a, _)| *a).collect();
    let return_values: Vec<f64> = pairs.iter().map(|(_, b)| *b).collect();
    let signal_ranks = average_ranks(&signal_values);
    let return_ranks = average_ranks(&return_values);
    pearson_corr(&signal_ranks, &return_ranks)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;

    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && values[order[start]].total_cmp(&values[order[end]]).is_eq()
        {
            end += 1;
        }

        let first_rank = start as f64 + 1.0;
        let last_rank = end as f64;
        let average_rank = (first_rank + last_rank) / 2.0;
        for &index in &order[start..end] {
            ranks[index] = average_rank;
        }
        start = end;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_ic_detects_perfect_ordering() {
        let signal = [10.0, 20.0, 30.0, 40.0];
        let returns = [-1.0, 0.0, 3.0, 8.0];
        let ic = rank_ic(&signal, &returns).unwrap();
        assert!((ic - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rank_ic_detects_inverse_ordering() {
        let signal = [10.0, 20.0, 30.0, 40.0];
        let returns = [8.0, 3.0, 0.0, -1.0];
        let ic = rank_ic(&signal, &returns).unwrap();
        assert!((ic + 1.0).abs() < 1e-12);
    }

    #[test]
    fn rank_ic_ignores_missing_pairs_without_imputation() {
        let signal = [1.0, f64::NAN, 3.0, 4.0];
        let returns = [1.0, 99.0, 3.0, 4.0];
        let ic = rank_ic(&signal, &returns).unwrap();
        assert!((ic - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_series_has_no_defined_correlation() {
        assert_eq!(pearson_corr(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None);
    }
}
