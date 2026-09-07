//! AlphaFactory core: a small, deterministic, point-in-time-safe formula VM.
//!
//! V0.1 intentionally has no ML dependencies. The first invariant is causal
//! correctness: changing future observations must never change past outputs.

use std::fmt;

pub type Series = Vec<f64>;
pub type FeatureMatrix = Vec<Series>;

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Input(usize),
    Const(f64),
    Add,
    Sub,
    Mul,
    SafeDiv { epsilon: f64 },
    Neg,
    Abs,
    Delay(usize),
    RollingMean(usize),
    RollingStd(usize),
    RollingZScore { window: usize, epsilon: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Program {
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self { instructions }
    }

    /// Evaluate a postfix program over feature-major time series.
    ///
    /// `features[feature_index][time_index]` is the only input layout in V0.1.
    /// All operators preserve the time axis and all rolling operators are causal.
    pub fn evaluate(&self, features: &FeatureMatrix) -> Result<Series, VmError> {
        let len = validate_features(features)?;
        let mut stack: Vec<Series> = Vec::new();

        for instruction in &self.instructions {
            match *instruction {
                Instruction::Input(index) => {
                    let input = features.get(index).ok_or(VmError::MissingFeature(index))?;
                    stack.push(input.clone());
                }
                Instruction::Const(value) => stack.push(vec![value; len]),
                Instruction::Add => binary(&mut stack, |a, b| a + b)?,
                Instruction::Sub => binary(&mut stack, |a, b| a - b)?,
                Instruction::Mul => binary(&mut stack, |a, b| a * b)?,
                Instruction::SafeDiv { epsilon } => {
                    if !epsilon.is_finite() || epsilon <= 0.0 {
                        return Err(VmError::InvalidParameter("SafeDiv epsilon"));
                    }
                    binary(&mut stack, |a, b| {
                        if a.is_nan() || b.is_nan() || b.abs() <= epsilon {
                            f64::NAN
                        } else {
                            a / b
                        }
                    })?;
                }
                Instruction::Neg => unary(&mut stack, |x| -x)?,
                Instruction::Abs => unary(&mut stack, f64::abs)?,
                Instruction::Delay(periods) => {
                    let input = pop_one(&mut stack)?;
                    stack.push(delay(&input, periods));
                }
                Instruction::RollingMean(window) => {
                    validate_window(window)?;
                    let input = pop_one(&mut stack)?;
                    stack.push(rolling_mean(&input, window));
                }
                Instruction::RollingStd(window) => {
                    validate_window(window)?;
                    let input = pop_one(&mut stack)?;
                    stack.push(rolling_std(&input, window));
                }
                Instruction::RollingZScore { window, epsilon } => {
                    validate_window(window)?;
                    if !epsilon.is_finite() || epsilon <= 0.0 {
                        return Err(VmError::InvalidParameter("RollingZScore epsilon"));
                    }
                    let input = pop_one(&mut stack)?;
                    stack.push(rolling_zscore(&input, window, epsilon));
                }
            }
        }

        if stack.len() != 1 {
            return Err(VmError::InvalidFinalStackDepth(stack.len()));
        }
        Ok(stack.pop().expect("length checked"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VmError {
    EmptyFeatureMatrix,
    EmptySeries,
    RaggedFeatureMatrix,
    MissingFeature(usize),
    StackUnderflow,
    InvalidFinalStackDepth(usize),
    InvalidParameter(&'static str),
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::EmptyFeatureMatrix => write!(f, "feature matrix is empty"),
            VmError::EmptySeries => write!(f, "feature series is empty"),
            VmError::RaggedFeatureMatrix => write!(f, "feature series have different lengths"),
            VmError::MissingFeature(index) => write!(f, "missing feature index {index}"),
            VmError::StackUnderflow => write!(f, "formula stack underflow"),
            VmError::InvalidFinalStackDepth(depth) => {
                write!(f, "formula ended with stack depth {depth}, expected 1")
            }
            VmError::InvalidParameter(name) => write!(f, "invalid parameter: {name}"),
        }
    }
}

impl std::error::Error for VmError {}

fn validate_features(features: &FeatureMatrix) -> Result<usize, VmError> {
    let first = features.first().ok_or(VmError::EmptyFeatureMatrix)?;
    if first.is_empty() {
        return Err(VmError::EmptySeries);
    }
    let len = first.len();
    if features.iter().any(|series| series.len() != len) {
        return Err(VmError::RaggedFeatureMatrix);
    }
    Ok(len)
}

fn validate_window(window: usize) -> Result<(), VmError> {
    if window == 0 {
        Err(VmError::InvalidParameter("rolling window"))
    } else {
        Ok(())
    }
}

fn pop_one(stack: &mut Vec<Series>) -> Result<Series, VmError> {
    stack.pop().ok_or(VmError::StackUnderflow)
}

fn unary<F>(stack: &mut Vec<Series>, op: F) -> Result<(), VmError>
where
    F: Fn(f64) -> f64,
{
    let input = pop_one(stack)?;
    stack.push(input.into_iter().map(op).collect());
    Ok(())
}

fn binary<F>(stack: &mut Vec<Series>, op: F) -> Result<(), VmError>
where
    F: Fn(f64, f64) -> f64,
{
    let rhs = pop_one(stack)?;
    let lhs = pop_one(stack)?;
    stack.push(
        lhs.into_iter()
            .zip(rhs)
            .map(|(a, b)| op(a, b))
            .collect(),
    );
    Ok(())
}

/// Pure lag. The first `periods` values are unavailable, never back-filled.
pub fn delay(input: &[f64], periods: usize) -> Series {
    if periods == 0 {
        return input.to_vec();
    }
    let mut out = vec![f64::NAN; input.len()];
    for t in periods..input.len() {
        out[t] = input[t - periods];
    }
    out
}

/// Causal trailing rolling mean over [t-window+1, t].
/// Warm-up values are NaN and NaNs inside a full window invalidate the result.
pub fn rolling_mean(input: &[f64], window: usize) -> Series {
    if window == 0 {
        return vec![f64::NAN; input.len()];
    }
    let mut out = vec![f64::NAN; input.len()];
    if window > input.len() {
        return out;
    }

    for t in (window - 1)..input.len() {
        let slice = &input[(t + 1 - window)..=t];
        if slice.iter().all(|x| x.is_finite()) {
            out[t] = slice.iter().sum::<f64>() / window as f64;
        }
    }
    out
}

/// Population standard deviation using only the trailing causal window.
pub fn rolling_std(input: &[f64], window: usize) -> Series {
    if window == 0 {
        return vec![f64::NAN; input.len()];
    }
    let mut out = vec![f64::NAN; input.len()];
    if window > input.len() {
        return out;
    }

    for t in (window - 1)..input.len() {
        let slice = &input[(t + 1 - window)..=t];
        if !slice.iter().all(|x| x.is_finite()) {
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        let variance = slice
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / window as f64;
        out[t] = variance.sqrt();
    }
    out
}

pub fn rolling_zscore(input: &[f64], window: usize, epsilon: f64) -> Series {
    let means = rolling_mean(input, window);
    let stds = rolling_std(input, window);
    input
        .iter()
        .zip(means.iter().zip(stds.iter()))
        .map(|(x, (mean, std))| {
            if !x.is_finite() || !mean.is_finite() || !std.is_finite() || *std <= epsilon {
                f64::NAN
            } else {
                (x - mean) / std
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_or_close(a: f64, b: f64) {
        if a.is_nan() && b.is_nan() {
            return;
        }
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn future_mutation_cannot_change_past_rolling_values() {
        let program = Program::new(vec![
            Instruction::Input(0),
            Instruction::RollingZScore {
                window: 3,
                epsilon: 1e-12,
            },
        ]);

        let original = vec![vec![1.0, 2.0, 4.0, 8.0, 16.0]];
        let mut mutated_future = original.clone();
        mutated_future[0][4] = -1_000_000.0;

        let a = program.evaluate(&original).unwrap();
        let b = program.evaluate(&mutated_future).unwrap();

        // Mutation at t=4 must not alter outputs at t<=3.
        for t in 0..4 {
            assert_same_or_close(a[t], b[t]);
        }
        assert_ne!(a[4], b[4]);
    }

    #[test]
    fn delay_is_strictly_backward_looking() {
        let got = delay(&[10.0, 20.0, 30.0, 40.0], 2);
        assert!(got[0].is_nan());
        assert!(got[1].is_nan());
        assert_eq!(got[2], 10.0);
        assert_eq!(got[3], 20.0);
    }

    #[test]
    fn rolling_mean_requires_a_complete_past_window() {
        let got = rolling_mean(&[1.0, 2.0, 3.0, 4.0], 3);
        assert!(got[0].is_nan());
        assert!(got[1].is_nan());
        assert_eq!(got[2], 2.0);
        assert_eq!(got[3], 3.0);
    }

    #[test]
    fn postfix_program_composes_features_without_future_data() {
        // (feature0 - delay(feature0, 1)) * feature1
        let program = Program::new(vec![
            Instruction::Input(0),
            Instruction::Input(0),
            Instruction::Delay(1),
            Instruction::Sub,
            Instruction::Input(1),
            Instruction::Mul,
        ]);
        let features = vec![vec![10.0, 12.0, 11.0], vec![2.0, 3.0, 4.0]];
        let got = program.evaluate(&features).unwrap();
        assert!(got[0].is_nan());
        assert_eq!(got[1], 6.0);
        assert_eq!(got[2], -4.0);
    }

    #[test]
    fn malformed_formula_is_rejected() {
        let program = Program::new(vec![Instruction::Input(0), Instruction::Add]);
        let err = program.evaluate(&vec![vec![1.0, 2.0]]).unwrap_err();
        assert_eq!(err, VmError::StackUnderflow);
    }
}
