use std::collections::BTreeMap;

const DIV_EPS: f64 = 1e-12;
const STD_EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Input(String),
    Const(f64),
    Add,
    Sub,
    Mul,
    SafeDiv,
    Neg,
    Abs,
    Delay(usize),
    RollingMean(usize),
    RollingStd(usize),
    RollingZScore(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    EmptyFormula,
    UnknownInput(String),
    LengthMismatch {
        input: String,
        expected: usize,
        actual: usize,
    },
    StackUnderflow {
        token_index: usize,
    },
    InvalidWindow {
        token_index: usize,
        window: usize,
    },
    FinalStackDepth {
        depth: usize,
    },
}

#[derive(Debug, Clone, Default)]
pub struct Frame {
    len: usize,
    columns: BTreeMap<String, Vec<f64>>,
}

impl Frame {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            columns: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn insert(&mut self, name: impl Into<String>, values: Vec<f64>) -> Result<(), EvalError> {
        let name = name.into();
        if values.len() != self.len {
            return Err(EvalError::LengthMismatch {
                input: name,
                expected: self.len,
                actual: values.len(),
            });
        }
        self.columns.insert(name, values);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&[f64]> {
        self.columns.get(name).map(Vec::as_slice)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    tokens: Vec<Token>,
}

impl Formula {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens }
    }

    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    pub fn canonical(&self) -> String {
        self.tokens
            .iter()
            .map(|token| match token {
                Token::Input(name) => format!("I:{}:{}", name.len(), name),
                Token::Const(value) => format!("C:{:016x}", value.to_bits()),
                Token::Add => "ADD".to_string(),
                Token::Sub => "SUB".to_string(),
                Token::Mul => "MUL".to_string(),
                Token::SafeDiv => "SAFE_DIV".to_string(),
                Token::Neg => "NEG".to_string(),
                Token::Abs => "ABS".to_string(),
                Token::Delay(lag) => format!("DELAY:{lag}"),
                Token::RollingMean(window) => format!("ROLL_MEAN:{window}"),
                Token::RollingStd(window) => format!("ROLL_STD:{window}"),
                Token::RollingZScore(window) => format!("ROLL_Z:{window}"),
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    pub fn stable_hash64(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in self.canonical().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub fn evaluate(&self, frame: &Frame) -> Result<Vec<f64>, EvalError> {
        if self.tokens.is_empty() {
            return Err(EvalError::EmptyFormula);
        }

        let mut stack: Vec<Vec<f64>> = Vec::new();

        for (token_index, token) in self.tokens.iter().enumerate() {
            match token {
                Token::Input(name) => {
                    let values = frame
                        .get(name)
                        .ok_or_else(|| EvalError::UnknownInput(name.clone()))?;
                    stack.push(values.to_vec());
                }
                Token::Const(value) => stack.push(vec![*value; frame.len()]),
                Token::Add => apply_binary(&mut stack, token_index, |a, b| a + b)?,
                Token::Sub => apply_binary(&mut stack, token_index, |a, b| a - b)?,
                Token::Mul => apply_binary(&mut stack, token_index, |a, b| a * b)?,
                Token::SafeDiv => apply_binary(&mut stack, token_index, |a, b| {
                    if b.abs() <= DIV_EPS {
                        f64::NAN
                    } else {
                        a / b
                    }
                })?,
                Token::Neg => apply_unary(&mut stack, token_index, |x| -x)?,
                Token::Abs => apply_unary(&mut stack, token_index, f64::abs)?,
                Token::Delay(lag) => {
                    let input = pop_one(&mut stack, token_index)?;
                    stack.push(delay(&input, *lag));
                }
                Token::RollingMean(window) => {
                    validate_window(token_index, *window)?;
                    let input = pop_one(&mut stack, token_index)?;
                    stack.push(rolling_mean(&input, *window));
                }
                Token::RollingStd(window) => {
                    validate_window(token_index, *window)?;
                    let input = pop_one(&mut stack, token_index)?;
                    stack.push(rolling_std(&input, *window));
                }
                Token::RollingZScore(window) => {
                    validate_window(token_index, *window)?;
                    let input = pop_one(&mut stack, token_index)?;
                    stack.push(rolling_zscore(&input, *window));
                }
            }
        }

        if stack.len() != 1 {
            return Err(EvalError::FinalStackDepth { depth: stack.len() });
        }

        Ok(stack.pop().expect("stack length checked"))
    }
}

fn validate_window(token_index: usize, window: usize) -> Result<(), EvalError> {
    if window == 0 {
        Err(EvalError::InvalidWindow {
            token_index,
            window,
        })
    } else {
        Ok(())
    }
}

fn pop_one(stack: &mut Vec<Vec<f64>>, token_index: usize) -> Result<Vec<f64>, EvalError> {
    stack
        .pop()
        .ok_or(EvalError::StackUnderflow { token_index })
}

fn apply_unary<F>(
    stack: &mut Vec<Vec<f64>>,
    token_index: usize,
    op: F,
) -> Result<(), EvalError>
where
    F: Fn(f64) -> f64,
{
    let input = pop_one(stack, token_index)?;
    let output = input
        .into_iter()
        .map(|x| if x.is_finite() { op(x) } else { f64::NAN })
        .collect();
    stack.push(output);
    Ok(())
}

fn apply_binary<F>(
    stack: &mut Vec<Vec<f64>>,
    token_index: usize,
    op: F,
) -> Result<(), EvalError>
where
    F: Fn(f64, f64) -> f64,
{
    if stack.len() < 2 {
        return Err(EvalError::StackUnderflow { token_index });
    }
    let rhs = stack.pop().expect("length checked");
    let lhs = stack.pop().expect("length checked");
    let output = lhs
        .into_iter()
        .zip(rhs)
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                let value = op(a, b);
                if value.is_finite() {
                    value
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        })
        .collect();
    stack.push(output);
    Ok(())
}

fn delay(input: &[f64], lag: usize) -> Vec<f64> {
    if lag == 0 {
        return input.to_vec();
    }

    let mut out = vec![f64::NAN; input.len()];
    if lag >= input.len() {
        return out;
    }
    out[lag..].copy_from_slice(&input[..input.len() - lag]);
    out
}

fn rolling_mean(input: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; input.len()];
    for end in 0..input.len() {
        if end + 1 < window {
            continue;
        }
        let slice = &input[end + 1 - window..=end];
        if slice.iter().any(|x| !x.is_finite()) {
            continue;
        }
        out[end] = slice.iter().sum::<f64>() / window as f64;
    }
    out
}

fn rolling_std(input: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; input.len()];
    for end in 0..input.len() {
        if end + 1 < window {
            continue;
        }
        let slice = &input[end + 1 - window..=end];
        if slice.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let mean = slice.iter().sum::<f64>() / window as f64;
        let variance = slice
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<f64>()
            / window as f64;
        out[end] = variance.sqrt();
    }
    out
}

fn rolling_zscore(input: &[f64], window: usize) -> Vec<f64> {
    let means = rolling_mean(input, window);
    let stds = rolling_std(input, window);
    input
        .iter()
        .zip(means.iter().zip(stds.iter()))
        .map(|(x, (mean, std))| {
            if !x.is_finite() || !mean.is_finite() || !std.is_finite() || *std <= STD_EPS {
                f64::NAN
            } else {
                (*x - *mean) / *std
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_or_nan(lhs: f64, rhs: f64) {
        if lhs.is_nan() && rhs.is_nan() {
            return;
        }
        assert!((lhs - rhs).abs() < 1e-12, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn delay_never_reads_the_future() {
        let mut frame = Frame::new(5);
        frame.insert("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let formula = Formula::new(vec![Token::Input("x".into()), Token::Delay(2)]);
        let out = formula.evaluate(&frame).unwrap();
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert_eq!(&out[2..], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn rolling_mean_has_strict_nan_warmup() {
        let mut frame = Frame::new(5);
        frame.insert("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let formula = Formula::new(vec![Token::Input("x".into()), Token::RollingMean(3)]);
        let out = formula.evaluate(&frame).unwrap();
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert_eq!(&out[2..], &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn future_mutation_cannot_change_past_outputs() {
        let formula = Formula::new(vec![
            Token::Input("x".into()),
            Token::RollingZScore(3),
            Token::Delay(1),
        ]);

        let mut before = Frame::new(8);
        before
            .insert("x", vec![1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 8.0, 9.0])
            .unwrap();
        let out_before = formula.evaluate(&before).unwrap();

        let mut after = before.clone();
        after
            .insert("x", vec![1.0, 2.0, 4.0, 3.0, 5.0, 7000.0, -8000.0, 9000.0])
            .unwrap();
        let out_after = formula.evaluate(&after).unwrap();

        for index in 0..5 {
            assert_same_or_nan(out_before[index], out_after[index]);
        }
    }

    #[test]
    fn malformed_postfix_formula_is_rejected() {
        let frame = Frame::new(3);
        let formula = Formula::new(vec![Token::Add]);
        assert_eq!(
            formula.evaluate(&frame),
            Err(EvalError::StackUnderflow { token_index: 0 })
        );
    }

    #[test]
    fn zero_window_is_rejected() {
        let mut frame = Frame::new(3);
        frame.insert("x", vec![1.0, 2.0, 3.0]).unwrap();
        let formula = Formula::new(vec![Token::Input("x".into()), Token::RollingMean(0)]);
        assert_eq!(
            formula.evaluate(&frame),
            Err(EvalError::InvalidWindow {
                token_index: 1,
                window: 0
            })
        );
    }

    #[test]
    fn canonical_hash_is_deterministic_and_parameter_sensitive() {
        let a = Formula::new(vec![Token::Input("close".into()), Token::RollingMean(20)]);
        let b = Formula::new(vec![Token::Input("close".into()), Token::RollingMean(20)]);
        let c = Formula::new(vec![Token::Input("close".into()), Token::RollingMean(21)]);
        assert_eq!(a.canonical(), b.canonical());
        assert_eq!(a.stable_hash64(), b.stable_hash64());
        assert_ne!(a.stable_hash64(), c.stable_hash64());
    }
}
