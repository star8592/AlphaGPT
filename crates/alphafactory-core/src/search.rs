use std::collections::BTreeSet;

use crate::formula::{Formula, Token};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    NoInputs,
    NoWindows,
}

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub inputs: Vec<String>,
    pub windows: Vec<usize>,
}

impl SearchSpace {
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.inputs.is_empty() {
            return Err(SearchError::NoInputs);
        }
        if self.windows.is_empty() || self.windows.iter().any(|window| *window == 0) {
            return Err(SearchError::NoWindows);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DeterministicFormulaGenerator {
    rng: SplitMix64,
}

impl DeterministicFormulaGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }

    pub fn generate(
        &mut self,
        space: &SearchSpace,
        max_depth: usize,
    ) -> Result<Formula, SearchError> {
        space.validate()?;
        let mut tokens = Vec::new();
        self.emit_expression(space, max_depth, &mut tokens);
        Ok(Formula::new(tokens))
    }

    fn emit_expression(&mut self, space: &SearchSpace, depth: usize, out: &mut Vec<Token>) {
        if depth == 0 {
            self.emit_terminal(space, out);
            return;
        }

        match self.rng.choose(12) {
            0 | 1 | 2 => self.emit_terminal(space, out),
            3 => {
                self.emit_expression(space, depth - 1, out);
                self.emit_expression(space, depth - 1, out);
                out.push(Token::Add);
            }
            4 => {
                self.emit_expression(space, depth - 1, out);
                self.emit_expression(space, depth - 1, out);
                out.push(Token::Sub);
            }
            5 => {
                self.emit_expression(space, depth - 1, out);
                self.emit_expression(space, depth - 1, out);
                out.push(Token::Mul);
            }
            6 => {
                self.emit_expression(space, depth - 1, out);
                self.emit_expression(space, depth - 1, out);
                out.push(Token::SafeDiv);
            }
            7 => {
                self.emit_expression(space, depth - 1, out);
                out.push(Token::Neg);
            }
            8 => {
                self.emit_expression(space, depth - 1, out);
                out.push(Token::Abs);
            }
            9 => {
                self.emit_expression(space, depth - 1, out);
                out.push(Token::Delay(self.pick_window(space)));
            }
            10 => {
                self.emit_expression(space, depth - 1, out);
                out.push(Token::RollingMean(self.pick_window(space)));
            }
            _ => {
                self.emit_expression(space, depth - 1, out);
                if self.rng.choose(2) == 0 {
                    out.push(Token::RollingStd(self.pick_window(space)));
                } else {
                    out.push(Token::RollingZScore(self.pick_window(space)));
                }
            }
        }
    }

    fn emit_terminal(&mut self, space: &SearchSpace, out: &mut Vec<Token>) {
        if self.rng.choose(5) == 0 {
            let value = match self.rng.choose(5) {
                0 => -2.0,
                1 => -1.0,
                2 => 0.0,
                3 => 1.0,
                _ => 2.0,
            };
            out.push(Token::Const(value));
        } else {
            let index = self.rng.choose(space.inputs.len());
            out.push(Token::Input(space.inputs[index].clone()));
        }
    }

    fn pick_window(&mut self, space: &SearchSpace) -> usize {
        let index = self.rng.choose(space.windows.len());
        space.windows[index]
    }
}

#[derive(Debug, Default)]
pub struct AlphaRegistry {
    canonical_formulas: BTreeSet<String>,
}

impl AlphaRegistry {
    pub fn register(&mut self, formula: &Formula) -> bool {
        self.canonical_formulas.insert(formula.canonical())
    }

    pub fn contains(&self, formula: &Formula) -> bool {
        self.canonical_formulas.contains(&formula.canonical())
    }

    pub fn len(&self) -> usize {
        self.canonical_formulas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.canonical_formulas.is_empty()
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn choose(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::Frame;

    fn space() -> SearchSpace {
        SearchSpace {
            inputs: vec!["ret".into(), "volume".into()],
            windows: vec![2, 3, 5, 10],
        }
    }

    #[test]
    fn same_seed_generates_same_formula_sequence() {
        let mut lhs = DeterministicFormulaGenerator::new(42);
        let mut rhs = DeterministicFormulaGenerator::new(42);

        for _ in 0..100 {
            let a = lhs.generate(&space(), 4).unwrap();
            let b = rhs.generate(&space(), 4).unwrap();
            assert_eq!(a.canonical(), b.canonical());
            assert_eq!(a.stable_hash64(), b.stable_hash64());
        }
    }

    #[test]
    fn generated_formulas_are_structurally_executable() {
        let mut frame = Frame::new(32);
        frame
            .insert("ret", (0..32).map(|x| x as f64 / 100.0).collect())
            .unwrap();
        frame
            .insert("volume", (1..=32).map(|x| x as f64).collect())
            .unwrap();

        let mut generator = DeterministicFormulaGenerator::new(7);
        for _ in 0..500 {
            let formula = generator.generate(&space(), 5).unwrap();
            let output = formula.evaluate(&frame).unwrap();
            assert_eq!(output.len(), 32);
        }
    }

    #[test]
    fn registry_rejects_exact_duplicates() {
        let formula = Formula::new(vec![Token::Input("ret".into()), Token::RollingMean(5)]);
        let mut registry = AlphaRegistry::default();
        assert!(registry.register(&formula));
        assert!(!registry.register(&formula));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn empty_search_spaces_are_rejected() {
        let mut generator = DeterministicFormulaGenerator::new(1);
        let bad = SearchSpace {
            inputs: Vec::new(),
            windows: vec![5],
        };
        assert_eq!(generator.generate(&bad, 3), Err(SearchError::NoInputs));
    }
}
