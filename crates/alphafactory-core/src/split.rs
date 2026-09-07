use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkForwardConfig {
    pub initial_train: usize,
    pub validation: usize,
    pub test: usize,
    pub step: usize,
    pub purge: usize,
    pub embargo: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkForwardFold {
    pub train: Range<usize>,
    pub validation: Range<usize>,
    pub test: Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    EmptyTrain,
    EmptyValidation,
    EmptyTest,
    ZeroStep,
}

pub fn build_walk_forward_folds(
    total_len: usize,
    config: WalkForwardConfig,
) -> Result<Vec<WalkForwardFold>, SplitError> {
    if config.initial_train == 0 {
        return Err(SplitError::EmptyTrain);
    }
    if config.validation == 0 {
        return Err(SplitError::EmptyValidation);
    }
    if config.test == 0 {
        return Err(SplitError::EmptyTest);
    }
    if config.step == 0 {
        return Err(SplitError::ZeroStep);
    }

    let mut folds = Vec::new();
    let mut train_end = config.initial_train;

    loop {
        let validation_start = match train_end.checked_add(config.purge) {
            Some(value) => value,
            None => break,
        };
        let validation_end = match validation_start.checked_add(config.validation) {
            Some(value) => value,
            None => break,
        };
        let test_start = match validation_end.checked_add(config.embargo) {
            Some(value) => value,
            None => break,
        };
        let test_end = match test_start.checked_add(config.test) {
            Some(value) => value,
            None => break,
        };

        if test_end > total_len {
            break;
        }

        folds.push(WalkForwardFold {
            train: 0..train_end,
            validation: validation_start..validation_end,
            test: test_start..test_end,
        });

        train_end = match train_end.checked_add(config.step) {
            Some(value) => value,
            None => break,
        };
    }

    Ok(folds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_respects_purge_and_embargo() {
        let config = WalkForwardConfig {
            initial_train: 100,
            validation: 20,
            test: 20,
            step: 20,
            purge: 5,
            embargo: 3,
        };
        let folds = build_walk_forward_folds(180, config).unwrap();
        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0].train, 0..100);
        assert_eq!(folds[0].validation, 105..125);
        assert_eq!(folds[0].test, 128..148);
        assert_eq!(folds[1].train, 0..120);
        assert_eq!(folds[1].validation, 125..145);
        assert_eq!(folds[1].test, 148..168);
    }

    #[test]
    fn splitter_never_crosses_dataset_end() {
        let config = WalkForwardConfig {
            initial_train: 50,
            validation: 10,
            test: 10,
            step: 10,
            purge: 2,
            embargo: 2,
        };
        let folds = build_walk_forward_folds(73, config).unwrap();
        assert_eq!(folds.len(), 1);
        assert!(folds[0].test.end <= 73);
    }

    #[test]
    fn invalid_zero_lengths_are_rejected() {
        let config = WalkForwardConfig {
            initial_train: 0,
            validation: 10,
            test: 10,
            step: 10,
            purge: 0,
            embargo: 0,
        };
        assert_eq!(
            build_walk_forward_folds(100, config),
            Err(SplitError::EmptyTrain)
        );
    }
}
