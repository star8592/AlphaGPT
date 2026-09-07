#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCostModel {
    pub fee_bps_one_way: f64,
    pub slippage_bps_one_way: f64,
}

impl LinearCostModel {
    pub fn cost_for_turnover(&self, turnover: f64) -> f64 {
        if !turnover.is_finite() || turnover < 0.0 {
            return f64::NAN;
        }
        let bps = self.fee_bps_one_way + self.slippage_bps_one_way;
        turnover * bps / 10_000.0
    }

    pub fn net_return(&self, gross_return: f64, turnover: f64, carry_cost: f64) -> f64 {
        if !gross_return.is_finite() || !carry_cost.is_finite() {
            return f64::NAN;
        }
        let trading_cost = self.cost_for_turnover(turnover);
        if !trading_cost.is_finite() {
            return f64::NAN;
        }
        gross_return - trading_cost - carry_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs_are_explicit_and_scale_with_turnover() {
        let model = LinearCostModel {
            fee_bps_one_way: 5.0,
            slippage_bps_one_way: 10.0,
        };
        assert!((model.cost_for_turnover(1.0) - 0.0015).abs() < 1e-12);
        assert!((model.cost_for_turnover(2.0) - 0.0030).abs() < 1e-12);
    }

    #[test]
    fn net_return_includes_carry_cost() {
        let model = LinearCostModel {
            fee_bps_one_way: 5.0,
            slippage_bps_one_way: 5.0,
        };
        let net = model.net_return(0.02, 1.0, 0.001);
        assert!((net - 0.018).abs() < 1e-12);
    }
}
