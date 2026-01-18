# s-nnls-rs

Sparse nonlinear least squares (NLLS) solver using a Levenberg-Marquardt step and sparse QR.

## How it works
- Minimize `0.5 * ||r(x)||^2` for residuals `r(x)` with Jacobian `J(x)`.
- Build `[J; sqrt(lambda) I] p = [-r; 0]` and solve with sparse QR.
- Update `lambda` based on actual vs predicted decrease to accept or reject the step.

## Usage
```rust
use s_nnls_rs::{JacobianPattern, JacobianValuesMut, LmSolver, Problem, SolverOptions};
use faer_core::Parallelism;

struct OneD;
impl Problem for OneD {
    fn residuals(&mut self, x: &[f64], r: &mut [f64]) {
        r[0] = x[0] - 1.0;
    }
    fn jacobian(&mut self, _x: &[f64], jac: &mut JacobianValuesMut<'_>) {
        jac.values_of_col_mut(0)[0] = 1.0;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CSC pattern with 0-based indices.
    let pattern = JacobianPattern::new(1, 1, vec![0, 1], vec![0])?;
    let mut solver = LmSolver::new(pattern, Parallelism::None)?;
    let mut problem = OneD;
    let mut x = vec![0.0];
    let stats = solver.solve(&mut problem, &mut x, &SolverOptions::default(), None)?;
    println!("status: {:?}, x: {:?}", stats.status, x);
    Ok(())
}
```

## Notes
- `JacobianPattern` uses 0-based indices with sorted rows per column.
- Set `SolverOptions { verbose: true, .. }` to print iteration diagnostics.
