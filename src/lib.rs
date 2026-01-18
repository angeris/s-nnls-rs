//! Sparse nonlinear least squares with a Levenberg-Marquardt step.
//!
//! The solver is designed for fixed sparsity patterns and reusable allocations.

mod pattern;
mod report;
mod solver;

pub use pattern::{JacobianPattern, PatternError};
pub use report::{IterationReport, Reporter, SolveStatus, SolverStats, StdoutReporter};
pub use solver::{JacobianValuesMut, LmSolver, Problem, SolveError, SolverError, SolverOptions};
