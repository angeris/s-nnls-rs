use comfy_table::{Cell, CellAlignment, ContentArrangement, Table, presets};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    ConvergedGradient,
    ConvergedStep,
    ConvergedCost,
    MaxIterations,
    NumericalFailure,
}

#[derive(Debug, Clone)]
pub struct SolverStats {
    pub status: SolveStatus,
    pub iterations: usize,
    pub cost: f64,
    pub grad_inf: f64,
    pub step_norm: f64,
    pub lambda: f64,
}

#[derive(Debug, Clone)]
pub struct IterationReport {
    pub iteration: usize,
    pub cost: f64,
    pub trial_cost: f64,
    pub rho: f64,
    pub lambda: f64,
    pub step_norm: f64,
    pub grad_inf: f64,
    pub accepted: bool,
}

pub(crate) fn emit_line(line: &str) {
    if log::log_enabled!(log::Level::Info) {
        log::info!("{line}");
    } else {
        println!("{line}");
    }
}

pub trait Reporter {
    fn on_iteration(&mut self, report: &IterationReport);
    fn on_finish(&mut self) {}
}

pub struct StdoutReporter {
    rows: Vec<IterationReport>,
}

impl StdoutReporter {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }
}

impl Default for StdoutReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Reporter for StdoutReporter {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.rows.push(report.clone());
    }

    fn on_finish(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        if !log::log_enabled!(log::Level::Info) {
            println!();
        }
        let mut table = Table::new();
        table.load_preset(presets::UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);
        table.set_header(vec![
            Cell::new("iter").set_alignment(CellAlignment::Right),
            Cell::new("cost").set_alignment(CellAlignment::Right),
            Cell::new("trial").set_alignment(CellAlignment::Right),
            Cell::new("rho").set_alignment(CellAlignment::Right),
            Cell::new("lambda").set_alignment(CellAlignment::Right),
            Cell::new("step").set_alignment(CellAlignment::Right),
            Cell::new("grad").set_alignment(CellAlignment::Right),
            Cell::new("accepted"),
        ]);
        for row in &self.rows {
            table.add_row(vec![
                Cell::new(row.iteration).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.4e}", row.cost)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.4e}", row.trial_cost)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.3}", row.rho)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.1e}", row.lambda)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.1e}", row.step_norm)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.1e}", row.grad_inf)).set_alignment(CellAlignment::Right),
                Cell::new(if row.accepted { "yes" } else { "no" }),
            ]);
        }

        for line in table.to_string().lines() {
            emit_line(line);
        }
        self.rows.clear();
    }
}
