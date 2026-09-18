//! Elementwise regions planned at prepare time.
//!
//! The prepared executor dispatches one command per scheduled operation. This
//! module plans the regions that may instead execute as one fused command, so
//! the decision is made once (outside the timed region) with the same
//! segmentation and eligibility the unprepared path uses.
//!
//! A region is only planned when the segmented executor would also accept it:
//! `segment_exec_program` extracts candidates and `build_elementwise_fusion_plan`
//! proves eligibility. The region keeps the original instruction range, the
//! consecutive scheduled nodes it covers, its external inputs, **all live-outs**
//! (the segment's outputs are exactly the values used at or after the segment or
//! returned by the program), and the input liveness the boundary needs.

use std::collections::HashMap;
use std::ops::Range;

use tenferro_tensor::backend::ElementwiseFusionPlan;

use crate::exec::{ExecInstruction, ExecProgram};
use crate::runtime::schedule::{ScheduledGraph, ScheduledNode};
use crate::segment::{build_elementwise_fusion_plan, segment_exec_program, Segment};

/// One elementwise region that may execute as a single fused command.
#[derive(Debug)]
pub(crate) struct ElementwiseRegion {
    /// Instruction indices in the staging program.
    #[allow(
        dead_code,
        reason = "consumed by the region executor in the prepared execution path"
    )]
    pub(crate) instruction_range: Range<usize>,
    /// Scheduled operation nodes covered by the region, consecutive and in
    /// schedule order.
    pub(crate) node_indices: Vec<usize>,
    /// Instructions the region executes, in order.
    pub(crate) instructions: Vec<ExecInstruction>,
    /// Slots read by the region and produced outside it.
    pub(crate) input_slots: Vec<usize>,
    /// Every live-out of the region: produced inside, used at or after the
    /// region, or returned by the program.
    pub(crate) output_slots: Vec<usize>,
    /// For each `input_slots` entry, whether the region holds the last use.
    pub(crate) input_last_use: Vec<bool>,
    /// Fusion plan built by the shared eligibility check.
    pub(crate) plan: ElementwiseFusionPlan,
}

impl ElementwiseRegion {
    pub(crate) fn retained_bytes(&self) -> Option<usize> {
        let instructions = self
            .instructions
            .len()
            .checked_mul(std::mem::size_of::<ExecInstruction>())?;
        let nodes = self
            .node_indices
            .len()
            .checked_mul(std::mem::size_of::<usize>())?;
        let inputs = self
            .input_slots
            .len()
            .checked_mul(std::mem::size_of::<usize>())?;
        let outputs = self
            .output_slots
            .len()
            .checked_mul(std::mem::size_of::<usize>())?;
        let last_use = self
            .input_last_use
            .len()
            .checked_mul(std::mem::size_of::<bool>())?;
        let plan = self
            .plan
            .ops()
            .len()
            .checked_mul(std::mem::size_of::<
                tenferro_tensor::backend::ElementwiseFusionInst,
            >())?
            .checked_add(
                self.plan
                    .outputs()
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?;
        instructions
            .checked_add(nodes)?
            .checked_add(inputs)?
            .checked_add(outputs)?
            .checked_add(last_use)?
            .checked_add(plan)?
            .checked_add(std::mem::size_of::<Self>())
    }
}

/// Plan the elementwise regions of a prepared program.
///
/// Regions are skipped when the segmentation only yields a candidate that the
/// shared eligibility check rejects, or when the candidate does not map to
/// consecutive scheduled operation nodes: those keep the per-instruction
/// dispatch path.
pub(crate) fn plan_elementwise_regions(
    staging: &ExecProgram,
    schedule: &ScheduledGraph,
) -> Vec<ElementwiseRegion> {
    let node_of_instruction: HashMap<usize, usize> = schedule
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| match node {
            ScheduledNode::Operation(operation) => {
                Some((operation.instruction_index(), node_index))
            }
            _ => None,
        })
        .collect();

    let mut regions = Vec::new();
    let mut cursor = 0usize;
    for segment in segment_exec_program(staging) {
        let Segment::Fused {
            instructions,
            input_slots,
            output_slots,
            last_use,
        } = segment
        else {
            cursor += 1;
            continue;
        };
        let start = cursor;
        cursor += instructions.len();
        if instructions.len() < 2 {
            continue;
        }
        let Some(plan) = build_elementwise_fusion_plan(&instructions, &input_slots, &output_slots)
        else {
            continue;
        };
        let node_indices: Option<Vec<usize>> = (start..start + instructions.len())
            .map(|instruction| node_of_instruction.get(&instruction).copied())
            .collect();
        let Some(node_indices) = node_indices else {
            continue;
        };
        if !node_indices.windows(2).all(|pair| pair[1] == pair[0] + 1) {
            continue;
        }
        regions.push(ElementwiseRegion {
            instruction_range: start..start + instructions.len(),
            node_indices,
            instructions,
            input_slots,
            output_slots,
            input_last_use: last_use,
            plan,
        });
    }
    regions
}

/// Execution counters for planned elementwise regions.
///
/// `fused` counts regions executed as one fused command; `fallbacks` counts
/// regions whose fusion the backend rejected and which therefore dispatched
/// their instructions one by one. Together they are the runtime evidence for
/// execution-path parity tests.
#[derive(Debug, Default)]
pub(crate) struct RegionExecutionCounters {
    fused: std::sync::atomic::AtomicUsize,
    fallbacks: std::sync::atomic::AtomicUsize,
}

impl RegionExecutionCounters {
    pub(crate) fn record_fused(&self) {
        self.fused
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn record_fallback(&self) {
        self.fallbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn fused(&self) -> usize {
        self.fused.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn fallbacks(&self) -> usize {
        self.fallbacks.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Number of regions and the instructions they cover, for parity tests.
pub(crate) fn region_summary(regions: &[ElementwiseRegion]) -> (usize, usize) {
    (
        regions.len(),
        regions.iter().map(|region| region.instructions.len()).sum(),
    )
}
