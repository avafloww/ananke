//! The final admission gate: does the CPU-side reservation the packer built
//! up actually fit in host RAM?

use crate::{
    allocator::placement::{packer::Packer, reserve::sum_reserved, types::PackError},
    config::DeviceSlot,
};

impl<'a> Packer<'a> {
    /// Reject the pack if the bytes the packer wants to keep on the host exceed
    /// the available host RAM (minus the configured `[devices.cpu] reserved_gb`).
    /// Skipped when the snapshot carries no CPU info.
    pub(crate) fn check_cpu_capacity(&self) -> Result<(), PackError> {
        let needed = self.per_device.get(&DeviceSlot::Cpu).copied().unwrap_or(0);
        if needed == 0 {
            return Ok(());
        }
        // Mirror `gpu_available`'s two views. The optimistic path trusts
        // the pledge book (`total - reserved-by-others`) instead of live
        // free RAM — previews for an already-running service would
        // otherwise measure the service's own resident memory as
        // unavailable and report that it "cannot fit" the placement it
        // is actively holding (first seen with GLM-5.2's ~180 GiB CPU
        // side; smaller hybrids fit inside the leftover RAM by luck).
        let slot = DeviceSlot::Cpu;
        let Some(free) = self.snapshot.free_bytes(&slot) else {
            return Ok(());
        };
        let total = self.snapshot.total_bytes(&slot).unwrap_or(free);
        let reserved_here = sum_reserved(self.reserved, &slot, &self.svc.name);
        let via_pledge = total.saturating_sub(reserved_here);
        let avail = if self.optimistic_remaining {
            via_pledge
        } else {
            free.min(via_pledge)
        };
        let available = avail.saturating_sub(self.svc.reserves.cpu_bytes);
        if needed > available {
            return Err(PackError::CpuDoesNotFit { needed, available });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocator::{
            AllocationTable,
            placement::{
                entry::{pack, pack_optimistic},
                test_support::{GIB, moe_estimate, moe_svc},
            },
        },
        config::OffloadMode,
        devices::CpuSnapshot,
    };

    /// The "preview a running hybrid" shape: the service's own ~10 GiB CPU
    /// side has consumed live RAM (free is tiny), but the pledge book says
    /// the memory is spoken for by *this* service. Optimistic packing must
    /// succeed (the preview of the placement it already holds); conservative
    /// packing must still respect live free RAM.
    #[test]
    fn optimistic_cpu_check_trusts_pledges_over_live_free_ram() {
        let e = moe_estimate(10, 100, 300); // 10 GiB model, 1 GiB non-expert
        let mut snap = crate::allocator::placement::test_support::snapshot(&[4]);
        snap.cpu = Some(CpuSnapshot {
            total_bytes: 128 * GIB,
            available_bytes: 2 * GIB, // the running child ate the rest
        });
        let alloc = AllocationTable::new();
        let svc = moe_svc(OffloadMode::Auto);

        assert!(
            pack_optimistic(&e, &svc, &snap, &alloc).is_ok(),
            "optimistic preview must trust total - reserved-by-others"
        );
        assert!(
            matches!(
                pack(&e, &svc, &snap, &alloc),
                Err(PackError::CpuDoesNotFit { .. })
            ),
            "conservative pack must still respect live free RAM"
        );
    }
}
