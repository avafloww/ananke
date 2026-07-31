//! Tests for [`crate::supervise::spawn::llama_cpp::render_llama_cpp_argv`] / [`crate::supervise::spawn::llama_cpp`]'s flag rendering,
//! exercised end-to-end through [`crate::supervise::spawn::render_argv`].

use std::{collections::BTreeMap, path::PathBuf};

use smol_str::SmolStr;

use crate::{
    allocator::placement::CommandArgs,
    config::validate::{
        DeviceSlot, GenerationStallTrigger, Lifecycle, NumaStrategy, PlacementPolicy,
        ServiceConfig, SplitMode,
        test_fixtures::{expect_llama_cpp, minimal_service},
    },
    devices::Allocation,
    supervise::spawn::render_argv,
};

fn base_service() -> ServiceConfig {
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Gpu(0), 10240);
    let mut svc = minimal_service("demo");
    svc.port = 11435;
    svc.private_port = 41000;
    svc.lifecycle = Lifecycle::Persistent;
    svc.placement_override = placement;
    svc.placement_policy = PlacementPolicy::GpuOnly;
    let lc = expect_llama_cpp(&mut svc);
    lc.model = PathBuf::from("/m/x.gguf");
    lc.context = Some(8192);
    lc.flash_attn = Some(true);
    lc.cache_type_k = Some(SmolStr::new("q8_0"));
    lc.cache_type_v = Some(SmolStr::new("q8_0"));
    svc
}
#[test]
fn renders_core_flags() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert_eq!(cmd.binary, "llama-server");
    assert!(cmd.args.contains(&"-m".to_string()));
    assert!(cmd.args.iter().any(|a| a == "/m/x.gguf"));
    assert!(cmd.args.iter().any(|a| a == "-c"));
    assert!(cmd.args.iter().any(|a| a == "8192"));
    assert!(cmd.args.iter().any(|a| a == "-fa"));
    assert!(cmd.args.iter().any(|a| a == "--port"));
    assert!(cmd.args.iter().any(|a| a == "41000"));
    assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "0");
}

#[test]
fn renders_ik_runtime_flags() {
    use crate::config::{IkSettings, RuntimeConfig};
    let mut svc = base_service();
    let mut placement = BTreeMap::new();
    placement.insert(DeviceSlot::Gpu(0), 24576);
    placement.insert(DeviceSlot::Gpu(1), 24576);
    svc.placement_override = placement;
    {
        let lc = expect_llama_cpp(&mut svc);
        lc.runtime = RuntimeConfig::Ik(IkSettings {
            mla: Some(1),
            dsa: true,
            attn_max_batch: Some(512),
            runtime_repack: false,
        });
        lc.cache_type_k = None;
        lc.cache_type_v = None;
        lc.ubatch_size = Some(2048);
        lc.spec_type = Some(SmolStr::new("mtp:n_max=4,p_min=0.5"));
    }
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    let a = &cmd.args;
    let i = a.iter().position(|x| x == "-mla").unwrap();
    assert_eq!(a[i + 1], "1");
    assert!(a.iter().any(|x| x == "-dsa"));
    assert!(a.iter().any(|x| x == "-fidx"));
    let i = a.iter().position(|x| x == "-amb").unwrap();
    assert_eq!(a[i + 1], "512");
    assert!(!a.iter().any(|x| x == "-rtr"));
}

#[test]
fn mainline_runtime_emits_no_ik_flags() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    for flag in ["-mla", "-dsa", "-fidx", "-amb", "-rtr"] {
        assert!(!cmd.args.iter().any(|a| a == flag), "unexpected {flag}");
    }
}

#[test]
fn renders_numa_flag() {
    let mut svc = base_service();
    expect_llama_cpp(&mut svc).numa = Some(NumaStrategy::Distribute);
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    let i = cmd.args.iter().position(|a| a == "--numa").unwrap();
    assert_eq!(cmd.args[i + 1], "distribute");
}

#[test]
fn omits_numa_flag_when_unset() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(!cmd.args.iter().any(|a| a == "--numa"));
}

#[test]
fn override_tensor_rules_emit_one_comma_joined_flag() {
    // Current llama.cpp honours only the last `-ot`; multiple rules must
    // therefore collapse into a single comma-joined flag, not one per rule.
    let mut svc = base_service();
    expect_llama_cpp(&mut svc).override_tensor = vec![
        r"blk\.0\.ffn_(gate|up|down)_exps=CPU".into(),
        r"blk\.1\.ffn_(gate|up|down)_exps=CUDA1".into(),
    ];
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    let ot_positions: Vec<usize> = cmd
        .args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a == "-ot").then_some(i))
        .collect();
    assert_eq!(
        ot_positions.len(),
        1,
        "exactly one -ot flag expected, got {:?}",
        cmd.args
    );
    assert_eq!(
        cmd.args[ot_positions[0] + 1],
        r"blk\.0\.ffn_(gate|up|down)_exps=CPU,blk\.1\.ffn_(gate|up|down)_exps=CUDA1"
    );
}

#[test]
fn renders_mtp_spec_flags() {
    let mut svc = base_service();
    {
        let lc = expect_llama_cpp(&mut svc);
        lc.parallel = Some(2);
        lc.spec_type = Some(SmolStr::new("draft-mtp"));
        lc.spec_draft_n_max = Some(2);
    }
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    let np = cmd.args.iter().position(|a| a == "-np").unwrap();
    assert_eq!(cmd.args[np + 1], "2");
    let st = cmd.args.iter().position(|a| a == "--spec-type").unwrap();
    assert_eq!(cmd.args[st + 1], "draft-mtp");
    let n = cmd
        .args
        .iter()
        .position(|a| a == "--spec-draft-n-max")
        .unwrap();
    assert_eq!(cmd.args[n + 1], "2");
}

#[test]
fn renders_separate_draft_and_server_flags() {
    let mut svc = base_service();
    {
        let lc = expect_llama_cpp(&mut svc);
        lc.flash_attn = Some(true);
        lc.cache_type_k = Some(SmolStr::new("f16"));
        lc.cache_type_v = Some(SmolStr::new("f16"));
        lc.parallel = Some(4);
        lc.kv_unified = Some(true);
        lc.cache_idle_slots = Some(false);
        lc.metrics = Some(true);
        lc.slots = Some(true);
        lc.spec_type = Some(SmolStr::new("draft-mtp"));
        lc.spec_draft_n_max = Some(2);
        lc.draft_model = Some(PathBuf::from("/m/mtp-draft.gguf"));
    }
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(cmd.args.iter().any(|a| a == "--kv-unified"));
    assert!(cmd.args.iter().any(|a| a == "--no-cache-idle-slots"));
    assert!(cmd.args.iter().any(|a| a == "--metrics"));
    assert!(cmd.args.iter().any(|a| a == "--slots"));
    let md = cmd.args.iter().position(|a| a == "-md").unwrap();
    assert_eq!(cmd.args[md + 1], "/m/mtp-draft.gguf");
}

#[test]
fn omits_server_toggle_flags_when_unset() {
    // The base service leaves kv_unified/cache_idle_slots/metrics/slots
    // and draft_model unset; none of their flags should appear.
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(!cmd.args.iter().any(|a| a == "--kv-unified"));
    assert!(!cmd.args.iter().any(|a| a == "--no-cache-idle-slots"));
    assert!(!cmd.args.iter().any(|a| a == "--metrics"));
    assert!(!cmd.args.iter().any(|a| a == "--slots"));
    assert!(!cmd.args.iter().any(|a| a == "-md"));
}

#[test]
fn embedding_modality_injects_embeddings_flag() {
    let mut svc = base_service();
    svc.modality = ananke_api::shared::Modality::Embedding;
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(cmd.args.iter().any(|a| a == "--embeddings"));
    // Chat services don't get it.
    let svc = base_service();
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(!cmd.args.iter().any(|a| a == "--embeddings"));
}

#[test]
fn generation_stall_injects_metrics_flag() {
    let mut svc = base_service();
    svc.auto_restart.generation_stall = Some(GenerationStallTrigger::default());
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(cmd.args.iter().any(|a| a == "--metrics"));
}

#[test]
fn explicit_metrics_false_suppresses_injection() {
    let mut svc = base_service();
    svc.auto_restart.generation_stall = Some(GenerationStallTrigger::default());
    expect_llama_cpp(&mut svc).metrics = Some(false);
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(!cmd.args.iter().any(|a| a == "--metrics"));
}

#[test]
fn omits_spec_flags_when_unset() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(!cmd.args.iter().any(|a| a == "--spec-type"));
    assert!(!cmd.args.iter().any(|a| a == "--spec-draft-n-max"));
}

#[test]
fn renders_mmproj_when_present() {
    let mut svc = base_service();
    expect_llama_cpp(&mut svc).mmproj = Some(PathBuf::from("/m/x-mmproj.gguf"));
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    let idx = cmd.args.iter().position(|a| a == "--mmproj").unwrap();
    assert_eq!(cmd.args[idx + 1], "/m/x-mmproj.gguf");
}

#[test]
fn cpu_only_renders_ngl_zero_and_empty_cuda_env() {
    let mut svc = base_service();
    svc.placement_policy = PlacementPolicy::CpuOnly;
    svc.placement_override.clear();
    svc.placement_override.insert(DeviceSlot::Cpu, 10240);
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    let ngl_idx = cmd.args.iter().position(|a| a == "-ngl").unwrap();
    assert_eq!(cmd.args[ngl_idx + 1], "0");
    assert_eq!(cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "");
}

#[test]
fn placement_cmd_args_override_ngl_and_add_tensor_split() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let ca = CommandArgs {
        ngl: Some(24),
        tensor_split: Some(vec![12, 12]),
        override_tensor: vec![],
        split_mode: None,
        main_gpu: None,
        n_cpu_moe: None,
    };
    let cmd = render_argv(&svc, &alloc, Some(&ca)).unwrap();
    let ngl_idx = cmd.args.iter().position(|a| a == "-ngl").unwrap();
    assert_eq!(cmd.args[ngl_idx + 1], "24");
    let ts_idx = cmd.args.iter().position(|a| a == "--tensor-split").unwrap();
    assert_eq!(cmd.args[ts_idx + 1], "12,12");
    // Layer split (split_mode = None) must not emit --split-mode/--main-gpu.
    assert!(!cmd.args.iter().any(|a| a == "--split-mode"));
    assert!(!cmd.args.iter().any(|a| a == "--main-gpu"));
}

#[test]
fn placement_cmd_args_emit_n_cpu_moe() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let ca = CommandArgs {
        ngl: Some(999),
        tensor_split: None,
        override_tensor: vec![],
        split_mode: None,
        main_gpu: None,
        n_cpu_moe: Some(25),
    };
    let cmd = render_argv(&svc, &alloc, Some(&ca)).unwrap();
    let idx = cmd
        .args
        .iter()
        .position(|a| a == "--n-cpu-moe")
        .expect("--n-cpu-moe emitted");
    assert_eq!(cmd.args[idx + 1], "25");
    // The coarse path pins nothing per-tensor and lets the runtime split.
    assert!(!cmd.args.iter().any(|a| a == "-ot"));
    assert!(!cmd.args.iter().any(|a| a == "--tensor-split"));
}

#[test]
fn placement_cmd_args_emit_tensor_split_mode_flags() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let ca = CommandArgs {
        ngl: Some(999),
        tensor_split: Some(vec![1, 1]),
        override_tensor: vec![],
        split_mode: Some(SplitMode::Tensor),
        main_gpu: Some(0),
        n_cpu_moe: None,
    };
    let cmd = render_argv(&svc, &alloc, Some(&ca)).unwrap();
    let sm = cmd.args.iter().position(|a| a == "--split-mode").unwrap();
    assert_eq!(cmd.args[sm + 1], "tensor");
    let mg = cmd.args.iter().position(|a| a == "--main-gpu").unwrap();
    assert_eq!(cmd.args[mg + 1], "0");
    let ts = cmd.args.iter().position(|a| a == "--tensor-split").unwrap();
    assert_eq!(cmd.args[ts + 1], "1,1");
}

#[test]
fn placement_cmd_args_emit_weighted_tensor_split() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let ca = CommandArgs {
        ngl: Some(999),
        tensor_split: Some(vec![13, 5]),
        override_tensor: vec![],
        split_mode: Some(SplitMode::Tensor),
        main_gpu: Some(0),
        n_cpu_moe: None,
    };
    let cmd = render_argv(&svc, &alloc, Some(&ca)).unwrap();
    let ts = cmd.args.iter().position(|a| a == "--tensor-split").unwrap();
    assert_eq!(cmd.args[ts + 1], "13,5");
    let sm = cmd.args.iter().position(|a| a == "--split-mode").unwrap();
    assert_eq!(cmd.args[sm + 1], "tensor");
}

/// Regression for the scenario-01 `CUDA_VISIBLE_DEVICES=` empty-env bug:
/// `SupervisorInit::allocation` is built from `placement_override` when the
/// registry is constructed. For any estimator-driven service (no override),
/// that bundle is empty, so rendering `render_argv` against it would emit
/// `CUDA_VISIBLE_DEVICES=` and the child would silently fall back to CPU.
/// The supervisor must thread the *packed* allocation into `render_argv`.
/// This test demonstrates the discriminator: the two allocations produce
/// different env values, so a regression that swaps back to `init.allocation`
/// would be caught by the supervisor-level smoke path.
#[test]
fn render_uses_supplied_allocation_for_cuda_env() {
    let svc = base_service();
    // Empty override (estimator-driven) → `init.allocation` is empty.
    let empty_alloc = Allocation::from_override(&BTreeMap::new());
    let empty_cmd = render_argv(&svc, &empty_alloc, None).unwrap();
    assert_eq!(
        empty_cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(),
        "",
        "init.allocation with no GPU entries must render as empty (CPU fallback)"
    );

    // A packed allocation that placed layers on GPU 1 → env should list it.
    let mut placed = BTreeMap::new();
    placed.insert(DeviceSlot::Gpu(1), 4096);
    let packed_alloc = Allocation::from_override(&placed);
    let packed_cmd = render_argv(&svc, &packed_alloc, None).unwrap();
    assert_eq!(packed_cmd.env.get("CUDA_VISIBLE_DEVICES").unwrap(), "1");
}

#[test]
fn custom_binary_replaces_llama_server() {
    let mut svc = base_service();
    expect_llama_cpp(&mut svc).binary = PathBuf::from("/opt/bin/special-llama-server");
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert_eq!(cmd.binary, "/opt/bin/special-llama-server");
    // `-m <path>` still leads the argv, same shape as the default.
    assert_eq!(cmd.args[0], "-m");
    assert_eq!(cmd.args[1], "/m/x.gguf");
}

#[test]
fn launcher_template_splats_args_and_substitutes_model() {
    let mut svc = base_service();
    {
        let lc = expect_llama_cpp(&mut svc);
        lc.launcher = Some(vec![
            "/opt/podman-wrap.sh".into(),
            "{model}".into(),
            "{args}".into(),
        ]);
    }
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert_eq!(cmd.binary, "/opt/podman-wrap.sh");
    // Model is positional; standard `-m <path>` is *not* prepended.
    assert_eq!(cmd.args[0], "/m/x.gguf");
    assert!(
        !cmd.args.iter().any(|a| a == "-m"),
        "`-m` must not leak into the launcher argv: {:?}",
        cmd.args
    );
    // Splat covers the rest of llama-server's flags.
    assert!(cmd.args.iter().any(|a| a == "-c"));
    assert!(cmd.args.iter().any(|a| a == "8192"));
    assert!(cmd.args.iter().any(|a| a == "--port"));
    assert!(cmd.args.iter().any(|a| a == "41000"));
}

#[test]
fn launcher_splat_inside_arg_is_rejected() {
    let mut svc = base_service();
    expect_llama_cpp(&mut svc).launcher = Some(vec!["wrap.sh".into(), "--foo={args}".into()]);
    let alloc = Allocation::from_override(&svc.placement_override);
    let err = match render_argv(&svc, &alloc, None) {
        Ok(_) => panic!("expected splat-misuse error"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("{args}"),
        "expected splat-misuse error, got {err}"
    );
}

// ----- env_inherit tests (base_service-dependent; the SpawnConfig-only
// resolve_env tests live in `spawn::tests`, which needs no service fixture) -----

#[test]
fn default_service_has_env_inherit_true() {
    let svc = base_service();
    assert!(svc.env_inherit);
}

#[test]
fn render_argv_default_produces_env_inherit_true() {
    let svc = base_service();
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(cmd.env_inherit);
}

#[test]
fn render_argv_env_inherit_false_propagates() {
    let mut svc = base_service();
    svc.env_inherit = false;
    let alloc = Allocation::from_override(&svc.placement_override);
    let cmd = render_argv(&svc, &alloc, None).unwrap();
    assert!(!cmd.env_inherit);
}

/// An operator who already passes `--cache-ram` through `extra_args` must
/// not get a second, conflicting `-cram` from the daemon — and the
/// estimate has to read their value so the reservation matches what the
/// runtime is actually capped at. Production configs predate the
/// dedicated key and still set it this way.
#[test]
fn an_operator_supplied_cache_ram_is_not_duplicated() {
    let mut svc = base_service();
    svc.extra_args = vec!["--cache-ram".into(), "0".into()];
    let alloc = Allocation::from_override(&svc.placement_override);
    let args = render_argv(&svc, &alloc, None).unwrap().args;
    assert_eq!(
        args.iter()
            .filter(|a| *a == "-cram" || *a == "--cache-ram")
            .count(),
        1,
        "exactly one cache-ram flag should reach the child: {args:?}"
    );
    assert_eq!(
        crate::config::service_inputs::cache_ram_from_extra_args(&svc.extra_args),
        Some(0),
        "the estimator must read the operator's value"
    );
}
