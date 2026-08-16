//! Render llama-server argv: the model path, the launcher-template split
//! (which holds `-m` back so `{model}` can be positioned freely), and every
//! flag `render_llama_server_flags` derives from the service config and the
//! placement-derived `CommandArgs`.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use ananke_allocator::placement::CommandArgs;
use ananke_api::shared::modality::Modality;
use ananke_config::validate::{LlamaCppConfig, PlacementPolicy, ServiceConfig};
use ananke_devices::{Allocation, cuda_env};
use ananke_estimate::service_inputs::cache_ram_from_extra_args;
use ananke_templates::{PlaceholderContext, substitute_launcher_argv};

use crate::spawn::SpawnConfig;

pub(super) fn render_llama_cpp_argv(
    svc: &ServiceConfig,
    lc: &LlamaCppConfig,
    alloc: &Allocation,
    cmd_args: Option<&CommandArgs>,
) -> Result<SpawnConfig, ananke_templates::SubstituteError> {
    let mut env: BTreeMap<String, String> = svc.env.clone();
    env.insert("CUDA_VISIBLE_DEVICES".into(), cuda_env::render(alloc));

    let standard_args = render_llama_server_flags(svc, lc, cmd_args);

    if let Some(launcher) = &lc.launcher {
        // Launcher template: `{model}` is exposed as a standalone
        // placeholder so wrappers can position it (e.g. for a container
        // volume mount). Every other flag — `--mmproj`, `-c`, port,
        // placement-derived `-ngl`/`--tensor-split`/`-ot`, sampling,
        // `extra_args` — flows through the `{args}` splat. The launcher
        // owns its own argv shape from there.
        let model_str = lc.model.to_string_lossy().into_owned();
        let ctx = PlaceholderContext {
            name: &svc.name,
            port: svc.private_port,
            model: Some(&model_str),
            allocation: alloc,
            static_reserve_mb: None,
            listen_host: None,
            host_port: svc.private_port,
        };
        let argv = substitute_launcher_argv(launcher, &standard_args, &ctx)?;
        // `launcher` is non-empty (the validator rejects an empty
        // launcher), and substitution never drops the first entry, so
        // argv has at least one element here.
        let mut iter = argv.into_iter();
        let binary = iter.next().unwrap_or_default();
        let args: Vec<String> = iter.collect();
        return Ok(SpawnConfig {
            binary,
            args,
            env,
            env_inherit: svc.env_inherit,
        });
    }

    let mut args: Vec<String> = Vec::new();
    args.push("-m".into());
    args.push(lc.model.to_string_lossy().into_owned());
    args.extend(standard_args);

    Ok(SpawnConfig {
        binary: lc.binary.to_string_lossy().into_owned(),
        args,
        env,
        env_inherit: svc.env_inherit,
    })
}

/// Render every llama-server flag *except* `-m <model>`. The model
/// path is held back so the `launcher` template's `{model}` placeholder
/// can position it freely (e.g. for a container volume mount); every
/// other flag — including `--mmproj <path>`, placement-derived
/// `-ngl`/`--tensor-split`/`-ot`, sampling, host/port, and `extra_args`
/// — is emitted here and reaches the launcher via the `{args}` splat.
/// Shared by the default and launcher rendering paths so both emit
/// identical flag sets.
///
/// Emit all `--override-tensor` rules as a *single* comma-joined flag
/// rather than one `-ot` per rule. Current llama.cpp deprecated repeated
/// `-ot` flags and now honours **only the last one** ("argument '-ot'
/// specified multiple times … only last value will be used"), so a
/// per-rule emission silently drops every rule but the last — which for
/// the packer's synthesised MoE offload leaves almost no experts moved off
/// the GPU, OOMing at load. Rules never contain a comma, so the
/// join is lossless and llama.cpp parses the combined value itself.
fn push_override_tensor(args: &mut Vec<String>, rules: &[String]) {
    if !rules.is_empty() {
        args.push("-ot".into());
        args.push(rules.join(","));
    }
}

fn render_llama_server_flags(
    svc: &ServiceConfig,
    lc: &LlamaCppConfig,
    cmd_args: Option<&CommandArgs>,
) -> Vec<String> {
    // Only a real packer layout overrides the service's own offload settings.
    // See [`CommandArgs::describes_offload`].
    let cmd_args = cmd_args.filter(|ca| ca.describes_offload());
    let mut args: Vec<String> = Vec::new();

    if let Some(ik) = lc.runtime.ik() {
        if let Some(m) = ik.mla {
            args.push("-mla".into());
            args.push(m.to_string());
        }
        if ik.dsa {
            args.push("-dsa".into());
            args.push("-fidx".into());
        }
        if let Some(amb) = ik.attn_max_batch {
            args.push("-amb".into());
            args.push(amb.to_string());
        }
        if ik.runtime_repack {
            args.push("-rtr".into());
        }
    }

    if let Some(mmproj) = &lc.mmproj {
        args.push("--mmproj".into());
        args.push(mmproj.to_string_lossy().into_owned());
    }
    if let Some(ctx) = lc.context {
        args.push("-c".into());
        args.push(ctx.to_string());
    }

    if let Some(ca) = cmd_args {
        // Placement engine provided -ngl; ignore the static config path.
        if let Some(ngl) = ca.ngl {
            args.push("-ngl".into());
            args.push(ngl.to_string());
        }
    } else {
        match svc.placement_policy {
            PlacementPolicy::CpuOnly => {
                args.push("-ngl".into());
                args.push("0".into());
            }
            PlacementPolicy::GpuOnly | PlacementPolicy::Hybrid => {
                if let Some(ngl) = lc.n_gpu_layers {
                    args.push("-ngl".into());
                    args.push(ngl.to_string());
                } else {
                    args.push("-ngl".into());
                    args.push("999".into());
                }
            }
        }
    }

    if lc.flash_attn == Some(true) {
        args.push("-fa".into());
        args.push("on".into());
    }
    if let Some(k) = &lc.cache_type_k {
        args.push("--cache-type-k".into());
        args.push(k.to_string());
    }
    if let Some(v) = &lc.cache_type_v {
        args.push("--cache-type-v".into());
        args.push(v.to_string());
    }
    if lc.jinja.unwrap_or(false) {
        args.push("--jinja".into());
    }
    if let Some(p) = &lc.chat_template_file {
        args.push("--chat-template-file".into());
        args.push(p.to_string_lossy().into_owned());
    }
    if let Some(t) = lc.threads {
        args.push("--threads".into());
        args.push(t.to_string());
    }
    if let Some(t) = lc.threads_batch {
        args.push("--threads-batch".into());
        args.push(t.to_string());
    }
    if let Some(numa) = lc.numa {
        args.push("--numa".into());
        args.push(numa.as_flag().into());
    }
    if let Some(b) = lc.batch_size {
        args.push("-b".into());
        args.push(b.to_string());
    }
    if let Some(b) = lc.ubatch_size {
        args.push("-ub".into());
        args.push(b.to_string());
    }
    if lc.mmap == Some(false) {
        args.push("--no-mmap".into());
    }
    if lc.mlock == Some(true) {
        args.push("--mlock".into());
    }
    if let Some(p) = lc.parallel {
        args.push("-np".into());
        args.push(p.to_string());
    }
    if lc.kv_unified == Some(true) {
        args.push("--kv-unified".into());
    }
    if lc.cache_idle_slots == Some(false) {
        args.push("--no-cache-idle-slots".into());
    }
    // Emitted even when it matches llama.cpp's own default: the prompt cache
    // is host RAM the packer reserves, and the reservation is only honest if
    // the runtime is capped at the number that was reserved. Skipped when the
    // operator already passes the flag through `extra_args` — duplicating it
    // would leave two conflicting values on the command line, and the
    // estimator reads theirs for the reservation.
    if cache_ram_from_extra_args(&svc.extra_args).is_none() {
        args.push("-cram".into());
        args.push(
            lc.cache_ram_mb
                .unwrap_or(ananke_estimate::host_buffer::DEFAULT_CACHE_RAM_MB)
                .to_string(),
        );
    }
    // An embedding service needs llama-server's embeddings endpoint enabled;
    // the pooling strategy comes from the GGUF's `{arch}.pooling_type`, so
    // the flag is all the modality implies.
    if svc.modality == Modality::Embedding {
        args.push("--embeddings".into());
    }
    // `--metrics` is also auto-injected when the generation-stall watchdog is
    // active, which polls the endpoint for progress counters. An explicit
    // `metrics = false` wins (and disables that watchdog at runtime); the
    // validator defaults `generation_stall` per template, so this branch only
    // fires for llama-cpp services.
    let genstall_needs_metrics =
        svc.auto_restart.generation_stall.is_some() && lc.metrics != Some(false);
    if lc.metrics == Some(true) || genstall_needs_metrics {
        args.push("--metrics".into());
    }
    if lc.slots == Some(true) {
        args.push("--slots".into());
    }
    if let Some(st) = &lc.spec_type {
        args.push("--spec-type".into());
        args.push(st.to_string());
    }
    if let Some(n) = lc.spec_draft_n_max {
        args.push("--spec-draft-n-max".into());
        args.push(n.to_string());
    }
    if let Some(md) = &lc.draft_model {
        args.push("-md".into());
        args.push(md.to_string_lossy().into_owned());
    }

    if let Some(ca) = cmd_args {
        // Placement-derived tensor-split and override-tensor rules take
        // precedence; lc.override_tensor is subsumed into CommandArgs by
        // the placement engine already.
        //
        // `--split-mode`/`--main-gpu` are emitted only for sharded
        // (tensor/row) packings; layer split leaves `split_mode` `None` so
        // llama.cpp keeps its default `layer` mode and the argv is unchanged.
        if let Some(mode) = ca.split_mode {
            args.push("--split-mode".into());
            args.push(mode.as_flag().into());
        }
        if let Some(mg) = ca.main_gpu {
            args.push("--main-gpu".into());
            args.push(mg.to_string());
        }
        if let Some(ref split) = ca.tensor_split {
            let split_str = split
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            args.push("--tensor-split".into());
            args.push(split_str);
        }
        // Coarse whole-layer expert offload. Both mainline and ik_llama accept
        // `--n-cpu-moe`; emitted instead of per-tensor `-ot` so the runtime's
        // fused CPU MoE kernel stays engaged. `-ngl 999` (from the packer) puts
        // every layer on GPU first, then this pulls the trailing N back to CPU.
        if let Some(n) = ca.n_cpu_moe {
            args.push("--n-cpu-moe".into());
            args.push(n.to_string());
        }
        push_override_tensor(&mut args, &ca.override_tensor);
    } else {
        push_override_tensor(&mut args, &lc.override_tensor);
    }

    // Sampling params are passed as extra flags when set.
    let s = &lc.sampling;
    if let Some(t) = s.temperature {
        args.push("--temp".into());
        args.push(t.to_string());
    }
    if let Some(p) = s.top_p {
        args.push("--top-p".into());
        args.push(p.to_string());
    }
    if let Some(k) = s.top_k {
        args.push("--top-k".into());
        args.push(k.to_string());
    }
    if let Some(m) = s.min_p {
        args.push("--min-p".into());
        args.push(m.to_string());
    }
    if let Some(r) = s.repeat_penalty {
        args.push("--repeat-penalty".into());
        args.push(r.to_string());
    }
    args.extend(svc.extra_args.iter().cloned());
    args.push("--host".into());
    args.push("127.0.0.1".into());
    args.push("--port".into());
    args.push(svc.private_port.to_string());

    args
}
