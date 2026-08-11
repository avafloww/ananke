#![cfg(feature = "test-fakes")]

mod common;

use std::path::{Path, PathBuf};

use ananke::config::TemplateConfig;
use ananke_estimate as estimator;
use common::synth_gguf;

#[test]
fn unreadable_separate_draft_does_not_use_target_embedded_head() {
    let model_path = Path::new("/fake/mtp-target.gguf");
    let draft_path = Path::new("/fake/missing-draft.gguf");
    let fs = synth_gguf::Builder::new()
        .kv_string("general.architecture", "qwen3")
        .kv_u32("qwen3.block_count", 2)
        .kv_u32("qwen3.context_length", 262144)
        .kv_u32("qwen3.nextn_predict_layers", 1)
        .kv_u32("qwen3.attention.head_count_kv", 4)
        .kv_u32("qwen3.attention.key_length", 128)
        .kv_u32("qwen3.attention.value_length", 128)
        .tensor_f16("blk.0.attn_q.weight", 512 * 1024)
        .tensor_f16("blk.1.attn_q.weight", 512 * 1024)
        .tensor_f16("output.weight", 2 * 512 * 1024)
        .tensor_f16("token_embd.weight", 4 * 512 * 1024)
        .into_in_memory_fs(model_path);

    let mut service = common::minimal_llama_service("mtp", 0);
    common::set_model_path(&mut service, model_path);
    let TemplateConfig::LlamaCpp(llama) = &mut service.template_config else {
        unreachable!();
    };
    llama.context = Some(262144);
    llama.spec_type = Some("draft-mtp".into());
    llama.draft_model = Some(PathBuf::from(draft_path));

    let inputs = ananke::config::estimator_inputs(&service).unwrap();
    let (_summary, estimate) = estimator::estimate_with_summary(&fs, &inputs).unwrap();

    assert_eq!(estimate.mtp.bytes, 0);
    assert_eq!(estimate.mtp.weight_bytes, 0);
}
