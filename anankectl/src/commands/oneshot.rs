//! Oneshot job command handlers (submit, run, list, kill).

use std::path::Path;

use ananke_api::oneshot::{
    create::{OneshotAllocation, OneshotDevices, OneshotRequest, OneshotResponse},
    get::OneshotStatus,
};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

/// Submit a oneshot job described by a TOML file.
pub async fn submit(client: &ApiClient, json: bool, path: &Path) -> Result<(), ApiClientError> {
    let toml_str = std::fs::read_to_string(path)
        .map_err(|e| ApiClientError::Usage(format!("read {}: {e}", path.display())))?;
    let req: OneshotRequest =
        toml::from_str(&toml_str).map_err(|e| ApiClientError::Usage(format!("parse TOML: {e}")))?;
    let resp: OneshotResponse = client.post_json("/api/oneshot", &req).await?;
    if json {
        output::print_json(&resp);
    } else {
        println!(
            "ok: {} (port {}, logs {})",
            resp.id, resp.port, resp.logs_url
        );
    }
    Ok(())
}

/// Submit a oneshot job built from inline CLI flags and a trailing command.
// The signature mirrors the clap CLI surface, so the argument count is owned
// by the flag definitions rather than this function; grouping them into a
// struct would fight the derive.
#[expect(clippy::too_many_arguments)]
pub async fn run(
    client: &ApiClient,
    json: bool,
    name: Option<String>,
    priority: u8,
    ttl: Option<String>,
    workdir: Option<std::path::PathBuf>,
    placement: String,
    reserve_gb: Option<f32>,
    min_reserve_gb: Option<f32>,
    max_reserve_gb: Option<f32>,
    command: Vec<String>,
) -> Result<(), ApiClientError> {
    let allocation = match (reserve_gb, min_reserve_gb, max_reserve_gb) {
        (Some(g), None, None) => OneshotAllocation {
            mode: Some("static".into()),
            reserve_gb: Some(g),
            min_reserve_gb: None,
            max_reserve_gb: None,
        },
        (None, Some(lo), Some(hi)) => OneshotAllocation {
            mode: Some("dynamic".into()),
            reserve_gb: None,
            min_reserve_gb: Some(lo),
            max_reserve_gb: Some(hi),
        },
        (None, None, None) => {
            return Err(ApiClientError::Usage(
                "must specify --reserve-gb or --min-reserve-gb + --max-reserve-gb".into(),
            ));
        }
        _ => {
            return Err(ApiClientError::Usage(
                "conflicting reservation flags".into(),
            ));
        }
    };

    let req = OneshotRequest {
        name,
        template: "command".into(),
        command: Some(command),
        workdir: workdir.map(|p| p.to_string_lossy().into_owned()),
        allocation,
        devices: Some(OneshotDevices {
            placement: Some(placement),
        }),
        priority: Some(priority),
        ttl,
        port: None,
        health: None,
        metadata: Default::default(),
    };
    let resp: OneshotResponse = client.post_json("/api/oneshot", &req).await?;
    if json {
        output::print_json(&resp);
    } else {
        println!(
            "ok: {} (port {}, logs {})",
            resp.id, resp.port, resp.logs_url
        );
    }
    Ok(())
}

/// List all known oneshot jobs in a table.
pub async fn list(client: &ApiClient, json: bool) -> Result<(), ApiClientError> {
    let rows: Vec<OneshotStatus> = client.get_json("/api/oneshot").await?;
    if json {
        output::print_json(&rows);
    } else {
        println!("{:<30} {:<12} {:<10} {:>6}", "ID", "NAME", "STATE", "PORT");
        for r in &rows {
            println!("{:<30} {:<12} {:<10} {:>6}", r.id, r.name, r.state, r.port);
        }
    }
    Ok(())
}

/// Cancel a running oneshot job by ID.
pub async fn kill(client: &ApiClient, _json: bool, id: &str) -> Result<(), ApiClientError> {
    client.delete(&format!("/api/oneshot/{id}")).await?;
    println!("ok: killed {id}");
    Ok(())
}
