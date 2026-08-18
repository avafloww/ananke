// Regenerating types makes a new backend field typed; it does not make it
// visible. These check that a containerized service's identity actually
// reaches the config grid, and that a native service's grid is unchanged by
// the field's existence.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ServiceDetail } from "../../api/client.ts";
import { ConfigGrid } from "./ServiceInfoCards.tsx";

/** A minimal detail payload; individual tests layer on what they exercise. */
function detail(overrides: Partial<ServiceDetail> = {}): ServiceDetail {
  return {
    name: "muse-glimmer",
    template: "llama-cpp",
    state: "running",
    lifecycle: "on_demand",
    idle_timeout_ms: 600_000,
    private_port: 48202,
    ...overrides,
  } as ServiceDetail;
}

describe("config_grid_renders_container_identity", () => {
  it("shows runtime, image, and network for a container service", () => {
    render(
      <ConfigGrid
        detail={detail({
          container: {
            runtime: "podman",
            image: "ghcr.io/ggml-org/llama.cpp:server-cuda",
            network: "host",
          },
        })}
      />,
    );

    expect(
      screen.getByText("podman · ghcr.io/ggml-org/llama.cpp:server-cuda"),
    ).toBeInTheDocument();
    expect(screen.getByText("Container network")).toBeInTheDocument();
    expect(screen.getByText("host")).toBeInTheDocument();
  });

  it("shows the live identity only while a container is running", () => {
    const running = {
      runtime: "docker",
      image: "ninfer:local",
      network: "host",
      container_id: "abcdef0123456789abcdef",
      container_name: "ananke-ninfer-42",
    };
    render(<ConfigGrid detail={detail({ container: running })} />);

    // The ID is abbreviated the way both runtimes display it.
    expect(screen.getByText("abcdef012345")).toBeInTheDocument();
    expect(screen.getByText("ananke-ninfer-42")).toBeInTheDocument();
  });

  it("omits the live identity for a configured but idle container", () => {
    render(
      <ConfigGrid
        detail={detail({
          container: {
            runtime: "docker",
            image: "ninfer:local",
            network: "bridge",
          },
        })}
      />,
    );

    expect(screen.getByText("docker · ninfer:local")).toBeInTheDocument();
    expect(screen.queryByText("Container ID")).not.toBeInTheDocument();
    expect(screen.queryByText("Container name")).not.toBeInTheDocument();
  });

  it("leaves a native service's grid untouched", () => {
    render(<ConfigGrid detail={detail()} />);
    expect(screen.getByText("llama-cpp")).toBeInTheDocument();
    expect(screen.queryByText("Container runtime")).not.toBeInTheDocument();
    expect(screen.queryByText("Container network")).not.toBeInTheDocument();
  });

  it("containerized_llama_detail_retains_serving_cards", () => {
    // A containerized llama.cpp service still is a llama.cpp service: the
    // container block is additional, never a replacement for the engine's
    // own configuration.
    render(
      <ConfigGrid
        detail={detail({
          template: "llama-cpp",
          container: {
            runtime: "docker",
            image: "ghcr.io/ggml-org/llama.cpp:server-cuda",
            network: "host",
          },
        })}
      />,
    );
    expect(screen.getByText("llama-cpp")).toBeInTheDocument();
    expect(screen.getByText("Context")).toBeInTheDocument();
    expect(screen.getByText(":48202")).toBeInTheDocument();
  });
});
