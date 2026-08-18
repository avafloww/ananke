// Baseline smoke test: proves the Vitest + React Testing Library harness
// runs cleanly against existing components before any container-related
// changes. Kept deliberately small so it is the gate for Phase 4b.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Badge } from "./Badge";

describe("Badge", () => {
  it("renders its children", () => {
    render(<Badge>healthy</Badge>);
    expect(screen.getByText("healthy")).toBeInTheDocument();
  });

  it("applies the semantic variant class", () => {
    const { container } = render(<Badge variant="success">ok</Badge>);
    expect(container.firstElementChild).toHaveClass("bg-success/12");
  });
});
