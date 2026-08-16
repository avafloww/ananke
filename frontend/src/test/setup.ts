// Vitest jsdom setup: load jest-dom matchers so component tests can use
// user-visible queries (`toBeInTheDocument`, `toHaveTextContent`, etc).
import "@testing-library/jest-dom/vitest";

// Component tests render real components, which call `t(...)`; initialising
// i18n here means a missing key shows up as a raw key in an assertion
// rather than silently rendering nothing.
import "../i18n.ts";

// Vitest runs without `globals`, so React Testing Library's automatic
// cleanup never registers. Without this, one test's rendered DOM stays in
// the document and the next test's `queryBy*` assertions see it.
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);
