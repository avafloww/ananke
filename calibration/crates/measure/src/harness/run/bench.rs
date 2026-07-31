//! The coding-agent prompts a growth run is driven with.
//!
//! Realism is the part that cannot be synthesised: prompt-cache behaviour and
//! generation both depend on what the tokens actually are, and filler that a
//! cache or a drafter finds unnaturally easy measures the wrong thing. Vendored
//! from the `llama-cpp-model-tuning` skill so a calibration run needs nothing
//! outside this repository; keep them in sync when either changes.
//!
//! Only the prompts are here. The harness drives the conversation itself, so a
//! throughput benchmark wrapped around them would go uncalled.

pub(crate) const SYSTEM: &str = "You are an expert software engineer. You have access to the following tools:

- read_file(path: str) -> str: Read the contents of a file.
- write_file(path: str, content: str) -> None: Write content to a file.
- run_command(cmd: str) -> str: Run a shell command and return stdout.
- search(query: str) -> str: Search the codebase for a pattern.

When you need to use a tool, format your response as a JSON object with \"tool\" and \"args\" keys.
Always explain your reasoning before taking action. Consider edge cases, error handling, and performance implications.";

pub(crate) const PROMPTS: &[&str] = &[
    "Write a Rust function that takes a Vec<PathBuf> and returns a HashMap<String, Vec<PathBuf>> grouping files by their extension. Handle edge cases like files with no extension, hidden files, and non-UTF8 paths. Use thiserror for error types.",
    "Refactor this Python class to use async/await instead of threading. The class manages a pool of worker connections and needs to handle timeouts gracefully: class WorkerPool: def __init__(self, size): self.workers = [Worker() for _ in range(size)] def submit(self, task): # ...",
    "Implement a TypeScript debounce function that supports cancellation and immediate execution. It should be generic over the function signature and properly handle 'this' binding. Include JSDoc comments.",
    "Write a SQL query to find the top 10 customers by total revenue in the last 30 days, including their email and last purchase date. Handle customers with no purchases. The schema has tables: customers(id, email, name), orders(id, customer_id, total, created_at).",
    "Debug this Go code — it deadlocks occasionally. The worker pool processes jobs from a channel but sometimes hangs on shutdown: func process(jobs <-chan Job) { for j := range jobs { handle(j) } }",
    "Write a Nix module that defines a systemd service running a Python script. The service should have a configurable package, environment variables, and a health check. Include an option for the listen port.",
    "Implement a C function that parses a simple HTTP request line (GET /path HTTP/1.1) without using strtok. Handle edge cases: leading whitespace, extra spaces, missing version. Return a struct with method, path, and version.",
    "Write a Dockerfile for a multi-stage build of a Rust application. The first stage builds with cargo, the second creates a minimal runtime image with only the binary and necessary certs. Use distroless as the final base.",
];
