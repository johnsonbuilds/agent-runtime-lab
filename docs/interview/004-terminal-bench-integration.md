# Terminal-Bench Integration — Interview Notes

## 1. Why did you introduce a ShellExecutor abstraction?

I wanted to separate the agent runtime from the execution environment.

The same agent runtime should be able to execute shell commands locally or inside an isolated environment such as Harbor without changing the agent loop.

The runtime depends on a ShellExecutor interface, while different implementations provide different execution environments.

For example:

- LocalShellExecutor executes commands locally.
- HarborShellExecutor executes commands inside the Harbor environment.

This keeps the core agent runtime independent of Harbor.

---

## 2. Why shouldn't the Agent Loop know about Harbor?

Harbor is an execution environment and evaluation harness, not part of the agent's reasoning logic.

The Agent Loop should only care about:

- sending messages to the LLM
- handling tool calls
- executing tools
- processing observations
- deciding when the run is complete

If Harbor-specific logic is added to the Agent Loop, the runtime becomes tightly coupled to one benchmark environment.

By isolating Harbor at the execution boundary, the same runtime can work with different environments.

---

## 3. Why did you refactor the runtime to be async-first?

When integrating with Harbor, I found that the environment execution API was asynchronous while my original runtime execution interfaces were synchronous.

One option was to introduce an async-to-sync bridge in the Harbor adapter.

Instead, I decided to make the runtime's I/O execution path async-first.

The execution path is now conceptually:

    Agent Loop
        ↓
    async LLM call
        ↓
    async Tool execution
        ↓
    async ShellExecutor
        ↓
    Local or Harbor environment

This avoids repeatedly switching between synchronous and asynchronous execution models and makes the runtime easier to extend to other I/O-bound tools and environments.

I did not make everything asynchronous just for the sake of it. The async boundary is mainly around I/O operations such as LLM calls and tool/environment execution.

---

## 4. How did you integrate your agent runtime with Terminal-Bench?

I use Harbor as the evaluation harness and implemented a thin Harbor `BaseAgent` adapter.

The adapter receives:

- the Terminal-Bench task instruction
- the Harbor execution environment
- the Harbor AgentContext

It then creates the runtime's tool registry with a Harbor-backed ShellExecutor and delegates execution to the existing `run_turn()` function.

The architecture is:

    Terminal-Bench
          ↓
        Harbor
          ↓
      HarborAgent
          ↓
    sunagent
          ↓
       Agent Loop
          ↓
      ToolRegistry
          ↓
    HarborShellExecutor
          ↓
      Harbor sandbox

I intentionally did not duplicate the Agent Loop inside the Harbor adapter.

---

## 5. What is the responsibility of the Harbor adapter?

The Harbor adapter is intentionally thin.

Its responsibilities are:

1. Receive the benchmark task instruction.
2. Receive the Harbor execution environment.
3. Connect the environment to the runtime's ShellExecutor.
4. Invoke the existing Agent Loop.
5. Preserve the runtime's tracing.
6. Record useful execution metadata in Harbor's AgentContext.

The adapter should not contain the core agent reasoning or tool-calling logic.

---

## 6. How did you preserve observability during Terminal-Bench execution?

The runtime already has JSONL tracing.

The Harbor adapter keeps that tracing and also incrementally updates `AgentContext.metadata`.

The runtime metadata includes information such as:

- task instruction
- execution status
- event count
- last event
- final answer
- error information
- trace path

The important design choice is that the context is updated incrementally as trace events are generated, rather than only after the run finishes.

This means useful execution information can still be preserved if the run fails or times out.

---

## 7. How did you validate the Harbor integration?

I used several levels of validation.

First, I verified the Harbor and Terminal-Bench environment independently using the Oracle agent.

A single Terminal-Bench task completed successfully with:

    Exceptions: 0
    Mean: 1.000
    Reward: 1.0

Then I added a HarborAgent smoke test using a fake LLM and fake Harbor environment.

The test verifies that:

    LLM tool call
        ↓
    Harbor environment
        ↓
    tool result
        ↓
    next LLM turn

works through the existing runtime.

After the implementation:

- 21 tests passed
- wheel build succeeded
- compile check passed
- git diff --check passed

---

## 8. What happened when you ran a real Terminal-Bench task with your runtime?

The first real run successfully reached the real LLM and executed tool calls through the Harbor environment.

The execution itself did not crash.

However, the agent did not complete the task within the time I allowed, so I manually terminated the run before the Terminal-Bench verifier produced a result.

Therefore, I don't consider this a successful benchmark result yet.

The current status is:

    Harbor integration       PASS
    Real LLM execution       PASS
    Real tool execution      PASS
    Task completion          NOT YET VERIFIED
    Verifier result          NOT AVAILABLE

The next step is to analyze the trajectory and determine why the agent did not terminate.

---

## 9. What would you investigate when an agent does not terminate?

I would first analyze the trajectory instead of immediately adding more features.

I would look for several possible failure modes:

1. Repeated tool calls
2. Repeated commands or states
3. The agent continuing after the task was effectively completed
4. Inefficient exploration
5. Context growth
6. Tool execution latency
7. LLM reasoning failure
8. Missing or unclear termination criteria

I would first classify the failure and then make the smallest runtime-level change supported by the evidence.

The intended development loop is:

    Benchmark
        ↓
    Trajectory
        ↓
    Failure analysis
        ↓
    Hypothesis
        ↓
    Minimal runtime change
        ↓
    Benchmark again

---

## 10. Why use a real benchmark instead of only unit tests?

Unit tests can verify whether individual components behave as expected, but they cannot tell us whether the complete agent can solve a real multi-step task.

Terminal-Bench provides:

- real task instructions
- a real execution environment
- real tool interaction
- task-specific verification
- an objective reward

This allows me to evaluate the runtime as a complete system rather than only testing individual components.

---

## 11. How do you use coding agents in your development workflow?

I use coding agents such as OpenCode for implementation, but I keep the engineering decisions under my control.

My workflow is:

    Identify a problem
        ↓
    Understand the technical issue
        ↓
    Design the architecture
        ↓
    Give the coding agent a precise task
        ↓
    Review the generated code
        ↓
    Run tests
        ↓
    Run the benchmark
        ↓
    Analyze failures
        ↓
    Decide the next change

I don't think manually typing every line is the important part of modern AI engineering.

The important part is being able to make good architectural decisions, review the implementation, verify the behavior, and understand why the system works or fails.

---

# Key Interview Expressions

## Explaining architecture

- "I wanted to separate X from Y."
- "The core runtime should remain independent of..."
- "I isolated the dependency at the execution boundary."
- "The adapter is intentionally thin."
- "I reused the existing Agent Loop instead of duplicating it."

## Explaining a technical decision

- "When I integrated X, I found that..."
- "One option was to..., but I decided to..."
- "The main reason was to avoid..."
- "This makes the runtime easier to extend to..."
- "I wanted to minimize coupling between..."

## Explaining a failure

- "The integration itself worked, but..."
- "The run did not complete within the allowed time."
- "I terminated the run before verification."
- "I don't consider this a successful benchmark result yet."
- "The next step is to analyze the trajectory and identify the failure mode."

## Explaining engineering methodology

- "I don't want to add features based on speculation."
- "I want the benchmark results to drive the next runtime improvement."
- "First, I would analyze the failure before changing the implementation."
- "I would make the smallest change supported by the evidence."