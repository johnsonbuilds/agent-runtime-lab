# Agent Loop — Interview Q&A

> This document contains interview questions and refined answers for understanding and explaining an Agent Loop from both an implementation and production-engineering perspective.

---

## 1. What is an agent loop?

### Short Answer

> **A minimal agent can be implemented as an iterative loop. First, we initialize the conversation with the user's message. Inside the loop, the agent sends the current conversation and the available tool schemas to the LLM. The LLM decides whether it needs to call a tool. If the response contains a tool call, the agent parses the arguments, executes the tool, and appends the tool result back to the conversation as an observation. The loop continues until the agent produces a final answer or reaches the maximum iteration limit.**

### Production Extension

> **The basic loop is relatively simple. The hard part is making it reliable in production. In a production system, we need to handle things like tool failures, timeouts, retries, context growth, observability, state recovery, and termination control.**

### Core Flow

```text
User Task
    ↓
Context Build
    ↓
LLM
    ↓
Tool Call?
   / \
 Yes  No
  ↓    ↓
Parse  Final Answer
  ↓
Dispatch
  ↓
Execute Tool
  ↓
Tool Result
  ↓
Observation
  ↓
State / Context Update
  ↓
Next LLM Call
```

### Important Detail

For a minimal implementation, a response without a tool call can be treated as the final answer.

For a production system, however, we may also need:

* output validation
* termination checks
* policy checks
* structured-output validation

---

# 2. What happens after the LLM generates a tool call?

### Answer

> **After the LLM generates a tool call, the runtime parses the tool name and arguments, dispatches the call to the corresponding tool, and executes it. The tool produces a result, which is then appended to the conversation as an observation. The LLM can then use that observation to update its understanding of the current state and decide what to do next.**

### Key Point

Do not say:

> The tool call produces new information.

More precise:

> **The tool execution produces a result.**

The flow is:

```text
LLM
 ↓
Tool Call
 ↓
Runtime
 ↓
Tool Execution
 ↓
Tool Result
 ↓
Observation
 ↓
Updated Context
 ↓
LLM
 ↓
Next Action
```

---

# 3. Why do we need to send the tool result back to the LLM?

### Answer

> **Because the tool result is essentially raw information from the environment. The LLM needs to interpret that information based on the current conversation context, combine it with what it already knows, and reason about what to do next.**

A stronger version:

> **The runtime itself doesn't necessarily know what the result means in the context of the user's goal. The LLM provides that reasoning layer: it interprets the observation, updates its understanding of the task, and decides whether to call another tool or produce a final answer.**

### Example

```text
User:
Find a good date for my trip.

        ↓

Weather Tool
        ↓
32°C, thunderstorms

        ↓

LLM interprets the result

        ↓

Maybe search hotel prices

        ↓

Compare weather + price + user constraints

        ↓

Choose a date
```

The runtime executes tools.

The LLM provides the task-level reasoning and next-action decision.

---

# 4. Why can't the runtime just execute the tool and continue itself?

### Answer

> **Because the runtime usually doesn't have enough semantic context to decide what the tool result means for the user's overall goal. The LLM can combine the observation with the user's intent, previous results, constraints, and other context, and then decide what action should happen next.**

### Key Distinction

```text
Runtime
    ↓
Execution

LLM
    ↓
Interpretation + Reasoning + Next Action
```

The runtime can execute:

```text
search_weather(...)
```

but it generally should not hard-code:

```text
if weather.bad:
    search_hotel()
else:
    book_flight()
```

That would move task-specific decision logic back into deterministic application code.

---

# 5. Why do we need to append the tool result to the conversation?

### Answer

> **Because the LLM needs the observation together with the rest of the task context. If we only keep the latest observation, we may lose important information from the previous context and misunderstand the user's intent.**

### Example

Suppose the user asks:

> Find a good date to travel.

Weather is only one factor.

The agent may also need:

* hotel prices
* flight prices
* travel dates
* user's budget
* school holidays
* previous decisions

If we only keep:

```text
latest_observation = weather result
```

the agent might know:

> There will be thunderstorms.

But it may no longer know:

> The user wants the best travel date based on both weather and hotel price.

Therefore:

> **The conversation history provides the context that allows the LLM to interpret each observation in relation to the overall task.**

---

# 6. Why can't we just keep the latest observation?

### Answer

> **Because an observation is not meaningful in isolation. The LLM needs the original task, user intent, constraints, previous decisions, and relevant observations to determine what the result means and what to do next.**

### Example

```text
User Goal:
Find the best travel date.

Observation 1:
Weather is good on Friday.

Observation 2:
Hotel prices are very high on Friday.

Observation 3:
The user cannot travel during school holidays.
```

The final decision requires all three pieces of information.

Keeping only:

```text
Observation 3
```

would be insufficient.

---

# 7. What if the conversation becomes very long?

### Answer

> **If the conversation becomes too long, it may exceed the LLM's context window. We need to manage the context by summarizing the conversation history and keeping only the information that is relevant to the user's goal and the current task.**

A stronger version:

> **We can summarize previous interactions into a compact context, remove redundant information and low-value historical details, and use that summary together with recent messages for the next reasoning step.**

### Main Problems With Long Context

```text
Long Conversation
       ↓
More Tokens
       ↓
Higher Cost
       ↓
Higher Latency
       ↓
Context Window Limit
       ↓
Potential Context Overflow
```

There is also another problem:

> Important information can become buried under a large amount of irrelevant history.

---

# 8. How would you prevent important information from being lost during context compression?

### Answer

> **We should explicitly preserve important user constraints, key decisions, and the current task state during context compression. These should be treated differently from low-value historical details, because losing them could change the agent's final decision.**

### Important Information

```text
User Constraints
    ↓
Task State
    ↓
Important Decisions
    ↓
Recent Messages
    ↓
Historical Summary
```

A useful design principle:

> **Don't rely on summarization alone for critical state.**

Why?

Because summarization is a form of lossy compression.

### Example

Original:

```text
User:
Don't book anything above $300,
and I cannot travel during school holidays.
```

Bad summary:

```text
User prefers affordable travel.
```

Two hard constraints have been lost:

```text
Budget <= $300
No school holidays
```

The agent could therefore make an incorrect decision.

---

# 9. What should be stored explicitly instead of relying only on conversation history?

### Answer

> **Critical state should be stored explicitly, while conversation history can be compressed.**

For example:

```text
Agent State
├── User Constraints
├── Task State
├── Important Decisions
├── Pending Actions
└── Relevant Results

Conversation History
├── Recent Messages
└── Historical Summary
```

This gives us an important architectural distinction:

```text
Conversation History
        +
Explicit Agent State
        +
Recent Context
        ↓
      LLM
```

This is more reliable than treating the entire conversation history as the only source of state.

---

# 10. How do you prevent an agent from entering an infinite loop?

### Basic Answer

> **A maximum iteration limit is the simplest safeguard against an infinite loop.**

### Senior-Level Answer

> **A maximum iteration limit is the simplest safeguard, but it isn't sufficient by itself. In a production system, I would also use execution timeouts, budget limits, and potentially detect repeated actions or repeated states to identify a stuck agent.**

### Possible Safeguards

```text
Max Iterations
      +
Timeout
      +
Token / Cost Budget
      +
Repeated Action Detection
      +
Repeated State Detection
      +
Cancellation
```

### Example

An agent may not exceed the iteration limit but still be stuck:

```text
search()
search()
search()
search()
...
```

Therefore:

> **Termination control is broader than just max iterations.**

---

# 11. What happens if a tool execution fails?

### Basic Answer

> **We can retry the tool execution with a limited number of attempts. If the retries are exhausted, we should record the failure and either let the agent recover using another action or terminate the task according to the system's error-handling policy.**

### Better Production Answer

> **Not every error should be retried. Transient errors such as network timeouts, rate limits, or temporary service failures may be retryable, while errors such as invalid arguments, permission errors, or missing resources may require a different recovery strategy.**

### Retry Strategy

```text
Tool Execution
      ↓
   Success?
   /     \
 Yes      No
 ↓        ↓
Continue  Classify Error
             ↓
        Retryable?
          /    \
        Yes     No
        ↓        ↓
      Retry   Recovery / Fail
```

Possible retry mechanisms include:

* maximum retry count
* exponential backoff
* jitter
* timeout
* error classification

---

# 12. What should happen after all retries fail?

### Answer

> **After retries are exhausted, the runtime should record the failure and decide whether the agent can recover. For example, the agent may try another tool or another strategy. If recovery is not possible, the runtime can terminate the task and return an appropriate failure result to the user.**

Important:

Do not assume:

```text
Tool failure
    ↓
LLM summarizes
    ↓
Final answer
```

Instead:

```text
Tool Failure
    ↓
Retry / Recovery
    ↓
Alternative Action?
    ↓
Yes → Continue Agent Loop
No  → Terminate Gracefully
```

This is part of **Agent Reliability**.

---

# 13. Why is context management a reliability problem?

### Answer

> **Because incorrect or incomplete context can cause the agent to make incorrect decisions even when the underlying LLM is capable of reasoning correctly.**

For example:

```text
Important Constraint
        ↓
Lost during compression
        ↓
LLM receives incomplete context
        ↓
LLM makes a reasonable decision
        ↓
But the decision is wrong for the user's actual goal
```

Therefore:

> **Context management is not only a cost and performance problem. It is also a correctness and reliability problem.**

---

# 14. What is the difference between conversation history and agent state?

### Status: Follow-up Question

This is the next question to answer.

A good direction to think about is:

```text
Conversation History
    ↓
What happened in the interaction?

Agent State
    ↓
What is the current state of the task?
```

For example:

### Conversation History

```text
User: Find a good travel date.
Assistant: I'll check the weather.
Tool: Friday has good weather.
Assistant: I'll check hotel prices.
Tool: Friday hotel prices are $500.
```

### Agent State

```text
goal:
  find_best_travel_date

constraints:
  budget <= $300

candidate_dates:
  Friday

weather:
  Friday = good

hotel_price:
  Friday = $500

next_action:
  search another date
```

The distinction becomes particularly important for long-running agents.

---



The important engineering questions are not only:

> **Can the agent call a tool?**

but also:

> **What state does the agent maintain?**

> **How does it interpret observations?**

> **How does it decide what to do next?**

> **How does it terminate?**

> **What happens when execution fails?**

> **How does it preserve important information as context grows?**

> **How does it recover from an interrupted or failed task?**

These are the questions that move the discussion from a basic LLM application to **Agent Runtime Engineering**.
