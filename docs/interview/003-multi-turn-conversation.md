# Muti-turn Conversation Questions

## Question 1

Can you walk me through how you implemented multi-turn conversation support in your agent runtime?

> I implemented multi-turn conversation support in a few steps.
> First, i defined a Coversation class,It has a messages property that stores the conversation history.
> This allows each turn to access the conversation history and provide it to the LLM as context.
> In other words, the message history is managed by the Conversation, not by an individual turn.

## Question 2

Why did you choose to put the message history in a separate Conversation object instead of keeping it inside the Agent or inside run_turn()?

> Because if the message history is stored inside run_turn(), it would be recreated or lost when the function returns, so the next turn would not have access to the previous messages.
> I also don't want to store the conversation history inside the Agent itself, because one Agent may handle multiple independent conversations. The conversation state should belong to a Conversation object, while the Agent provides the execution capability.

## Question 3

Why shouldn't the conversation history belong to the Agent itself?

> Because one Agent may handle multiple independent conversations. If the conversation history belonged to the Agent itself, the context from different users or conversations could get mixed together.

> So I separate execution from state: The Agent does not own conversation state, while the Conversation object owns the message history and conversational context.