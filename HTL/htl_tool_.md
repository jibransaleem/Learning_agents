# Yes — exactly. tool_node in your graph is a ToolNode instance you built like this:
 - pythontool_node = ToolNode([HumanInTheLoop, hotel_search])
- When execution reaches tool_node, it looks at the last message in state["messages"] (the one user_query just produced), reads its tool_calls — which includes the tool name and args the model picked — matches that name against the list you gave it (HumanInTheLoop or hotel_search), and calls the real Python function with those args. All automatic, no manual dispatch code from you.


- tool_node calls hotel_search(...), gets back a dict like {"internet_search": [...]}.
- ToolNode wraps that return value into a ToolMessage object — content is basically the stringified result, and it carries a    tool_call_id linking it back to the exact tool call that requested it.
    That ToolMessage gets appended to state["messages"] (via your add_messages reducer — appended, not overwritten).
    Your edge tool_node → user_query sends execution back to user_query.
    user_query runs again, calling model.invoke(state["messages"]) — and now state["messages"] includes that ToolMessage with the actual search results sitting right there in the history.
    The model reads it like any other message in the conversation, and either: decides it has enough info and responds with plain content (no more tool_calls) → routes to make_res, or decides it needs another tool call → loops again.

So concretely: the search result never goes into a special "hotel_list" field or gets handled separately — it just becomes part of the conversation the model sees. Your make_res node is the one place where you deliberately pull that raw message history back out and ask the model to reformat it into structured Hotel objects — that's the only point where the loose ToolMessage content gets turned into something structured again.so why we are passing complte mesaegs to user query not juist last message ?Weighed context retention necessity and API statelessness constraintsWeighed context retention necessity and API statelessness constraintsGood question — two separate reasons, one technical, one behavioral.
1. The technical reason: the API itself has no memory. ChatGroq/model.invoke() is a stateless HTTP call — Groq's servers don't remember your last request at all. Every single .invoke() call has to include everything the model needs to know, because as far as the API is concerned, each call is a brand new conversation from scratch. If you only sent the last message, the model would have zero idea what was searched, what the original question was, or what it already tried.
2. The behavioral reason: the model needs to reason over what's already happened. Concretely, in your hotel agent:

It needs to remember the original user query while looking at search results (otherwise "what did they actually ask for?" is lost).
It needs to see its own previous tool call and that tool's result to decide "do I have enough now, or do I need another search?" — that's literally how the loop terminates. Without seeing the ToolMessage from the last search, it has no way to judge "was that good enough."
If HumanInTheLoop was used, it needs to see the human's clarifying answer ("Karachi") in context — not as an isolated fact, but attached to the original ambiguous request it was responding to.

What happens if you passed only the last message instead (worth trying once as a thought experiment): after a tool call, the last message would just be the raw ToolMessage — the model would see search results with zero memory of what question those results were even answering. It'd have no way to decide if the results satisfy the original ask, or connect a human's one-word answer back to what was being clarified.
One thing to watch as a downside, which you'll hit eventually: passing the entire growing history means token cost climbs every loop iteration — same issue you ran into with the memory chat agent, which is why that one needed summarization. This hotel agent doesn't need it yet since the loop is short, but it's the same underlying tradeoff.