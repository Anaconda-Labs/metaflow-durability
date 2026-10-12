"""AgentDecorator — Metaflow StepDecorator for the @agent pattern.

Runs the full LLM loop within one Metaflow step. Tools are user-provided
callables; schemas are auto-generated via openai-function-calling.

LLM resolution (in priority order):
1. ``client=`` parameter — user provides an OpenAI-compatible client
2. Co-stacked inference decorator (``@llamacpp``, ``@vllm``) — detected
   at runtime via the step's decorator list
3. ``model=`` parameter — passed to litellm with any extra kwargs

This module is imported lazily by Metaflow's extension loader. Users
access the decorator via ``from metaflow import agent``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from metaflow.decorators import StepDecorator


def _build_tool_schemas(
    tools: list[Callable[..., Any]],
) -> list[dict[str, Any]]:
    """Generate OpenAI function-calling schemas from typed callables."""
    from openai_function_calling import FunctionInferrer

    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "schema"):
            schemas.append(tool.schema)
        else:
            inferred = FunctionInferrer.infer_from_function_reference(tool)
            schemas.append({"type": "function", "function": inferred.to_json_schema()})
    return schemas


def _build_tool_registry(
    tools: list[Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Map function names to callables for dispatch."""
    return {tool.__name__: tool for tool in tools}


def _resolve_completion_fn(
    attributes: dict[str, Any],
    step_decorators: list[Any],
) -> Callable[..., Any]:
    """Resolve the LLM completion function from available sources.

    Priority:
    1. Explicit client object → client.chat.completions.create
    2. Co-stacked inference decorator → decorator's OpenAI client
    3. litellm model string → litellm.completion with kwargs
    """
    client = attributes.get("client")
    if client is not None:

        def _client_completion(
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            call_kwargs: dict[str, Any] = {"messages": messages}
            if tools:
                call_kwargs["tools"] = tools
            return client.chat.completions.create(**call_kwargs)

        return _client_completion

    # Check for co-stacked inference decorators (@llamacpp, @vllm)
    for deco in step_decorators:
        if hasattr(deco, "get_openai_client"):
            infer_client = deco.get_openai_client()

            def _infer_completion(
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                _c: Any = infer_client,
                **kwargs: Any,
            ) -> Any:
                call_kwargs: dict[str, Any] = {"messages": messages}
                if tools:
                    call_kwargs["tools"] = tools
                return _c.chat.completions.create(**call_kwargs)

            return _infer_completion

    model = attributes.get("model")
    if model is not None:
        import litellm

        litellm.suppress_debug_info = True
        litellm_kwargs = dict(attributes.get("litellm_kwargs") or {})

        def _litellm_completion(
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                **litellm_kwargs,
            }
            if tools:
                call_kwargs["tools"] = tools
            return litellm.completion(**call_kwargs)

        return _litellm_completion

    raise ValueError(
        "@agent requires one of: model= (litellm string), client= (OpenAI-compatible), "
        "or a co-stacked inference decorator (@llamacpp, @vllm)."
    )


def _run_agent_loop(
    flow: Any,
    completion_fn: Callable[..., Any],
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    tool_registry: dict[str, Callable[..., Any]],
    max_iterations: int,
) -> None:
    """Run the full agent loop: LLM → tools → LLM → ... → done."""
    from metaflow_durability.dispatch import InProcessExecutor

    messages: list[dict[str, Any]] = list(flow.messages)
    all_dispatched: list[dict[str, Any]] = list(getattr(flow, "dispatched_tools", []))

    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system_prompt}, *messages]

    executor = InProcessExecutor(tool_registry)

    for iteration in range(max_iterations):
        response = completion_fn(
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
        )

        choice = response.choices[0]
        message = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        messages = [*messages, assistant_msg]

        if not message.tool_calls:
            flow.messages = messages
            flow.final_response = message.content
            flow.iteration = iteration
            flow.dispatched_tools = all_dispatched
            return

        parsed_calls: list[dict[str, Any]] = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            parsed_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                    "tier": "default",
                }
            )

        dispatch_result = executor.execute(parsed_calls)
        messages = [*messages, *dispatch_result.tool_messages]
        all_dispatched.extend(dispatch_result.to_dicts())

        tc_names = [d["tool_name"] for d in dispatch_result.to_dicts()]
        print(f"  iter {iteration}: [LLM -> {', '.join(tc_names)}]")

    flow.messages = messages
    flow.final_response = None
    flow.iteration = max_iterations
    flow.dispatched_tools = all_dispatched


class AgentDecorator(StepDecorator):
    """Metaflow step decorator that runs the full agent loop within one step.

    LLM source (pick one):
    - ``model`` — litellm model string (e.g. "anthropic/claude-sonnet-4")
    - ``client`` — pre-configured OpenAI-compatible client object
    - neither — auto-detect from co-stacked @llamacpp / @vllm decorator

    For litellm, pass provider-specific kwargs via ``litellm_kwargs``::

        @agent(
            model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
            litellm_kwargs={"aws_region_name": "us-west-2"},
            tools=[search],
        )
    """

    name = "agent"

    defaults: dict[str, Any] = {
        "model": None,
        "client": None,
        "system_prompt": "You are a helpful assistant with access to tools.",
        "tools": None,
        "tool_schemas": None,
        "litellm_kwargs": None,
        "max_iterations": 10,
    }  # type: ignore[assignment]

    _tool_schemas: list[dict[str, Any]] = []
    _tool_registry: dict[str, Callable[..., Any]] = {}
    _step_decorators: list[Any] = []

    def step_init(
        self,
        flow,
        graph,
        step_name,
        decorators,
        environment,
        flow_datastore,
        logger,
    ):
        """Resolve tool schemas/registry and capture sibling decorators."""
        tools = self.attributes.get("tools")
        tool_schemas = self.attributes.get("tool_schemas")

        if tools:
            self._tool_schemas = (
                tool_schemas if tool_schemas else _build_tool_schemas(tools)
            )
            self._tool_registry = _build_tool_registry(tools)
        elif tool_schemas:
            self._tool_schemas = tool_schemas
            self._tool_registry = {}
        else:
            self._tool_schemas = []
            self._tool_registry = {}

        self._step_decorators = list(decorators)

        # Auto-attach @card(type="agent") if not already present
        try:
            from metaflow.plugins.cards.card_decorator import CardDecorator

            has_agent_card = any(
                isinstance(d, CardDecorator) and d.attributes.get("type") == "agent"
                for d in decorators
            )
            if not has_agent_card:
                card_attrs = CardDecorator.defaults.copy()
                card_attrs["type"] = "agent"
                card_attrs["id"] = "agent"
                card_deco = CardDecorator(attributes=card_attrs)
                decorators.append(card_deco)
                card_deco.step_init(
                    flow,
                    graph,
                    step_name,
                    decorators,
                    environment,
                    flow_datastore,
                    logger,
                )
        except ImportError:
            pass

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        """Initialize artifacts before the step runs."""
        if not hasattr(flow, "messages") or flow.messages is None:
            flow.messages = []
        if not hasattr(flow, "iteration"):
            flow.iteration = 0
        if not hasattr(flow, "dispatched_tools"):
            flow.dispatched_tools = []
        if not hasattr(flow, "final_response"):
            flow.final_response = None

    def task_decorate(
        self,
        step_func,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
    ):
        """Wrap step to run the full agent loop, then the user's step body."""

        def _wrapped_step():
            step_func()
            completion_fn = _resolve_completion_fn(
                self.attributes, self._step_decorators
            )
            _run_agent_loop(
                flow=flow,
                completion_fn=completion_fn,
                system_prompt=self.attributes["system_prompt"],
                tool_schemas=self._tool_schemas,
                tool_registry=self._tool_registry,
                max_iterations=self.attributes["max_iterations"],
            )

        return _wrapped_step

    def task_post_step(
        self, step_name, flow, graph, retry_count, max_user_code_retries
    ):
        """Ensure dispatched_tools artifact exists."""
        if not hasattr(flow, "dispatched_tools"):
            flow.dispatched_tools = []
