# ollama-wrapper.py
# Author: Mikalai Varapai (TU/e stnr. 1945491)

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Any, Iterable, Iterator, Mapping
import httpx

Message = dict[str, str]

@dataclass
class OllamaBatchPrompt:
  model: str
  template: str

  system: str | None = None
  base_url: str = "http://localhost:11434"

  temperature: float = 0.2
  num_predict: int = 512
  keep_alive: str = "10m"

  options: dict[str, Any] = field(default_factory=dict)
  timeout: float | None = None

  _client: httpx.Client | None = field(default=None, init=False, repr=False)

  # create httpx client when entering `with` statement
  def __enter__(self) -> "OllamaBatchPrompt":
    self._client = httpx.Client(
      base_url=self.base_url,
      timeout=self.timeout,
      limits=httpx.Limits(
        max_connections=1,
        max_keepalive_connections=1,
      ),
    )
    return self
  
  # close httpx client after exiting `with` statement
  def __exit__(self, *args: object) -> None:
    if self._client is not None:
      self._client.close()
      self._client = None

  # helper function to substitute template parameters in the prompt
  def _render(self, params: Mapping[str, Any]) -> str:
    return Template(self.template).substitute(**params)
  
  # construct message list to pass into the API request.
  # appends system prompt if defined.
  def _messages(self, prompt: str) -> list[Message]:
    messages: list[Message] = []

    if self.system:
      messages.append({
        "role": "system",
        "content": self.system,
      })

    messages.append({
      "role": "user",
      "content": prompt,
    })

    return messages
  
  # constructs options list to pass into the API request.
  # uses override values if defined.
  def _options(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    opts = {
      "temperature": self.temperature,
      "num_predict": self.num_predict,
      **self.options,
    }

    if overrides:
      opts.update(overrides)

    return opts
  
  # guarantees an httpx client object for further execution.
  def _client_or_create(self) -> httpx.Client:
    if self._client is None:
      self._client = httpx.Client(
        base_url=self.base_url,
        timeout=self.timeout,
        limits=httpx.Limits(
          max_connections=1,
          max_keepalive_connections=1,
        ),
      )

  # send one prompt and return the response.
  def ask_one(
    self,
    params: Mapping[str, Any], # construct the prompt using these substitutions
    *,
    options: Mapping[str, Any] | None = None,
  ) -> str:
    prompt = self._render(params)

    payload = {
      "model": self.model,
      "messages": self._messages(prompt),
      "stream": False,
      "keep_alive": self.keep_alive,
      "options": self._options(options),
    }

    client = self._client_or_create()
    response = client.post("/api/chat", json=payload)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]
  
  # main function of the wrapper.
  def stream_prompts(
    self,
    items: Iterable[Mapping[str, Any]],
    *,
    options: Mapping[str, Any] | None = None,
  ) -> Iterator[str]:
    for params in items:
      yield self.ask_one(params, options=options)

  def close(self) -> None:
    if self._client is not None:
      self._client.close()
      self._client = None