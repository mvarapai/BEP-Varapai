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

  temperature: float = 0.0
  num_predict: int = 3
  keep_alive: str = "30m"

  options: dict[str, Any] = field(default_factory=dict)
  timeout: float | None = 60.0

  _client: httpx.Client | None = field(default=None, init=False, repr=False)

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

  def __exit__(self, *args: object) -> None:
    self.close()

  def close(self) -> None:
    if self._client is not None:
      self._client.close()
      self._client = None

  def _client_or_create(self) -> httpx.Client:
    if self._client is None:
      self.__enter__()

    assert self._client is not None
    return self._client

  def _render(self, params: Mapping[str, Any]) -> str:
    return Template(self.template).substitute(**params)

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

  def _request_options(self) -> dict[str, Any]:
    return {
      "temperature": self.temperature,
      "num_predict": self.num_predict,
      **self.options,
    }

  def ask_one(self, params: Mapping[str, Any]) -> str:
    prompt = self._render(params)

    payload = {
      "model": self.model,
      "messages": self._messages(prompt),
      "stream": False,
      "keep_alive": self.keep_alive,
      "options": self._request_options(),
    }

    client = self._client_or_create()
    response = client.post("/api/chat", json=payload)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()

  def stream_prompts(
    self,
    items: Iterable[Mapping[str, Any]],
  ) -> Iterator[str]:
    for item in items:
      yield self.ask_one(item)