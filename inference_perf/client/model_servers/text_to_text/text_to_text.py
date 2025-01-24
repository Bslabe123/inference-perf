from abc import abstractmethod
import json
import time
from typing import Any, TypedDict

import aiohttp
import requests


from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer

from client.model_servers.client import Model_Server_Client


class Text_To_Text_Request_Settings(TypedDict):
    prompt_len: int
    output_len: int
    best_of: int
    use_beam_search: bool
    top_k: int
    model: str
    timeout: float
    streaming: bool


class Request(TypedDict):
    headers: dict[str, str] | None
    json: dict[str, Any]
    stream: bool


class Response(TypedDict):
    num_output_tokens: int
    request_duration: float
    time_to_first_token: float | None


class Summary(TypedDict):
    pass


class Text_To_Text_Model_Server_Client(Model_Server_Client):
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    @abstractmethod
    def build_request(
        self, prompt: str, settings: Text_To_Text_Request_Settings
    ) -> Request:
        """
        Request headers and bodies depend on the specific model server
        """
        pass

    @abstractmethod
    def parse_response(
        self, response: requests.Response, settings: Text_To_Text_Request_Settings
    ) -> Response:
        """
        Since model server responses are not standardized
        """
        pass

    async def request(
        self, api_url: str, prompt: str, settings: Text_To_Text_Request_Settings
    ) -> Response | Exception:
        request: Request = self.build_request(prompt, settings)
        ttft: float = 0.0
        start_time: float = time.perf_counter()
        output: str = ""
        timeout = aiohttp.ClientTimeout(total=10000)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            try:
                async with session.post(api_url, **request, ssl=False) as response:
                    if settings["streaming"]:
                        async for chunk_bytes in response.content.iter_chunks():
                            chunk_bytes = chunk_bytes[0].strip()
                            if not chunk_bytes:
                                continue
                            timestamp = time.perf_counter()

                            if ttft == 0.0:
                                ttft = timestamp - start_time
                        standardized_resopnse = self.parse_response(response, settings)
                        standardized_resopnse["time_to_first_token"] = ttft
                        return standardized_resopnse
                    else:
                        return self.parse_response(await response, settings)
            except Exception as e:
                self.Errors.record_error(e)
                return e

    def summary(self) -> Summary:
        return {}
        # TODO: Implement me
