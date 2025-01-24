from typing import Any, List

import requests
from client.model_servers.text_to_text.text_to_text import (
    Response,
    Text_To_Text_Model_Server_Client,
    Text_To_Text_Request_Settings,
)


class Jetstream_Client(Text_To_Text_Model_Server_Client):

    def build_request(
        self, prompt: str, settings: Text_To_Text_Request_Settings
    ) -> Any:

        return {
            "json": {
                "prompt": prompt,
                "max_tokens": settings["output_len"],
            }
        }

    def parse_response(
        self, response: requests.Response, settings: Text_To_Text_Request_Settings
    ) -> Response:
        res: List[Any] = []  # response["response"]
        output_token_ids = self.tokenizer(res).input_ids

        return {
            "num_output_tokens": len(output_token_ids),
            "request_duration": 0.0,
            "time_to_first_token": None,
        }
        pass
