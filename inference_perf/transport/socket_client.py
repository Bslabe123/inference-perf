# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import socket
from typing import Generator, List, Optional

from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.base import DataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class SocketDataGenerator(DataGenerator):
    """
    A DataGenerator that pulls data from a remote socket service.
    This allows decoupling the data generation process from the load generator.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer], socket_path: str = "/tmp/datagen.sock") -> None:
        super().__init__(api_config, config, tokenizer)
        self.socket_path = socket_path

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        """
        Connects to the datagen service via Unix Domain Socket and yields data items.
        """
        import time
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            max_retries = 5
            for i in range(max_retries):
                try:
                    s.connect(self.socket_path)
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    if i == max_retries - 1:
                        raise Exception(f"Failed to connect to DataGen service at {self.socket_path} after {max_retries} attempts.")
                    time.sleep(1)

            while True:
                # Send request for next data item
                request = {"action": "get_next"}
                s.sendall(json.dumps(request).encode("utf-8") + b"\n")

                # Receive response
                response_data = b""
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response_data += chunk
                    if b"\n" in chunk:
                        break

                if not response_data:
                    break # Connection closed

                try:
                    response_json = json.loads(response_data.decode("utf-8").strip())
                except json.JSONDecodeError:
                    continue # Or raise error

                if response_json.get("type") == "done":
                    break

                data_type = response_json.get("type")
                data_dict = response_json.get("data")

                if data_type == "completion":
                    yield CompletionAPIData(**data_dict)
                elif data_type == "chat":
                    yield ChatCompletionAPIData(**data_dict)
                else:
                    raise Exception(f"Unknown data type received from socket: {data_type}")

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True
