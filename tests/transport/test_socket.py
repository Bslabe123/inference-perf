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
import threading
import unittest
from typing import List, Optional

from inference_perf.apis import CompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.transport.socket_client import SocketDataGenerator


class TestSocketDataGenerator(unittest.TestCase):

    def setUp(self):
        self.api_config = APIConfig(type=APIType.Completion)
        self.data_config = DataConfig(type="mock")
        
        import os
        self.socket_path = "/tmp/test_datagen.sock"
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
            
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen()
        
        self.server_thread = threading.Thread(target=self.run_mock_server)
        self.server_thread.start()

    def tearDown(self):
        self.server_thread.join()
        import os
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    def run_mock_server(self):
        conn, addr = self.server_sock.accept()
        with conn:
            data = conn.recv(1024)
            if data:
                req = json.loads(data.decode("utf-8").strip())
                if req.get("action") == "get_next":
                    response = {
                        "type": "completion",
                        "data": {
                            "prompt": "Hello socket",
                            "max_tokens": 10,
                            "model_response": ""
                        }
                    }
                    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")

            data = conn.recv(1024)
            if data:
                req = json.loads(data.decode("utf-8").strip())
                if req.get("action") == "get_next":
                    response = {"type": "done"}
                    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
        self.server_sock.close()

    def test_get_data(self):
        generator = SocketDataGenerator(
            self.api_config, self.data_config, tokenizer=None, socket_path=self.socket_path
        )
        data_items = list(generator.get_data())
        
        self.assertEqual(len(data_items), 1)
        self.assertIsInstance(data_items[0], CompletionAPIData)
        self.assertEqual(data_items[0].prompt, "Hello socket")


if __name__ == "__main__":
    unittest.main()
