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
import os
import socket
import sys
from typing import Optional

from inference_perf.datagen.base import DataGenerator


class SocketDataServer:
    """
    Serves data from a DataGenerator over a Unix Domain Socket.
    """

    def __init__(self, datagen: DataGenerator, socket_path: str = "/tmp/datagen.sock") -> None:
        self.datagen = datagen
        self.socket_path = socket_path

    def serve(self):
        """
        Starts the socket server and listens for requests.
        """
        import threading
        data_iterator = self.datagen.get_data()
        iterator_lock = threading.Lock()

        def handle_client(conn):
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    
                    try:
                        req = json.loads(data.decode("utf-8").strip())
                    except json.JSONDecodeError:
                        continue

                    if req.get("action") == "get_next":
                        with iterator_lock:
                            try:
                                item = next(data_iterator)
                                item_dict = item.dict()
                                
                                data_type = "unknown"
                                from inference_perf.apis import CompletionAPIData, ChatCompletionAPIData
                                if isinstance(item, CompletionAPIData):
                                    data_type = "completion"
                                elif isinstance(item, ChatCompletionAPIData):
                                    data_type = "chat"

                                response = {"type": data_type, "data": item_dict}
                            except StopIteration:
                                response = {"type": "done"}
                        
                        conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
                        
                        if response.get("type") == "done":
                            break
                    else:
                        pass

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            s.bind(self.socket_path)
            s.listen()
            print(f"DataGen Socket Server listening on {self.socket_path}")

            while True:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=handle_client, args=(conn,))
                client_thread.daemon = True
                client_thread.start()
                pass

        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        print("DataGen Socket Server stopped.")
