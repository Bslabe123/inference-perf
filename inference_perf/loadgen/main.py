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

import asyncio
import logging
import time
from argparse import ArgumentParser
from typing import Optional

from inference_perf.config import read_config, ModelServerType
from inference_perf.loadgen import LoadGenerator
from inference_perf.transport.socket_client import SocketDataGenerator
from inference_perf.client.modelserver import (
    vLLMModelServerClient,
    MockModelServerClient,
)
from inference_perf.client.requestdatacollector.jsonl import JSONLRequestDataCollector
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description="Standalone Load Generator")
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument("--socket-path", default="/tmp/datagen.sock", help="Path to DataGen Unix socket")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL file for metrics")
    args = parser.parse_args()

    config = read_config(args.config)

    # 1. Create Tokenizer (if needed by clients, although LoadGen itself might not need it if DataGen sends pre-tokenized or if client handles it)
    tokenizer: Optional[CustomTokenizer] = None
    if config.tokenizer and config.tokenizer.pretrained_model_name_or_path:
        tokenizer = CustomTokenizer(config.tokenizer)

    # 2. Create JSONL Collector
    collector = JSONLRequestDataCollector(args.output_jsonl)

    # 3. Create Model Server Client
    model_server_client = None
    if config.server:
        if config.server.type == ModelServerType.VLLM:
            model_server_client = vLLMModelServerClient(
                collector,
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=[], # Simplified for standalone
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
                lora_config=config.load.lora_traffic_split,
            )
        elif config.server.type == ModelServerType.MOCK:
            model_server_client = MockModelServerClient(
                collector,
                api_config=config.api,
                timeout=config.load.request_timeout,
            )
        else:
            raise Exception(f"Unsupported model server type for standalone loadgen: {config.server.type}")
    else:
        raise Exception("Model server config missing")

    # 4. Create Socket Data Generator (Client)
    datagen = SocketDataGenerator(config.api, config.data, tokenizer, socket_path=args.socket_path)

    # 5. Create Load Generator
    loadgen = LoadGenerator(datagen, config.load)

    # 6. Run Load Generator
    async def _run():
        await loadgen.run(model_server_client)

    logger.info(f"Starting Load Generator. Writing results to {args.output_jsonl}")
    asyncio.run(_run())
    logger.info("Load Generator finished.")

    # Save stage_runtime_info
    import json
    stage_info_path = args.output_jsonl.replace(".jsonl", "_stage_info.json")
    with open(stage_info_path, "w") as f:
        # Convert objects to dicts
        info_dict = {
            str(k): {
                "stage_id": v.stage_id,
                "rate": v.rate,
                "start_time": v.start_time,
                "end_time": v.end_time,
                "status": v.status.value,
                "concurrency_level": v.concurrency_level,
            }
            for k, v in loadgen.stage_runtime_info.items()
        }
        json.dump(info_dict, f)
    logger.info(f"Saved stage info to {stage_info_path}")


if __name__ == "__main__":
    main()
