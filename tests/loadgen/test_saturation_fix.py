
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import sys
import time
import inspect
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType, SweepConfig, StageGenType
from inference_perf.client.modelserver import ModelServerClient
from typing import Any

# Patch asyncio.TaskGroup for Python < 3.11 if needed
if sys.version_info < (3, 11):
    class MockTaskGroup:
        async def __aenter__(self) -> "MockTaskGroup":
            return self
        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass
        def create_task(self, coro: Any) -> Any:
            return asyncio.create_task(coro)
    asyncio.TaskGroup = MockTaskGroup

# Patch typing.TypeAlias for Python < 3.10 if needed
if sys.version_info < (3, 10):
    import typing
    typing.TypeAlias = typing.Any

class TestSaturationFix(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_datagen = MagicMock()
        self.mock_datagen.trace = None
        self.mock_datagen.get_request_count.return_value = 1000
        self.mock_datagen.get_data.return_value = iter([MagicMock(prefered_worker_id=-1) for _ in range(1000)])

        self.sweep_config = SweepConfig(
            type=StageGenType.LINEAR,
            num_requests=100,
            num_stages=1, # Single stage for simple verification
            stage_duration=10, # 10s duration
            saturation_percentile=90,
            timeout=20
        )
        self.load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=1,
            worker_max_concurrency=10,
            sweep=self.sweep_config,
            stages=[]
        )
        
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    @patch("inference_perf.loadgen.load_generator.time.perf_counter")
    @patch("inference_perf.loadgen.load_generator.sleep")
    async def test_saturation_accounting_for_warmup(self, mock_sleep, mock_perf_counter) -> None:
        client = MagicMock(spec=ModelServerClient)
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        finished_requests_counter = MagicMock()
        request_phase = MagicMock()
        cancel_signal = MagicMock()
        
        # Setup Time Simulation
        current_time = 0.0
        def get_time():
            nonlocal current_time
            return current_time
        
        mock_perf_counter.side_effect = get_time
        
        # Smart sleep mock that advances time only when called by driver (run_stage)
        async def fake_sleep(seconds):
            nonlocal current_time
            # Check who called us
            # Stack: [0]=fake_sleep, [1]=caller
            # We look for 'aggregator' in the stack
            is_aggregator = False
            for frame in inspect.stack():
                if frame.function == "aggregator":
                    is_aggregator = True
                    break
            
            if is_aggregator:
                # Aggregator waits for time to pass
                wake_time = current_time + seconds
                while current_time < wake_time:
                    await asyncio.sleep(0)
            else:
                # Driver (mock_run_stage) advances time
                current_time += seconds
                await asyncio.sleep(0)

        mock_sleep.side_effect = fake_sleep
        
        # Mock run_stage to simulate REAL behavior: 1s warmup, then linear requests
        async def mock_run_stage(stage_id, rate, duration, *args, **kwargs):
            nonlocal current_time
            
            # Simulate 1s warmup
            await fake_sleep(1.0)
            
            # Now simulate request processing for 'duration' seconds
            # We update finished_requests_counter over time
            start_generating_time = current_time
            end_generating_time = start_generating_time + duration
            
            target_total_requests = int(rate * duration)
            
            # Iterate in steps to allow aggregator to sample
            step_size = 0.5
            while current_time < end_generating_time:
                # Advance time manually to update counter BEFORE yielding to aggregator
                current_time += step_size
                
                elapsed = current_time - start_generating_time
                progress = min(1.0, elapsed / duration)
                
                with finished_requests_counter.get_lock():
                    finished_requests_counter.value = int(target_total_requests * progress)
                
                # Yield to let aggregator run (it waits for time to advance)
                await asyncio.sleep(0)
            
            # Ensure final value is set
            with finished_requests_counter.get_lock():
                finished_requests_counter.value = target_total_requests
            
            # Wait a bit to let aggregator capture the final value (since aggregator runs concurrently)
            await fake_sleep(0.5)

        self.load_generator.run_stage = AsyncMock(side_effect=mock_run_stage)

        finished_requests_counter.get_lock = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__ = MagicMock()
        finished_requests_counter.get_lock.return_value.__exit__ = MagicMock()
        finished_requests_counter.value = 0

        # We set rate=100.
        # With bug: measured ~90. Saturation detected (90 < 0.9*100 is False, but 89.9 < 90 is True).
        # We expect test to FAIL if saturation is detected incorrectly.
        
        await self.load_generator.preprocess(
            client,
            request_queue,
            active_requests_counter,
            finished_requests_counter,
            request_phase,
            cancel_signal
        )
        
        # Check generated stages
        max_stage_rate = max(s.rate for s in self.load_generator.stages)
        print(f"Max stage rate: {max_stage_rate}")
        
        # Bug present -> Saturation detected at ~90 -> Max Rate ~ 180.
        # Fixed -> No saturation -> Max Rate ~ 200.
        
        self.assertGreater(max_stage_rate, 195.0, "Throughput was underestimated, saturation detected incorrectly!")

if __name__ == "__main__":
    unittest.main()
