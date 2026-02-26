
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import sys
import inspect
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType, SweepConfig, StageGenType
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.circuit_breaker import CircuitBreaker

# Compatibility patches
from typing import Any
if sys.version_info < (3, 11):
    class MockTaskGroup:
        async def __aenter__(self) -> "MockTaskGroup":
            return self
        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass
        def create_task(self, coro: Any) -> Any:
            return asyncio.create_task(coro)
    asyncio.TaskGroup = MockTaskGroup

if sys.version_info < (3, 10):
    import typing
    typing.TypeAlias = typing.Any

class TestSaturationRobustness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_datagen = MagicMock()
        self.mock_datagen.trace = None
        self.mock_datagen.get_request_count.return_value = 1000
        self.mock_datagen.get_data.return_value = iter([MagicMock(prefered_worker_id=-1) for _ in range(10000)]) # Plenty of data

        self.sweep_config = SweepConfig(
            type=StageGenType.LINEAR,
            num_requests=100, # Target max QPS
            num_stages=5,     # Steps: 20, 40, 60, 80, 100
            stage_duration=5,
            saturation_percentile=90,
            timeout=10
        )
        self.load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=1,
            worker_max_concurrency=100,
            sweep=self.sweep_config,
            stages=[]
        )
        
        self.circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.circuit_breaker.is_open.return_value = False
        self.circuit_breaker.name = "mock_cb"

        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    @patch("inference_perf.loadgen.load_generator.time.perf_counter")
    @patch("inference_perf.loadgen.load_generator.sleep")
    async def run_preprocess_with_simulation(self, behavior_func, mock_sleep, mock_perf_counter):
        """
        Helper to run preprocess with a simulated server behavior.
        behavior_func(rate) -> achieved_throughput
        """
        client = MagicMock(spec=ModelServerClient)
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        finished_requests_counter = MagicMock()
        request_phase = MagicMock()
        cancel_signal = MagicMock()
        
        current_time = 0.0
        def get_time():
            nonlocal current_time
            return current_time
        mock_perf_counter.side_effect = get_time

        async def fake_sleep(seconds):
            nonlocal current_time
            # Check stack for aggregator
            is_aggregator = False
            for frame in inspect.stack():
                if frame.function == "aggregator":
                    is_aggregator = True
                    break
            
            if is_aggregator:
                wake_time = current_time + seconds
                while current_time < wake_time:
                    await asyncio.sleep(0)
            else:
                current_time += seconds
                await asyncio.sleep(0)
        mock_sleep.side_effect = fake_sleep

        async def mock_run_stage(stage_id, rate, duration, *args, **kwargs):
            nonlocal current_time
            # check CB in run_stage loop
            # logic: if CB open, exit early.
            # loadgen checks CB inside the loop. 
            # We need to simulate that check if we want to test correct exit.
            # But here we are mocking run_stage. 
            # So we must reproduce the CB check logic IF we want to test it via run_stage side effect.
            # OR we rely on loadgen.preprocess calling run_stage, and if run_stage returns, it checks measurement.
            
            # If behavior_func returns None, it means "crash/stop" or CB trip?
            # Let's say behavior_func determines throughput.
            
            achieved_throughput = behavior_func(rate, duration)
            if achieved_throughput is None:
                 # Simulate CB trip or abort
                 return

            # Simulate 1s warmup
            await fake_sleep(1.0)
            
            start_generating_time = current_time
            end_generating_time = start_generating_time + duration
            target_total_requests = int(achieved_throughput * duration)
            
            # Linear update
            step_size = 0.5
            while current_time < end_generating_time:
                current_time += step_size
                elapsed = current_time - start_generating_time
                progress = min(1.0, elapsed / duration)
                with finished_requests_counter.get_lock():
                    finished_requests_counter.value = int(target_total_requests * progress)
                await asyncio.sleep(0)

            # Finalize
            with finished_requests_counter.get_lock():
                finished_requests_counter.value = target_total_requests
            await fake_sleep(0.5)

        self.load_generator.run_stage = AsyncMock(side_effect=mock_run_stage)
        
        # Setup counters
        finished_requests_counter.get_lock = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__ = MagicMock()
        finished_requests_counter.get_lock.return_value.__exit__ = MagicMock()
        finished_requests_counter.value = 0

        await self.load_generator.preprocess(
            client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
        )

    async def test_linear_success(self):
        """Test ideal case where server handles all load."""
        def behavior(rate, duration):
            return rate # Server perfectly matches target rate
        
        await self.run_preprocess_with_simulation(behavior)
        
        # Should not have detected saturation. Max rate used for generation should be ~200 (100 * 2).
        max_rate = max(s.rate for s in self.load_generator.stages)
        self.assertAlmostEqual(max_rate, 200.0, delta=1.0)

    async def test_hard_saturation_limit(self):
        """Test server capping at 50 QPS."""
        # Steps: 20, 40, 60, 80, 100.
        # 20 -> 20 (OK)
        # 40 -> 40 (OK)
        # 60 -> 50 (Fail: 50 < 60*0.9=54) -> Saturation detected!
        
        def behavior(rate, duration):
            return min(rate, 50.0)
            
        await self.run_preprocess_with_simulation(behavior)
        
        # Saturation detected at step 3 (Target 60).
        # Measured throughput ~50.
        # New stages generated based on 50 * 2 = 100 QPS max.
        max_rate = max(s.rate for s in self.load_generator.stages)
        self.assertAlmostEqual(max_rate, 100.0, delta=1.0)

    async def test_early_saturation(self):
        """Test server saturated immediately (e.g. 10 QPS cap)."""
        # Step 1: 20 QPS.
        # Measured: 10.
        # 10 < 20*0.9 (18). Saturation detected immediately.
        
        def behavior(rate, duration):
            return min(rate, 10.0)
        
        await self.run_preprocess_with_simulation(behavior)
        
        # Saturation detected at step 1.
        # Measured 10. Stages based on 20 max.
        max_rate = max(s.rate for s in self.load_generator.stages)
        self.assertAlmostEqual(max_rate, 20.0, delta=1.0)
        
    async def test_tolerance_boundary(self):
        """Test throughput exactly at 89% (fail) and 91% (pass)."""
        # We need precise control.
        # Let's set tolerance to 0.9 in code (default is hardcoded 0.90 in preprocess).
        
        # Case 1: 89% -> Fail
        # Use single step to isolate.
        self.sweep_config.num_stages = 1
        self.sweep_config.num_requests = 100
        
        def behavior_fail(rate, duration):
            return rate * 0.89
            
        await self.run_preprocess_with_simulation(behavior_fail)
        max_rate_fail = max(s.rate for s in self.load_generator.stages)
        # Saturated at ~89. Max rate ~178.
        self.assertLess(max_rate_fail, 180.0)
        
        # Case 2: 91% -> Pass
        # Reset load generator
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)
            
        def behavior_pass(rate, duration):
            return rate * 0.91
            
        await self.run_preprocess_with_simulation(behavior_pass)
        max_rate_pass = max(s.rate for s in self.load_generator.stages)
        self.assertGreater(max_rate_pass, 180.0)

    async def test_low_throughput(self):
        """Test very low throughput (0.25 QPS)."""
        # Target: 0.25 QPS. Duration: 20s. Expect 5 requests.
        # If we measure 4 requests in ~19s, we get ~0.21 QPS. 0.21 < 0.225 (90% of 0.25).
        # We need more samples to smooth out discrete quantization error.
        # Let's target 0.5 QPS for 20s -> 10 requests.
        # Or Just increase duration for 0.25 QPS to 40s -> 10 requests.
        self.sweep_config.num_requests = 1 
        self.sweep_config.num_stages = 4   # 0.25, 0.5, 0.75, 1.0
        self.sweep_config.stage_duration = 40 # Increased to 40s
        
        # Reset load gen
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

        def behavior(rate, duration):
            return rate # Ideal behavior
            
        await self.run_preprocess_with_simulation(behavior)
        
        # Should not saturate.
        # Max rate ~ 1.0 * 2 = 2.0.
        max_rate = max(s.rate for s in self.load_generator.stages)
        print(f"DEBUG: Low Throughput Max Rate: {max_rate}")
        self.assertAlmostEqual(max_rate, 2.0, delta=0.5)

    async def test_high_throughput(self):
        """Test very high throughput (1000 QPS)."""
        self.sweep_config.num_requests = 1000
        self.sweep_config.num_stages = 5
        self.sweep_config.stage_duration = 5
        
        # Reset load gen
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)
            
        def behavior(rate, duration):
            return rate # Ideal
            
        await self.run_preprocess_with_simulation(behavior)
        
        # Should not saturate.
        # Max rate ~ 2000.
        max_rate = max(s.rate for s in self.load_generator.stages)
        self.assertAlmostEqual(max_rate, 2000.0, delta=10.0)


if __name__ == "__main__":
    unittest.main()
