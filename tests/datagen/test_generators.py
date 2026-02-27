
import unittest
from unittest.mock import MagicMock, patch
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.datagen.synthetic_datagen import SyntheticDataGenerator
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution
from inference_perf.utils.custom_tokenizer import CustomTokenizer

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 1000
    
    def get_tokenizer(self):
        return self

    def encode(self, text):
        return [1] * len(text)
    
    def decode(self, tokens):
        return " ".join([str(t) for t in tokens])
    
    def __call__(self, text, **kwargs):
        # mimic call behavior
        return {"input_ids": [1] * len(text)}

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.api_config = APIConfig(type=APIType.Completion)
        self.data_config = DataConfig(
            input_distribution=Distribution(min=10, max=10, mean=10, std_dev=0, total_count=100),
            output_distribution=Distribution(min=10, max=10, mean=10, std_dev=0, total_count=100)
        )
        self.mock_tokenizer_wrapper = MagicMock(spec=CustomTokenizer)
        self.mock_hf_tokenizer = MockTokenizer()
        self.mock_tokenizer_wrapper.get_tokenizer.return_value = self.mock_hf_tokenizer
        self.mock_tokenizer_wrapper.count_tokens.side_effect = lambda x: len(x) if isinstance(x, list) else 0

    def test_random_generator_prompts(self):
        generator = RandomDataGenerator(self.api_config, self.data_config, self.mock_tokenizer_wrapper)
        # Force generation
        data = next(generator.get_data())
        payload = generator.load_lazy_data(data)
        
        self.assertIsInstance(payload.prompt, list)
        self.assertIsInstance(payload.prompt[0], int)
        self.assertEqual(len(payload.prompt), 10) # input_len is 10

    def test_synthetic_generator_prompts(self):
        generator = SyntheticDataGenerator(self.api_config, self.data_config, self.mock_tokenizer_wrapper)
        # Force generation
        data = next(generator.get_data())
        payload = generator.load_lazy_data(data)
        
        self.assertIsInstance(payload.prompt, list)
        self.assertIsInstance(payload.prompt[0], int)
        self.assertEqual(len(payload.prompt), 10) # input_len is 10

    def test_mock_generator_prompts(self):
        # Local import to avoid top-level dependency issues if any
        from inference_perf.datagen.mock_datagen import MockDataGenerator
        # MockDataGenerator does not support IO distribution
        mock_config = DataConfig(
            type=self.data_config.type,
            path=self.data_config.path,
            input_distribution=None,
            output_distribution=None
        )
        generator = MockDataGenerator(self.api_config, mock_config, self.mock_tokenizer_wrapper)
        data = next(generator.get_data())
        
        self.assertIsInstance(data.prompt, list)
        self.assertIsInstance(data.prompt[0], int)

if __name__ == '__main__':
    unittest.main()
