
import unittest
from unittest.mock import MagicMock, PropertyMock
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.config import APIConfig, DataConfig, SharedPrefix, CustomTokenizerConfig, APIType
from transformers import AutoTokenizer

class TestSharedPrefixDataGenerator(unittest.TestCase):
    def setUp(self):
        self.api_config = MagicMock(spec=APIConfig)
        self.api_config.type = APIType.Completion
        self.config = MagicMock(spec=DataConfig)
        self.config.input_distribution = None
        self.config.output_distribution = None
        self.config.trace = None
        self.mock_tokenizer_wrapper = MagicMock()
        
        # Mock tokenizer that behaves deterministically:
        # id I -> "word{I}"
        # text "wordI wordJ" -> [I, J] (assuming space separation)
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = 100
        self.mock_tokenizer.model_max_length = 1024
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token = None
        
        def decode(token_ids, skip_special_tokens=True):
            return " ".join([f"word{t}" for t in token_ids])
            
        def batch_decode(token_ids_list, skip_special_tokens=True):
            return [decode(ids) for ids in token_ids_list]
            
        def encode(text, add_special_tokens=False, truncation=True, max_length=None):
            # parse back "word{I}"
            tokens = []
            if not text:
                return {"input_ids": []}
            parts = text.split(" ")
            for p in parts:
                if p.startswith("word"):
                    try:
                        tokens.append(int(p[4:]))
                    except ValueError:
                        pass # Should not happen in this controlled test
            return {"input_ids": tokens}
            
        self.mock_tokenizer.decode.side_effect = decode
        self.mock_tokenizer.batch_decode.side_effect = batch_decode
        self.mock_tokenizer.side_effect = encode
        
        self.mock_tokenizer_wrapper.get_tokenizer.return_value = self.mock_tokenizer
        
    def test_prompt_length_consistency(self):
        # Setup specific config for this test
        # We want strict length checking
        shared_prefix_config = SharedPrefix(
            num_groups=1,
            num_prompts_per_group=10,
            system_prompt_len=10,
            question_len=20,
            output_len=10,
            enable_multi_turn_chat=False
        )
        self.config.shared_prefix = shared_prefix_config
        
        # Initialize generator
        generator = SharedPrefixDataGenerator(self.api_config, self.config, self.mock_tokenizer_wrapper)
        
        # Verify generated prompts
        self.assertTrue(len(generator.prompts) > 0)
        
        for prompt in generator.prompts:
            # Re-tokenize to check length
            # Now prompt should be list[int]
            self.assertIsInstance(prompt, list)
            tokens = prompt
            
            # Expected length = system_prompt_len + question_len
            expected_len = 10 + 20
            
            self.assertEqual(len(tokens), expected_len, 
                f"Prompt length mismatch: expected {expected_len}, got {len(tokens)}")

    def test_multi_turn_chat_initialization(self):
        shared_prefix_config = SharedPrefix(
            num_groups=1,
            num_prompts_per_group=10,
            system_prompt_len=10,
            question_len=20,
            output_len=10,
            enable_multi_turn_chat=True
        )
        self.config.shared_prefix = shared_prefix_config
        
        generator = SharedPrefixDataGenerator(self.api_config, self.config, self.mock_tokenizer_wrapper)
        
        self.assertTrue(len(generator.prompts) > 0)
        self.assertTrue(len(generator.user_sessions) > 0)
        
        # Verify prompts are list[int] (questions)
        for prompt in generator.prompts:
            self.assertIsInstance(prompt, list)
            self.assertEqual(len(prompt), 20) # question_len
            
        # Verify user sessions have list[int] context (shared prefix)
        for session in generator.user_sessions:
            self.assertIsInstance(session.contexts, list)
            self.assertEqual(len(session.contexts), 10) # system_prompt_len

if __name__ == '__main__':
    unittest.main()
