"""Exercise the real Qwen3.5 implementation with tiny random weights."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mlx') is not None,'Use .venv-generative/bin/python for MLX tests')
class GenerativeModelTests(unittest.TestCase):
    def test_gradients_frozen_backbone_no_click_and_adapter_roundtrip(self):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen3_5 import Model,ModelArgs
        from generative_model import GenerativeCTR
        from generative import save_trainable
        class Tokenizer:
            def encode(self,text,**kwargs):return [ord(text)-ord('A')+16] if text in 'ABCDEFGHIJK' else [3]
        config=dict(lora_layers=4,lora_rank=2,lora_scale=4.,gradient_checkpointing=False,label_tokens=list('ABCDEFGHIJK'))
        def make():
            mx.random.seed(7)
            tc=dict(model_type='qwen3_5_text',hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                    num_attention_heads=2,num_key_value_heads=1,head_dim=16,vocab_size=128,
                    linear_num_key_heads=1,linear_num_value_heads=1,linear_key_head_dim=128,
                    linear_value_head_dim=128,tie_word_embeddings=True)
            return GenerativeCTR(Model(ModelArgs(model_type='qwen3_5',text_config=tc)),Tokenizer(),
                dict(event=3,history=4,candidate=5),config,np.ones(11,dtype=np.float32)/11)
        model=make();rng=np.random.default_rng(8)
        inputs=[mx.array(rng.normal(size=s).astype(np.float32)) for s in [(2,3),(2,20,4),(2,10,5)]]
        layer=model.backbone.layers[0].linear_attn.in_proj_qkv.linear
        before=mx.array(layer.weight);mx.eval(before)
        model.train()
        loss,grads=nn.value_and_grad(model,lambda m:nn.losses.cross_entropy(m(*inputs),mx.array([10,2]),reduction='mean'))(model)
        mx.eval(loss,grads)
        flattened=dict(tree_flatten(grads))
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreater(float(mx.linalg.norm(flattened['candidate_projection.weight'])),0)
        self.assertTrue(any(float(mx.linalg.norm(g))>0 for name,g in flattened.items() if name.endswith('lora_b')))
        optim.SGD(.001).update(model,grads);mx.eval(model.trainable_parameters())
        np.testing.assert_array_equal(np.asarray(before),np.asarray(layer.weight))
        model.eval();expected=np.asarray(model.probabilities(*inputs))
        self.assertEqual(expected.shape,(2,10));self.assertTrue((expected.sum(1)<1).all())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'adapters.safetensors';save_trainable(model,path)
            reloaded=make();reloaded.load_weights(str(path),strict=False);reloaded.eval()
            np.testing.assert_allclose(np.asarray(reloaded.probabilities(*inputs)),expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
