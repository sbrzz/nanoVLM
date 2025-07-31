import onnxruntime as ort
import numpy as np
from loguru import logger

# Load the ONNX model
session = ort.InferenceSession(r"onnx_export/decoder.onnx",
                               providers=["CPUExecutionProvider"])

logger.info("Prefill sanity check")
logger.info("---------------------")

decoder_input = np.load("checkpoints/initial_combined_embeds.npy")
start_pos = np.array(0).astype(np.int64)

# prefill can be simulated by using a zero dimension
past_k = np.random.randn(1, 1, 0, 36).astype(np.float32)
past_v = np.random.randn(1, 1, 0, 36).astype(np.float32)

# Prepare input dict
inputs = {
    "decoder_input": decoder_input,
    "past_key_0": past_k,
    "past_value_0": past_v,
    "decoder_start_pos": start_pos
}

# Run inference
outputs = session.run(None, inputs)

# Parse outputs and check validity
prefill_outputs_reference = np.load("checkpoints/prefill_output.npy")
prefill_outputs = outputs[0]

if not np.allclose(prefill_outputs_reference, prefill_outputs, atol=1e-5):
    raise ValueError("not working correctly")

present_key_0_reference = np.load("checkpoints/present_key_0.npy")
present_key_0 = outputs[1]

if not np.allclose(present_key_0_reference, present_key_0, atol=1e-5):
    raise ValueError("not working correctly")

present_value_0_reference = np.load("checkpoints/present_value_0.npy")
present_value_0 = outputs[2]

if not np.allclose(present_value_0_reference, present_value_0, atol=1e-5):
    raise ValueError("not working correctly")

# Print results
# print("✅ Decoder Output shape:", prefill_outputs.shape)
# print("✅ Present key shape:", present_key_0.shape)
# print("✅ Present value shape:", present_value_0.shape)

logger.success("prefill sanity check passed ---------------------")

logger.info("Decode sanity check")
logger.info("---------------------")

next_token_embed = np.load("checkpoints/decode_phase_next_token_embed_0.npy")
start_pos = np.array(61).astype(np.int64)

# prefill can be simulated by using a zero dimension
past_k = np.load("checkpoints/decode_phase_past_key_0.npy")
past_v = np.load("checkpoints/decode_phase_past_value_0.npy")

# Prepare input dict
inputs = {
    "decoder_input": next_token_embed,
    "past_key_0": past_k,
    "past_value_0": past_v,
    "decoder_start_pos": start_pos
}

# Run inference
outputs = session.run(None, inputs)

# Parse outputs and check validity
output_reference = np.load("checkpoints/decode_phase_decode_step_output_0.npy")
decode_output = outputs[0]

if not np.allclose(output_reference, decode_output, atol=1e-5):
    raise ValueError("not working correctly")

present_key_0_reference = np.load("checkpoints/decode_phase_present_key_0.npy")
present_key_0 = outputs[1]

if not np.allclose(present_key_0_reference, present_key_0, atol=1e-2):
    logger.error("decode phase present_key_0 mismatch")
    raise ValueError

present_value_0_reference = np.load("checkpoints/decode_phase_present_value_0.npy")
present_value_0 = outputs[2]

if not np.allclose(present_value_0_reference, present_value_0, atol=1e-5):
    logger.error("decode phase present_value_0 mismatch")
    raise ValueError

logger.info("Decode sanity check passed ---------------------")