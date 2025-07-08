import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(model_name, device_map="auto"):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        logger.info(f"Loaded model and processor: {model_name}")
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model or processor: {e}")
        raise

def allocate_gpus(num_gpus):
    if torch.cuda.is_available() and num_gpus > 0:
        devices = [f'cuda:{i}' for i in range(num_gpus)]
        logger.info(f"Allocated GPUs: {devices}")
        return devices
    else:
        logger.error("No available GPUs found.")
        raise RuntimeError("No available GPUs found.")

def process_inputs(conversation, processor):
    try:
        inputs = processor.apply_chat_template(
            conversation,
            load_audio_from_video=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            video_fps=1,
            padding=True,
            use_audio_in_video=True,
        )
        logger.info("Processed inputs successfully.")
        return inputs
    except Exception as e:
        logger.error(f"Error processing inputs: {e}")
        raise

def decode_outputs(text_ids, processor):
    try:
        # If model.generate returns a tuple, take the first element (token ids)
        if isinstance(text_ids, tuple):
            text_ids = text_ids[0]
        # Ensure text_ids is on CPU and is a list of lists
        if hasattr(text_ids, "cpu"):
            text_ids = text_ids.cpu().tolist()
        elif isinstance(text_ids, torch.Tensor):
            text_ids = text_ids.tolist()
        decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.info("Decoded outputs successfully.")
        return decoded
    except Exception as e:
        logger.error(f"Error decoding outputs: {e}")
        raise