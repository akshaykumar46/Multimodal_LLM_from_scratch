MODEL_PATH="/Users/akshaykumar/Documents/Projects/vision_language_model/paligemma-3b-pt-224"
PROMPT="caption en "
IMAGE_FILE_PATH="/Users/akshaykumar/Documents/Projects/vision_language_model/story_pic.jpg"
MAX_TOKENS_TO_GENERATE=20
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="True"
ONLY_CPU="False"


python inference.py \
	--model_path "$MODEL_PATH" \
	--prompt "$PROMPT" \
	--image_file_path "$IMAGE_FILE_PATH" \
	--max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
	--temperature "$TEMPERATURE" \
	--top_p "$TOP_P" \
	--do_sample "$DO_SAMPLE" \
	--only_cpu "$ONLY_CPU" \
