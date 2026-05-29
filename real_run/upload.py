from huggingface_hub import login, HfApi

login()

api = HfApi()

api.upload_folder(
    folder_path="./qwen_math_sft/test(5)",
    repo_id="meloncastle/qwen-cse151b",
    repo_type="model",
)