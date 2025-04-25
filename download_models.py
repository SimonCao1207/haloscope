from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    models_to_path = {
        "meta-llama/Llama-3.1-8B-Instruct" : "models/llama-3.1-8B-Instruct",
        "lucadiliello/BLEURT-20" : "models/BLEURT-20",
    }
    for repo_id, local_dir in models_to_path.items():
        download_model(repo_id, local_dir)
