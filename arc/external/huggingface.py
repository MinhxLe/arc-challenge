from datasets import DatasetDict
from arc import settings


def upload_dataset(repo_id: str, dataset: DatasetDict) -> None:
    dataset.push_to_hub(
        repo_id,
        token=settings.HF_API_TOKEN,
        private=True,
    )
