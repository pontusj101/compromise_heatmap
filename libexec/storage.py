import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

application_default_credentials = "auth.json"


class GcsStorageBucket:
    def __init__(self, bucket_name, credentials_path=None):
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = storage.Client(credentials=credentials)
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth.json"
            self.client = storage.Client()

        self.bucket = self.client.bucket(bucket_name)

    def upload(self, local_path: Path, online_path: Path):
        blob = self.bucket.blob(online_path.as_posix())
        blob.upload_from_filename(local_path.as_posix())

    def download(self, online_path: Path, local_path: Path):
        blob = self.bucket.blob(online_path.as_posix())
        blob.download_to_filename(local_path.as_posix())

    def list_files(self, prefix: Path):
        return [blob.name for blob in self.bucket.list_blobs(prefix=prefix)]
