import io
import torch
import json
from google.cloud import storage

class BucketManager:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.bucket = storage.Client().bucket(bucket_name)

    def torch_load_from_bucket(self, blob_name):
        blob = self.bucket.blob(blob_name)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        data = torch.load(buffer)
        buffer.close()
        return data

    def torch_save_to_bucket(self, data, blob_name):
        buffer = io.BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(buffer)
        buffer.close()

    def json_load_from_bucket(self, blob_name):
        blob = self.bucket.blob(blob_name)
        json_data = blob.download_as_string()
        return json.loads(json_data.decode('utf-8'))
    
    def json_save_to_bucket(self, data, blob_name):
        json_data = json.dumps(data)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json_data)