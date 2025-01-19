import io
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import boto3
import io
from botocore.exceptions import ClientError
from keras.api.models import save_model, load_model
import tempfile
import h5py


class WasabiClient:
    def __init__(self):
        access_key = os.getenv("WASABI_ACCESS_KEY")
        secret_key = os.getenv("WASABI_SECRET_KEY")
        self.bucket = os.getenv("WASABI_BUCKET_NAME")
        self.s3_client = boto3.client(
            's3',
            endpoint_url='https://s3.eu-central-2.wasabisys.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def fetch_data(self, key: str):
        cloud_key = f"Stock_Data/{key}.csv"

        try:
            file_response = self.s3_client.get_object(Bucket=self.bucket, Key=cloud_key)
            file_content = file_response['Body'].read().decode('utf-8')  # Fix typo
            return pd.read_csv(io.StringIO(file_content))
        except ClientError as e:
            print(f"Error fetching data: {e}")
            return None

    def save_model_to_cloud(self, code: str, model):
        cloud_key = f"Models/{code}.keras"

        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as temp_file:
                # Save the model to the temporary file
                model.save(temp_file.name)
                temp_file.seek(0)  # Reset file pointer to the beginning

                # Read the content of the temporary file into memory
                model_buffer = temp_file.read()

            # Upload the file content to S3
            self.s3_client.put_object(Bucket=self.bucket,
                                      Key=cloud_key,
                                      Body=model_buffer,
                                      ContentType='application/octet-stream'
                                      )
            print(f"Successfully uploaded model for {code} to {cloud_key}.")
        except ClientError as e:
            print(f"Error uploading model: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def fetch_model(self, code: str):
        cloud_key = f"Models/{code}.keras"
        try:
            # Fetch the binary model file from S3
            file_response = self.s3_client.get_object(Bucket=self.bucket, Key=cloud_key)
            model_content = io.BytesIO(file_response['Body'].read())  # Read content into BytesIO

            # Save the model content to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".keras") as tmp_file:
                tmp_file.write(model_content.getvalue())  # Write BytesIO content to temp file
                tmp_file.flush()  # Ensure content is written

                # Load the model from the temporary file
                model = load_model(tmp_file.name)

            print(f"Successfully loaded model for {code}.")
            return model
        except ClientError as e:
            print(f"Error fetching model: {e}")
            return None


def get_wassabi_client():
    global wassabi_client
    if wassabi_client is None:
        raise Exception("WasabiClient is not initialized. Check BackendConfig or initialization logic.")
    return wassabi_client


# Setter for initialization
def set_wassabi_client(client):
    global wassabi_client
    wassabi_client = client


def initialize_wasabi_client():
    return WasabiClient()


wassabi_client = None
