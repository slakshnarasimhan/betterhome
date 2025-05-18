import boto3
import os
from botocore.exceptions import ClientError

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'betterhome-recommendation'
        
    def upload_file(self, file_path, s3_key):
        """Upload a file to S3 bucket"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            return True
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return False
            
    def get_file_url(self, s3_key):
        """Generate a presigned URL for the S3 object"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=3600  # URL expires in 1 hour
            )
            return url
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            return None 