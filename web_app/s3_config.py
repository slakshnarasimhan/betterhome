import boto3
import os
from botocore.exceptions import ClientError

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'betterhome-recommendations'
        
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

    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3 by attempting a HEAD request."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):
                return False
            # Other errors (e.g., permissions) should be surfaced for debugging
            print(f"Error checking existence for {s3_key}: {e}")
            return False

    def list_files(self, prefix: str) -> list[str]:
        """List all object keys under a given prefix."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            keys: list[str] = []
            for page in page_iterator:
                contents = page.get('Contents') or []
                for obj in contents:
                    key = obj.get('Key')
                    if key:
                        keys.append(key)
            return keys
        except ClientError as e:
            print(f"Error listing files under prefix {prefix}: {e}")
            return []
