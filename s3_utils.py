import boto3

s3 = boto3.client('s3')

def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    s3.upload_file(file_name, bucket, object_name)
    print("Upload Successful")


def download_file(bucket, object_name, file_name):
    s3.download_file(bucket, object_name, file_name)
    print("Download Successful")