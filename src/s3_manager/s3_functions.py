import os
import boto3
import botocore.exceptions
from . import ACCESS_KEY, SECRET_KEY, REGION, BUCKETNAME

def upload_file_s3(file_path, s3_path):
    try:
        # Create Client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=REGION
        )

        # Create s3 location
        filename = file_path.split("/")[-1]
        s3_key = os.path.join(s3_path, filename)

        # Upload file
        if filename.split(".")[-1] == "html":
            s3_client.upload_file(file_path, BUCKETNAME, s3_key, ExtraArgs={'ContentType': 'text/html'})
        else:
            s3_client.upload_file(file_path, BUCKETNAME, s3_key)

    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontradp {e}")

    except botocore.exceptions.NoCredentialsError:
        print("❌ Error: No se encontraron credenciales de AWS. Verifica tus variables de entorno.")

    except botocore.exceptions.PartialCredentialsError:
        print("❌ Error: Las credenciales de AWS están incompletas. Verifica que AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY estén configuradas.")

    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ Error del cliente S3 ({error_code}): {e.response['Error']['Message']}")

    except Exception as e:
        print(f"❌ Error inesperado: {e}")


def download_file_s3(file_name, s3_path, download_path):
    try:
        # Create Client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=REGION
        )

        # Create s3 location
        s3_key = os.path.join(s3_path, file_name)

        # Download file
        s3_client.download_file(BUCKETNAME, s3_key, download_path)

    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontradp {e}")

    except botocore.exceptions.NoCredentialsError:
        print("❌ Error: No se encontraron credenciales de AWS. Verifica tus variables de entorno.")

    except botocore.exceptions.PartialCredentialsError:
        print("❌ Error: Las credenciales de AWS están incompletas. Verifica que AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY estén configuradas.")

    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ Error del cliente S3 ({error_code}): {e.response['Error']['Message']}")

    except Exception as e:
        print(f"❌ Error inesperado: {e}")        