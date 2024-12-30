class S3DownloadError(Exception):
    def __init__(self, message, s3_path):
        self.message = message
        self.s3_path = s3_path
        super().__init__(self.message)


class UnzipError(Exception):
    def __init__(self, message, zip_path):
        self.message = message
        self.zip_path = zip_path
        super().__init__(self.message)


class S3ListError(Exception):
    def __init__(self, message, s3_bucket, s3_base_object_path):
        self.message = message
        self.s3_bucket = s3_bucket
        self.s3_base_object_path = s3_base_object_path
        super().__init__(self.message)
        

class DataValidationError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        

class CudaError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)        