# avatar-creator

## Workflow

- App init: download and load all models (dwpose, face_alignment, face_landmark, vae, etc). Most of them are downloaded from huggingface.
- User upload a video, then app will:
  - extract frames from video
  - detect faces
  - detect landmarks
  - generate masks
  - save preprocessed data to s3
  - generate avatar default video and then upload to s3

