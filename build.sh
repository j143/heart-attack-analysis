
PROJECT_ID="ai-experiments-001"

IMAGE_URI="gcr.io/$PROJECT_ID/mpg:v1"

docker build ./ -t $IMAGE_URI

docker run $IMAGE_URI