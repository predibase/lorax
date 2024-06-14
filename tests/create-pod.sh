IMAGE_NAME="$1"

runpodctl create pods \
  --name lorax-tests-new \
  --gpuType "NVIDIA A40" \
  --imageName "$IMAGE_NAME" \
  --containerDiskSize 100 \
  --volumeSize 100 \
  --ports "8080/http" \
  --args "--port 8080 --model-id predibase/Mistral-7B-v0.1-dequantized --adapter-source hub --default-adapter-source pbase --max-batch-prefill-tokens 32768 --max-total-tokens 8192 --max-input-length 8191 --max-concurrent-requests 1024" | awk '{print $2}' > pod_name.txt