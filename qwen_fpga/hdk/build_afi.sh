#!/usr/bin/env bash
# build_afi.sh -- AWS F2 HDK flow: HLS IP -> CL DCP -> AFI -> (load AGFI).
#
# Run ON the F2 (or an F2-build) instance after cloning aws-fpga and exporting
# the GEMV IP with hdk/run_hls.tcl. This wraps the supported HDK/Vivado path:
# aws_build_dcp_from_cl.py produces the DCP, then create-fpga-image makes the AFI.
#
# Required env:
#   AWS_FPGA_REPO_DIR  path to a cloned github.com/aws/aws-fpga (f2 branch)
#   CL_DIR             custom-logic dir (e.g. .../hdk/cl/examples/cl_qwen_gemv)
#   S3_BUCKET          S3 bucket for the AFI tarball
#   S3_DCP_KEY         S3 prefix for the DCP   (default: cl_qwen_gemv/dcp)
#   S3_LOGS_KEY        S3 prefix for AFI logs  (default: cl_qwen_gemv/logs)
set -euo pipefail

: "${AWS_FPGA_REPO_DIR:?set AWS_FPGA_REPO_DIR to your aws-fpga checkout}"
: "${CL_DIR:?set CL_DIR to the custom-logic directory}"
: "${S3_BUCKET:?set S3_BUCKET for AFI artifacts}"
S3_DCP_KEY="${S3_DCP_KEY:-cl_qwen_gemv/dcp}"
S3_LOGS_KEY="${S3_LOGS_KEY:-cl_qwen_gemv/logs}"

echo "==> sourcing HDK"
cd "$AWS_FPGA_REPO_DIR"
source hdk_setup.sh
source sdk_setup.sh
export CL_DIR

echo "==> building DCP from custom logic (Vivado P&R)"
cd "$CL_DIR/build/scripts"
./aws_build_dcp_from_cl.py -c cl_qwen_gemv --no-encrypt

echo "==> uploading DCP to s3://$S3_BUCKET/$S3_DCP_KEY"
DCP_TAR=$(ls -t "$CL_DIR"/build/checkpoints/to_aws/*.Developer_CL.tar | head -1)
aws s3 cp "$DCP_TAR" "s3://$S3_BUCKET/$S3_DCP_KEY/"

echo "==> creating AFI"
aws ec2 create-fpga-image \
  --name cl_qwen_gemv \
  --description "INT4 HBM-streaming GEMV for Qwen decode" \
  --input-storage-location Bucket="$S3_BUCKET",Key="$S3_DCP_KEY/$(basename "$DCP_TAR")" \
  --logs-storage-location  Bucket="$S3_BUCKET",Key="$S3_LOGS_KEY" \
  | tee afi_ids.json

AGFI=$(python3 -c "import json;print(json.load(open('afi_ids.json'))['FpgaImageGlobalId'])")
AFI=$(python3  -c "import json;print(json.load(open('afi_ids.json'))['FpgaImageId'])")
echo "AFI=$AFI  AGFI=$AGFI"
echo "==> waiting for AFI to become available (can take ~hour)"
until aws ec2 describe-fpga-images --fpga-image-ids "$AFI" \
        --query 'FpgaImages[0].State.Code' --output text | grep -q available; do
  sleep 60; echo "  ...still building"
done

echo "==> AFI available. Load on an F2 instance with:"
echo "    sudo fpga-load-local-image -S 0 -I $AGFI"
echo "    export QWEN_FPGA_BACKEND=xrt"
echo "    export QWEN_FPGA_XCLBIN=<path-to-awsxclbin-or-AGFI-handle>"
