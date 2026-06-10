#!/bin/bash
# Run this BEFORE starting any job on the shared server.
echo "=========================================="
echo "Current GPU usage (make sure it's free):"
echo "=========================================="
nvidia-smi
echo ""
echo "=========================================="
echo "Users currently logged in:"
echo "=========================================="
who
echo ""
echo "If the GPU shows significant memory in use by another process,"
echo "DO NOT start your job. Wait or coordinate with the other user."
