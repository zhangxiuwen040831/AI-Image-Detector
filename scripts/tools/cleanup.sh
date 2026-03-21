#!/bin/bash

# Cleanup script for user data

echo "Starting cleanup process..."

# 1. Remove application code and user files in home directory
rm -rf ~/.* ~/* 2>/dev/null

# 2. Clean up data disk
rm -rf /root/lanyun-tmp/* 2>/dev/null

# 3. Clean up log files
find /var/log -type f -name "*.log" -exec truncate -s 0 {} \;

# 4. Remove temporary files
rm -rf /tmp/* /var/tmp/* 2>/dev/null

# 5. Clean package manager caches
apt clean

# 6. Verify cleanup
echo "Cleanup completed. Verifying system state..."
df -h
echo "Home directory contents:"
ls -la ~/
echo "Data disk contents:"
ls -la /root/lanyun-tmp/

echo "Cleanup process finished successfully!"
