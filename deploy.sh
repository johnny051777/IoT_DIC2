#!/usr/bin/env bash
# Deploy script for CRISP-DM Regression Studio
set -e

echo "=== 1. Checking Git Status ==="
git status

echo "=== 2. Staging files ==="
git add .
git commit -m "feat: complete CRISP-DM regression studio for 50 Startups & Boston Housing with custom CSV support" || true

echo "=== 3. Pushing to GitHub ==="
git branch -M main
git remote add origin https://github.com/johnny051777/IoT_DIC2.git 2>/dev/null || true
git push -u origin main

echo "=== 4. Deploying to Vercel ==="
npx vercel --prod
