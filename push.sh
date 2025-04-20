#!/bin/bash
# push.sh - A script to add, commit, and push changes to the main branch

# Check for any local changes before pushing
echo "Checking for local changes..."
git status

# Add all changes to staging
echo "Staging all changes..."
git add .

# Commit changes with a default message if no message is provided as an argument
commit_message=${1:-"demos"}
echo "Committing changes with message: '$commit_message'"
git commit -m "$commit_message"

# Push changes to the main branch
echo "Pushing changes to origin/main..."
git push -u origin main

echo "Push completed!"
