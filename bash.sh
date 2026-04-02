#!/bin/bash

# 1. Define the line we want to add
# Using $(pwd) makes it dynamic so it works on any folder the script is in
VAR_LINE="export PYTHONPATH=\$PYTHONPATH:$(pwd)"

# 2. Determine which config file to use (.bashrc for Linux, .zshrc for Mac)
if [ -f "$HOME/.zshrc" ]; then
    CONF_FILE="$HOME/.zshrc"
else
    CONF_FILE="$HOME/.bashrc"
fi

# 3. Check if the line already exists to avoid cluttering the file
if grep -Fxq "$VAR_LINE" "$CONF_FILE"
then
    echo "Path already exists in $CONF_FILE. Skipping."
else
    echo "Adding PYTHONPATH to $CONF_FILE..."
    echo "$VAR_LINE" >> "$CONF_FILE"
    echo "Success! Please run 'source $CONF_FILE' or restart your terminal."
fi