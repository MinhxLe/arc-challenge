set dotenv-load

ssh:
  ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_IP

sync:
    rsync -avz \
    --include=".env" \
    --exclude-from=.gitignore \
    --exclude=.git \
    -e "ssh -p ${REMOTE_PORT}" \
    . ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PROJECT_DIR}
