# ChromaDB Corruption Recovery Guide

## Problem
You're seeing errors like:
```
thread '<unnamed>' panicked at rust/sqlite/src/db.rs:157:42:
range start index 10 out of range for slice of length 9
pyo3_runtime.PanicException: range start index 10 out of range for slice of length 9
Could not connect to tenant default_tenant. Are you sure it exists?
```

## Root Causes
1. **Abrupt server shutdown** - Server crashed while ChromaDB was writing
2. **Disk space issues** - Ran out of disk space during write operations
3. **Concurrent access** - Multiple processes accessing the same database
4. **File permissions** - Incorrect permissions on chroma_db directory

## Quick Fix (AWS Server)

### Option 1: Automatic Recovery (Recommended)
The code now includes automatic recovery. Just restart your server:

```bash
# SSH into AWS server
ssh your-aws-server

# Navigate to project
cd /var/www/voicebot-polly

# Stop server
pm2 stop server

# Start server (it will auto-recover if corruption detected)
pm2 start server
pm2 logs server
```

### Option 2: Manual Recovery Script
If automatic recovery doesn't work, use the recovery script:

```bash
# SSH into AWS server
ssh your-aws-server

# Navigate to project
cd /var/www/voicebot-polly

# Stop server
pm2 stop server

# Run recovery script
python scripts/recover_chromadb.py

# Restart server
pm2 restart server
```

### Option 3: Manual Steps
If scripts don't work:

```bash
# SSH into AWS server
ssh your-aws-server

# Navigate to project
cd /var/www/voicebot-polly

# Stop server
pm2 stop server

# Backup corrupted database
mv chroma_db chroma_db.backup.$(date +%Y%m%d_%H%M%S)

# Recreate database
mkdir -p chroma_db

# Reinitialize with data
python scripts/initialise_data.py

# Restart server
pm2 restart server

# Monitor logs
pm2 logs server
```

## Prevention

### 1. Graceful Shutdown
Always stop the server gracefully:
```bash
# Good ✅
pm2 stop server

# Bad ❌
kill -9 <pid>
```

### 2. Monitor Disk Space
```bash
# Check disk space regularly
df -h

# Set up alerts when disk usage > 80%
```

### 3. Single Process
Ensure only one server process is running:
```bash
# Check for duplicate processes
pm2 list
ps aux | grep python

# Kill duplicates if found
pm2 delete all
pm2 start ecosystem.config.js
```

### 4. File Permissions
Ensure correct permissions:
```bash
# Set correct ownership
sudo chown -R $USER:$USER /var/www/voicebot-polly/chroma_db

# Set correct permissions
chmod -R 755 /var/www/voicebot-polly/chroma_db
```

### 5. Regular Backups
Set up a cron job to backup the database:
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * cd /var/www/voicebot-polly && tar -czf chroma_db.backup.$(date +\%Y\%m\%d).tar.gz chroma_db/
```

## Monitoring

Check server health:
```bash
# View logs
pm2 logs server --lines 100

# Check status
pm2 status

# Monitor resources
pm2 monit
```

## Troubleshooting

### Issue: Auto-recovery not working
**Solution**: Run manual recovery script:
```bash
python scripts/recover_chromadb.py
```

### Issue: Permission denied
**Solution**: Fix permissions:
```bash
sudo chown -R $USER:$USER /var/www/voicebot-polly
```

### Issue: Out of disk space
**Solution**: Clean up old backups:
```bash
# Check disk usage
df -h

# Remove old backups (keep last 3)
cd /var/www/voicebot-polly
ls -t chroma_db.backup.* | tail -n +4 | xargs rm -rf
```

### Issue: Data not loading
**Solution**: Verify data files exist:
```bash
ls -la data/
python scripts/initialise_data.py
```

## Support

If issues persist:
1. Check PM2 logs: `pm2 logs server --err`
2. Check system logs: `journalctl -u pm2-$USER`
3. Verify Python environment: `which python` and `python --version`
4. Test ChromaDB separately: `python -c "import chromadb; print('OK')"`

## Changes Made

The code now includes:
1. **Auto-recovery**: Automatically detects and recovers from corruption
2. **Backup creation**: Corrupted databases are backed up before deletion
3. **Better error handling**: More descriptive error messages
4. **Recovery script**: Standalone script for manual recovery
