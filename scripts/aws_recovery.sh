#!/bin/bash

# ChromaDB Recovery Script for AWS Server
# Run this on your AWS server when you encounter ChromaDB corruption errors

set -e

echo "=================================================="
echo "🔧 ChromaDB Corruption Recovery Tool"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/var/www/voicebot-polly"
DB_DIR="${PROJECT_DIR}/chroma_db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${DB_DIR}.backup.${TIMESTAMP}"

# Check if running on AWS server
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Error: Project directory not found: $PROJECT_DIR${NC}"
    echo -e "${YELLOW}💡 Are you running this on the AWS server?${NC}"
    exit 1
fi

cd "$PROJECT_DIR"

# Step 1: Stop the server
echo -e "${YELLOW}🛑 Stopping server...${NC}"
pm2 stop server || echo -e "${YELLOW}⚠️  Server may not be running${NC}"
sleep 2

# Step 2: Backup corrupted database
if [ -d "$DB_DIR" ]; then
    echo -e "${YELLOW}📦 Backing up corrupted database...${NC}"
    echo "   From: $DB_DIR"
    echo "   To: $BACKUP_DIR"
    mv "$DB_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}✅ Backup created${NC}"
else
    echo -e "${YELLOW}⚠️  Database directory doesn't exist, skipping backup${NC}"
fi

# Step 3: Create fresh database directory
echo -e "${YELLOW}🔨 Creating fresh database directory...${NC}"
mkdir -p "$DB_DIR"
echo -e "${GREEN}✅ Fresh database directory created${NC}"

# Step 4: Set correct permissions
echo -e "${YELLOW}🔐 Setting correct permissions...${NC}"
chown -R $USER:$USER "$DB_DIR"
chmod -R 755 "$DB_DIR"
echo -e "${GREEN}✅ Permissions set${NC}"

# Step 5: Reinitialize data
echo ""
echo -e "${YELLOW}📝 Do you want to reinitialize the database with data? (y/n)${NC}"
read -r response

if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
    echo -e "${YELLOW}🔄 Reinitializing database...${NC}"
    python scripts/initialise_data.py
    echo -e "${GREEN}✅ Database reinitialized${NC}"
else
    echo -e "${YELLOW}⚠️  Skipping data initialization${NC}"
    echo -e "${YELLOW}💡 Run 'python scripts/initialise_data.py' manually later${NC}"
fi

# Step 6: Restart server
echo ""
echo -e "${YELLOW}🚀 Restarting server...${NC}"
pm2 restart server
sleep 3

# Step 7: Show logs
echo ""
echo -e "${GREEN}✅ Recovery completed!${NC}"
echo ""
echo -e "${YELLOW}📊 Checking server status...${NC}"
pm2 status

echo ""
echo -e "${YELLOW}📝 Recent logs:${NC}"
pm2 logs server --lines 20 --nostream

echo ""
echo "=================================================="
echo -e "${GREEN}✅ Recovery Process Complete!${NC}"
echo "=================================================="
echo ""
echo "💡 Next steps:"
echo "   1. Monitor logs: pm2 logs server"
echo "   2. Test the connection from your browser"
echo "   3. Check for any errors"
echo ""
echo "📁 Backup location: $BACKUP_DIR"
echo ""
