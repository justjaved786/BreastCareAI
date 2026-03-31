#!/usr/bin/env bash

set -o errexit

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🗄️ Initializing SQLite DB..."
python << END
from app import db
db.create_all()
print("Database created successfully!")
END

echo "🚀 Build completed!"