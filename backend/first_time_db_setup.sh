#!/bin/bash

# Set the PostgreSQL user to run the commands.  Adjust if needed.
PGUSER="postgres"

# Set the username and password for the new user. IMPORTANT: Use strong passwords!
DB_USER="postgres" # Or whatever username you want
DB_PASS="postgres" # Or a secure password

# Set the database name
DB_NAME="mydatabase"

# Function to execute psql commands
psql_cmd() {
  sudo -u "$PGUSER" psql -v ON_ERROR_STOP=1 <<EOSQL
$1
EOSQL
}

# Check if the database exists
DB_EXISTS=$(sudo -u "$PGUSER" psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")

if [ -z "$DB_EXISTS" ]; then
  # Create the user and set the password with superuser status.
  psql_cmd "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS' SUPERUSER;"

  # Create the database and set the owner.
  psql_cmd "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

  # Grant all privileges on the database to the user.
  psql_cmd "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

  # Grant privileges on the public schema to the user.
  psql_cmd "GRANT USAGE, CREATE ON SCHEMA public TO $DB_USER;"

  echo "PostgreSQL user and database created."
else
  echo "PostgreSQL database '$DB_NAME' already exists."
fi

echo "PostgreSQL setup complete."
