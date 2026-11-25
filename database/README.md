# Database workspace

Database scripts and versioned schema changes will live here.

- `migrations/`: ordered schema migrations.
- `seeds/`: development seed data.

The database engine, entity relationships, constraints and migration tooling will be implemented in the database-design phase. Do not store uploaded files or model binaries inside the database; use file/object storage and keep references in database records.
