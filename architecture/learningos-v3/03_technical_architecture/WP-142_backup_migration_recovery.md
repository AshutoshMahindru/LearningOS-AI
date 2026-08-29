# WP-142: Backup, Migration, and Recovery Architecture

## 1. Automated Backups
- Before applying any SQLite database migration, the system creates a compressed snapshot in `~/.learningos/backups/backup_pre_migration_{timestamp}.sql.gz`.
- Learners can export a portable JSON backup of their entire progress, evidence ledger, and ADR portfolio at any time via the Settings surface or CLI (`learningos export`).

## 2. Migration Framework & Rollback
- Migrations are strictly numbered SQL files executed in single transactions.
- If a migration fails, the transaction rolls back immediately and restores the pre-migration snapshot without loss of user data.
