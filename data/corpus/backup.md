# Data recovery

## Customer database recovery
The recovery point objective for the primary customer database is 15 minutes. Production snapshots are stored in the secondary region.

## Billing database recovery
The recovery point objective for the billing database is 5 minutes. Finance engineering reviews replication alerts during business hours.

## Staging snapshots
Staging database snapshots are retained for 7 days. They are deleted automatically at the end of the retention window.

## Production snapshots
Production database snapshots are retained for 35 days. The infrastructure team reviews the retention rule every quarter.

## Audit archive
Archived access audit records are retained for 18 months in immutable storage. The security team manages archive access.

## Restore rehearsal
The infrastructure team rehearses a full production restore on the first Tuesday of each month in an isolated account.

## Backup encryption
The backup service encrypts every snapshot using AES-256. A separate key is used for cross-region exports.
