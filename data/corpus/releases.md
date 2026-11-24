# Release operations

## Production change freeze
The release coordinator blocks routine production deployments from 16:00 UTC every Friday until 08:00 UTC on Monday.

## Staging deploy window
Routine staging deployments are available until 22:00 UTC on Fridays. No release coordinator approval is required.

## Canary traffic
New production builds initially receive 5 percent of customer traffic. The release controller then evaluates error rates.

## Rollback trigger
The release controller reverts a canary if its error rate exceeds 2 percent for 10 consecutive minutes.

## Emergency release approval
An emergency production deployment requires approval from both the incident commander and the release coordinator.

## Artifact retention
Signed release artifacts are kept for 180 days in the package registry. A scheduled task removes expired artifacts.

## Rollback validation
After a rollback, the release coordinator verifies the last known good version in the deployment dashboard.
