# Operations and tooling

Installation/update/uninstall and HTTPS helpers, diagnostics and benchmark tooling. Source checks and module imports follow the package paths under `src/hailo_services/`.

## Updating an existing native/systemd installation

Run the updater from the repository checkout:

```bash
sudo bash scripts/update.sh
```

A separate `git pull` is no longer required. When the source directory is a Git checkout, `update.sh` first pulls the configured upstream of the current branch with `git pull --ff-only`, then reinstalls the application from that checkout, updates the systemd unit, verifies selected installed modules, restarts the service and checks that it is active.

The fast-forward-only pull never creates an implicit merge or rewrites local commits. Existing non-conflicting working-tree changes are kept and are installed as part of the update; conflicting local changes cause Git to abort before the installed service is modified. A checkout without Git metadata can still be installed, but repository synchronization is skipped. If the pull updates `scripts/update.sh` itself, the updater restarts once and continues with the freshly pulled script.
