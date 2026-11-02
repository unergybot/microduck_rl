# HD1910 reference parameter asset

Unchanged Apache-2.0 parameter data from LuwuDynamics/xgoduck_rl revision
326d77a1122870bdefa2c36403937502c958e69c,
src/mjlab_microduck/robot/xgoduck/params/1910_m6.json.
SHA256 ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb.
Retrieved via the attributed local microduck-replica parameter mirror; LICENSE
is redistributed here. This is published reference data, not local identification.
The JSON actuator sts3215 selects BAM's controller implementation. max_velocity
100 disables target slew limiting; it is not a measured free speed. q_offset is
identified model bias, not a hardware joint calibration. Local integration is
independently implemented in workbench/engine.py; no reference studio code copied.
Microduck geometry is retained under its existing upstream asset license.
