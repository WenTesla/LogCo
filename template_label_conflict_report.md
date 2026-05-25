# Template Label Conflict Report

This report records whether the same log template can appear with both normal
and anomaly labels in the current local datasets.

## Summary

### Single `EventTemplate` Conflicts In Structured Logs

| Dataset | Rows | Templates | Anomaly rows | Conflict templates | Conflict rows | Anomaly rows in conflict |
|---|---:|---:|---:|---:|---:|---:|
| BGL | 4,713,493 | 1,847 | 944,896 | 9 (0.49%) | 89,237 (1.89%) | 69,004 (7.30%) |
| Spirit | 5,000,000 | 2,880 | 764,891 | 4 (0.14%) | 679 (0.01%) | 70 (0.01%) |
| Thunderbird | 9,959,160 | 4,992 | 4,934 | 2 (0.04%) | 815 (0.01%) | 803 (16.27%) |

`HDFS` structured data was not included in this single-template conflict check
because the loaded columns did not include both usable `EventTemplate` and
`Label` fields for the same check.

### Grouped `Templates` Representation Conflicts

| Dataset | Windows | Unique template sets | Anomaly windows | Conflict template sets | Conflict windows | Anomaly windows in conflict |
|---|---:|---:|---:|---:|---:|---:|
| BGL | 78,558 | 3,358 | 19,015 | 3 (0.09%) | 88 (0.11%) | 46 (0.24%) |
| Spirit | 83,333 | 22,455 | 30,215 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

## Notable Conflict Examples

### BGL

- Template: `data storage interrupt`
  - Count: 63,493
  - Anomaly rows: 63,491
  - Normal example: `data storage interrupt`
  - Anomaly example: `data storage interrupt`

- Template: `ciod: Error creating node map from file <*> <*> <*> <*>`
  - Count: 5,464
  - Anomaly rows: 2,512
  - Normal example: `ciod: Error creating node map from file /p/gb2/welcome3/32k_128x256x1_8x4x4.map: Bad file descriptor`
  - Anomaly example: `ciod: Error creating node map from file /p/gb2/cabot/miranda/newmaps/8k_128x64x1_8x4x4.map: No child processes`

- Template: `Microloader Assertion`
  - Count: 1,504
  - Anomaly rows: 1,503
  - Normal example: `Microloader Assertion`
  - Anomaly example: `Microloader Assertion`

- Template: `ciod: LOGIN <*> failed: <*> <*>`
  - Count: 1,288
  - Anomaly rows: 816
  - Normal example: `ciod: LOGIN chdir(/home/bertsch2/src/bgl_hello) failed: Permission denied`
  - Anomaly example: `ciod: LOGIN chdir(/p/gb1/stella/RAPTOR/2183) failed: Input/output error`

### Spirit

- Template: `netfs: Mounting NFS filesystems: <*>`
  - Count: 575
  - Anomaly rows: 62
  - Normal example: `netfs: Mounting NFS filesystems: succeeded`
  - Anomaly example: `netfs: Mounting NFS filesystems: failed`

- Template: `<*> authentication failure; <*> <*> euid=0 tty= <*> rhost=`
  - Count: 11
  - Anomaly rows: 5
  - Normal example: `chsh(pam_unix)[5473]: authentication failure; logname=#208# uid=1691 euid=0 tty= ruser= rhost=`
  - Anomaly example: `su(pam_unix)[12677]: authentication failure; logname=#99# uid=46912 euid=0 tty= ruser=#99# rhost=`

- Template: `kernel: Out of Memory: Killed process <*> <*>`
  - Count: 90
  - Anomaly rows: 2
  - Normal example: `kernel: Out of Memory: Killed process 24592 (gmond).`
  - Anomaly example: `kernel: Out of Memory: Killed process 28427 (tcsh).`

### Thunderbird

- Template: `mptscsih: ioc0: attempting <*> <*> <*>`
  - Count: 771
  - Anomaly rows: 760
  - Normal example: `mptscsih: ioc0: attempting bus reset! (sc=000001009e4cc500)`
  - Anomaly example: `mptscsih: ioc0: attempting task abort! (sc=000001017c83fd00)`

- Template: `Losing some ticks... checking if CPU frequency changed.`
  - Count: 44
  - Anomaly rows: 43
  - Normal example: `Losing some ticks... checking if CPU frequency changed.`
  - Anomaly example: `Losing some ticks... checking if CPU frequency changed.`

## Interpretation

- The sequence label rule, "one anomalous log makes the whole sequence
  anomalous", is internally consistent.
- The risk comes from using only `Templates` as model input: if normal and
  anomalous single logs share the same template, anomaly evidence may live in
  parameters or raw/regex-normalized content rather than in the template.
- In current grouped data, this risk is minimal for `Spirit` and small for
  `BGL` at the window-representation level.
- At the single-log level, `BGL` still has a non-trivial amount of anomaly rows
  inside mixed-label templates, and `Thunderbird` may be sensitive if grouped
  later using template-only inputs.
