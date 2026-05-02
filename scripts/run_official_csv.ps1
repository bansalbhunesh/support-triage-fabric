# Official batch with 8-column CSV (issue, subject, company + five graded fields).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Env:SUPPORT_AGENT_CSV_EVAL_MINIMAL = "1"
$in = if ($args[0]) { $args[0] } else { "support_tickets/support_tickets.csv" }
$out = if ($args[1]) { $args[1] } else { "support_tickets/output.csv" }
python code/main.py --quiet --csv $in $out
