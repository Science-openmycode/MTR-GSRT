param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('inventory','verify','smoke','plot','all-precomputed','generate-main','generate-baselines','evaluate',
               'run-framework','run-privacy','run-profile','run-mr','run-ablation','run-tstr','run-structure')]
  [string]$Stage,
  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$RemainingArgs
)

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($RemainingArgs.Count -gt 0) {
  & python (Join-Path $PackageRoot 'commands\reproduce.py') $Stage -- @RemainingArgs
} else {
  & python (Join-Path $PackageRoot 'commands\reproduce.py') $Stage
}
exit $LASTEXITCODE
