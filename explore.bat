@echo off
REM Open the knowledge explorer for the repository in this directory.
REM Falls back to `python -m icn.cli` when the console script is not on PATH.
setlocal
where icn-explore >nul 2>&1
if %errorlevel%==0 (
  icn-explore %*
) else (
  python -m icn.cli %*
)
endlocal
