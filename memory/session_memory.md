

<tool_call>
<function=read_file>
<parameter=path>
/app/ars.R
</parameter>
</function>
</tool_call>
<tool_call>
<function=run_command>
<parameter=command>
ls -la /app/*.txt 2>/dev/null && head -5 /app/normal_samples.txt 2>/dev/null && head -5 /app/exponential_samples.txt 2>/dev/null
</parameter>
</function>
</tool_call>
<tool_call>
<function=run_command>
<parameter=command>
cd /app && Rscript -e 'assign("interactive", function() TRUE, envir = globalenv()); source("ars.R"); cat("Test completed\n")' 2>&1
</parameter>
</function>
</tool_call>