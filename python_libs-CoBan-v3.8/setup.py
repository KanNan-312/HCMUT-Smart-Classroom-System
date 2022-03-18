import sys
import subprocess

proc = subprocess.Popen([sys.executable, "processing.py"], stderr=subprocess.PIPE)
result,err = proc.communicate()
exit_code = proc.wait()

print("==================================")
if str(err).find("ERROR:") > -1 or exit_code == 1:
    print("Failed to install packages.")
else:
    print("Successfully installed packages!!!")

