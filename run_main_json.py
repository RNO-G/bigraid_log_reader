import json
import subprocess
import os
from os.path import isfile, join
from main import plot
from matplotlib import pyplot as plt
# Load values from JSON
with open("2025-Drill-Log.json") as f:
    data = json.load(f)

#folder = r"datalog/"
folder = r"/Users/delia/Library/CloudStorage/OneDrive-SharedLibraries-NERC/BAS BigRAID - Documents/Season Reports/2025/DataLog/" 

folder = os.path.realpath(folder)
print(folder)
onlyfiles = [f for f in os.listdir(folder) if isfile(join(folder, f))]
#print(onlyfiles)

for i,d in enumerate(data):
    print(d)
    filename = f"2025-Plots/%02i_Site%02i_%02i_%s.pdf" % (d["number"], d["site"], d["hole"], d["geoloc"])
    print(filename)
    #filename2 = f"2025-Plots/{i}_Site{d['site']}_{d['hole']}_{d['geoloc']}.pdf"
    #print(filename, filename2)
    plot.callback(folder, d["start"], d["end"], filename)
    plt.close()

#for d in data[0:1]:
#    print(d)
#    filename = "2025-Plots/%02i_Site%02i_%02i_%s.pdf" % (d["number"], d["site"], d["hole"], d["geoloc"])
#    print(filename)
#    subprocess_string = "python3 main.py --folder %s  --start %s --end %s --out %s &" %(folder,d["start"],d["end"],filename) 
#    print("%s" % subprocess_string)
#    # Run main.py with these args
#    #subprocess.run(["python3", subprocess_string])
#    subprocess.run([subprocess_string])

