from log_reader import LogReader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def evaluate():

    logfile = "/home/windischhofer/drill/data_2024/2024_07_08_0000_BigRAID_Tagname.DAT"
    reader = LogReader(tagfile=logfile)
    df = reader.as_df()

    fig = plt.figure(figsize = (13, 5), layout = "constrained")
    fig.savefig("depth.pdf")

if __name__ == "__main__":
    evaluate()
