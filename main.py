from matplotlib import pyplot as plt

from log_reader import LogReader

if __name__ == "__main__":
    reader = LogReader(tagfile="./DataLog/2024 02 20 0000 BigRAID (Tagname).DAT")
    df = reader.as_df()
    df[['[PLC]DRILLACTIVECURRENT', '[PLC]DRILLFEEDBACKVEL']].plot()
    plt.show()
