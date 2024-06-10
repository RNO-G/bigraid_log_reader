import numpy as np

def preprocess(df):
    # Filter the data slightly, since the drill can sometimes record spurious bad values
    df = df.rolling(3).median()

    # Do some cleanup
    df[df["[PLC]DRILLACTIVECURRENT"].abs() > 1e10] = np.nan
    df.loc[df["[PLC]WIRESPOOLEDOUT"].abs() > 200, "[PLC]WIRESPOOLEDOUT"] = np.nan
    df["[PLC]WIRESPOOLEDOUT"] = df["[PLC]WIRESPOOLEDOUT"].ffill()
    df[df["[PLC]DRILLFEEDBACKVEL"].abs() > 100] = np.nan

    df = df.dropna()

    # Mark drill runs
    spooled_out = df["[PLC]WIRESPOOLEDOUT"] > 1.5
    df['run'] = 1.0
    df.loc[spooled_out.gt(spooled_out.shift()), 'run'] = 0
    run_ids = df['run'].ne(df['run'].shift()).cumsum() - 1
    in_run = df["run"] != 0.0
    df.loc[in_run, "run"] = run_ids.loc[in_run] / 2
    df.loc[df['run'] == 0.0, 'run'] = df['run'].shift(-1)
    df[df['run'] == 0.0] = 1.0

    # Add a column that indicates if active cutting of new ice was happening. Kind of a guess based on other parameters
    df["cutting"] = 0
    df.loc[(df["[PLC]DRILLACTIVECURRENT"] > 0.5) & (df["run"] > 0) & (df["[PLC]WIRESPOOLEDOUT"] > 1) & (df["[PLC]CABLESPEED"].abs() < 1), "cutting"] = 1
    # Filter out cutting state where the payout is not relatively close to the maximum
    group_max_out = df["[PLC]WIRESPOOLEDOUT"].groupby(df["run"]).transform("max")
    df.loc[((group_max_out / df["[PLC]WIRESPOOLEDOUT"]).fillna(0) > 10), "cutting"] = 0

    # Calculate the (running) cut depth drilled per run. For each run, this starts at 0 when drilling begins at
    # the bottom of the hole, and increases until drilling stops
    df["cut_depth"] = df[df["cutting"] == 1]["[PLC]WIRESPOOLEDOUT"].groupby(df["run"]).transform("first")
    df["cut_depth"] =  df["[PLC]WIRESPOOLEDOUT"] - df["cut_depth"]
    df["cut_depth"] = df['cut_depth'].groupby(df["run"]).transform("cummax")
    df["cut_depth"] = df["cut_depth"].fillna(0)

    # Calculate the weight-on-bit
    # Assumes cable weight of 159g/m and ice density of 0.91g/mL
    df["weight_on_bit"] = (-1 * df["[PLC]CABLETENSION"] + df["[PLC]WIRESPOOLEDOUT"] * 0.159 + 280 + ((0.285/2)**2 * np.pi * df['cut_depth'] * 910))

    # Normalize pitch and roll using the yaw angle. This should give fixed pitch and roll wrt. the hole
    # coordinate system, although there is no way to tell which direction pitch and roll are actually pointing in
    yaw_cos =  np.cos(np.deg2rad(df['[PLC]IMUYAW']))
    yaw_sin = np.sin(np.deg2rad(df['[PLC]IMUYAW']))
    df['hole_pitch'] = df['[PLC]IMUPITCH'] * yaw_cos - df['[PLC]IMUROLL'] * yaw_sin
    df['hole_roll'] = df['[PLC]IMUPITCH'] * yaw_sin + df['[PLC]IMUROLL'] * yaw_cos

    return df