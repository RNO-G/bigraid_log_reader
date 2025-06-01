from pathlib import Path
import struct
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StatusWord:
    ready_to_switch_on: bool
    switched_on: bool
    operation_enabled: bool
    fault: bool
    voltage_enabled: bool
    quick_stop: bool
    switch_on_disabled: bool
    warning: bool
    man_specific: bool
    remote: bool
    target_reached: bool
    internal_limit_active: bool
    operation_mode_specific_1: bool
    operation_mode_specific_2: bool

    @classmethod
    def decode(cls, bytes):
        if isinstance(bytes, int):
            x = bytes
        else:
            x = struct.unpack("<H", bytes)[0]
        return cls(
            ready_to_switch_on=bool((x & 1 << 0)),
            switched_on=bool((x & 1 << 1)),
            operation_enabled=bool((x & 1 << 2)),
            fault=bool((x & 1 << 3)),
            voltage_enabled=bool((x & 1 << 4)),
            quick_stop=bool((x & 1 << 5)),
            switch_on_disabled=bool((x & 1 << 6)),
            warning=bool((x & 1 << 7)),
            man_specific=bool((x & 1 << 8)),
            remote=bool((x & 1 << 9)),
            target_reached=bool((x & 1 << 10)),
            internal_limit_active=bool((x & 1 << 11)),
            operation_mode_specific_1=bool((x & 1 << 12)),
            operation_mode_specific_2=bool((x & 1 << 13))
        )


@dataclass
class PDO:
    command: str
    index: int
    is_float: bool
    is_query: bool
    data: float | int

    @classmethod
    def decode(cls, bytes):
        command, x, data_bytes = struct.unpack("<2sH4s", bytes)
        command = command.decode("utf-8")
        index = x & ~(1 << 15 | 1 << 14)
        is_float = bool(x & (1 << 15))
        is_query = bool(x & (1 << 14))
        if not is_float:
            data = struct.unpack("<i", data_bytes)[0]
        else:
            data = struct.unpack("<f", data_bytes)[0]

        return cls(
            command=command,
            index=index,
            is_float=is_float,
            is_query=is_query,
            data=data
        )

    def annotation(self):
        match self.command, self.is_query:
            case "VX", False:
                return f" ({self.data / 17} RPM)"
            case "SW", False:
                return f" ({StatusWord.decode(self.data)})"
        return ""

    def __str__(self):
        return f"{self.command}[{self.index}] ({'f' if self.is_float else 'i'}, {'q' if self.is_query else 's'}): {self.data}{self.annotation()}"


log_file_path = Path("./2025-05-30_PM.TXT")

with log_file_path.open("r") as log_file:
    for line in log_file.readlines():
        if line.startswith("#"):
            continue

        ts, _, can_id, msg_str = line.split(';')
        try:
            msg_bytes = bytes.fromhex(msg_str)

            d = datetime.strptime(f"{ts}000", "%dT%H%M%S%f")
            ts = f"{d.time()}"[:-3]
        except ValueError:
            continue

        match can_id:
            case "1ff":
                print(f"T-PDO1> {ts}: {StatusWord.decode(msg_bytes)}")
            case "37f":
                pdo = PDO.decode(msg_bytes)
                print(f"R-PDO2> {ts}: {pdo}")
            case "2ff":
                pdo = PDO.decode(msg_bytes)
                print(f"T-PDO2> {ts}: {pdo}")
            case "192" | "292" | "393" | "492" | "712" | "0":
                pass
            case "77F":
                print(f"Heartbeat> {ts}")
            case _:
                print(line.strip())
