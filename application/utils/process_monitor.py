"""
---------------------------------------------------------
process_monitor.py

Monitors the CPU usage of the Ollama inference engine
(llama-server.exe / ollama.exe) while a response is
being generated.

Tracks the Peak CPU utilization observed during
response generation.
---------------------------------------------------------
"""

import threading
import psutil


class ProcessMonitor:

    _peak_cpu = 0.0
    _running = False
    _thread = None

    @staticmethod
    def _get_llama_process():
        """
        Locate the Ollama inference process.
        """

        for proc in psutil.process_iter(["pid", "name"]):

            try:

                name = (proc.info["name"] or "").lower()

                if "llama-server" in name:
                    print(
                        f"[CPU Monitor] Monitoring process: "
                        f"{proc.info['name']} (PID {proc.pid})"
                    )

                    return psutil.Process(proc.pid)

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess
            ):
                continue

        print("[CPU Monitor] llama-server.exe is not running.")

        return None

    @classmethod
    def _monitor(cls):

        process = cls._get_llama_process()

        if process is None:
            print("[CPU Monitor] Ollama process not found.")
            return

        print(f"[CPU Monitor] Monitoring PID: {process.pid}")

        # Initialize CPU counter
        process.cpu_percent(None)

        while cls._running:

            try:

                cpu = process.cpu_percent(interval=0.5)

                if cpu > cls._peak_cpu:
                    cls._peak_cpu = cpu

                print(
                    f"[CPU Monitor] Sample = {cpu:.2f}% | "
                    f"Peak = {cls._peak_cpu:.2f}%"
                )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied
            ):
                break

    @classmethod
    def start(cls):

        cls._peak_cpu = 0.0
        cls._running = True

        cls._thread = threading.Thread(
            target=cls._monitor,
            daemon=True
        )

        cls._thread.start()

    @classmethod
    def stop(cls):

        cls._running = False

        if cls._thread:
            cls._thread.join()

    @classmethod
    def get_peak_cpu(cls):

        # Optional debug print
        # print(f"[CPU Monitor] Peak CPU = {cls._peak_cpu:.2f}%")

        logical_cpus = psutil.cpu_count(logical=True)


        return round(cls._peak_cpu / logical_cpus, 2)