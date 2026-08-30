# run_all.py
import os

print("Spúšťam history_worker...")
os.system("python history_worker.py")

print("Spúšťam station_worker...")
os.system("python station_worker.py")
