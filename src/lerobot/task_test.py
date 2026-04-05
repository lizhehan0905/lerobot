
#!/usr/bin/env python

import threading
import time

from pynput import keyboard

current_task_index = [0]


def on_press(key):
    try:
        if key.char == "1":
            current_task_index[0] = 0
            print(f"\n[Monitor] Switched to task 0")
        elif key.char == "2":
            current_task_index[0] = 1
            print(f"\n[Monitor] Switched to task 1")
        elif key.char == "3":
            current_task_index[0] = 2
            print(f"\n[Monitor] Switched to task 2")
    except AttributeError:
        pass


def monitor_keyboard():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def main():
    monitor_thread = threading.Thread(target=monitor_keyboard, daemon=True)
    monitor_thread.start()

    print("主线程开始运行，按下 1/2/3 切换任务")
    counter = 0
    while True:
        print(f"[Main] Task: {current_task_index[0]}, Counter: {counter}")
        counter += 1
        time.sleep(1)


if __name__ == "__main__":
    main()
