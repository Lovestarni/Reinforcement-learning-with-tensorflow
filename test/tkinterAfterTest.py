import time
import tkinter as tk

def __writeText():
    text.insert(tk.END, str(time.time())+'\n')
    # root.after(1000, __writeText)  # again forever

root = tk.Tk()
text = tk.Text(root)
text.pack()
root.after(1000, __writeText)
root.mainloop()