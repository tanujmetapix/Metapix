from PIL import Image, ImageTk
import tkinter as tk
import cv2
import argparse
from pathlib import Path

from handlers import RootHandler, ImageHandler


def _on_close(root):
    print('[INFO] closing...')
    root.destroy()


if __name__ == '__main__':
    
    root = tk.Tk()
    root.title('Image Filter')
    root.wm_protocol('WM_DELETE_WINDOW', lambda: _on_close(root))
    # root.geometry('500x500')

    img = cv2.imread('image_for_filter.jpeg')

    img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img2 = ImageTk.PhotoImage(img2)

    orig = tk.Label(root, image=img2)
    orig.grid(row=0, column=0)
    panel = tk.Label(root, image=img2)
    panel.grid(row=0, column=1)

    btn = tk.Button(root, text='Save', bd=5,
                    command=lambda: img_handler.save_img())
    btn.grid(row=1, column=1)

    img_handler = ImageHandler(img, img, '/output')
    root_handler = RootHandler(panel)

    root_handler.bind_root(root, img_handler, init=True)
    root.bind('<Escape>', lambda e: _on_close(root))
    root.mainloop()
