#!/usr/local/bin/python3
#encoding=utf-8
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2


class App(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master, width=400, height=300)
		self.pack()
		self.min = 0
		self.max = 100
		self.ori_img = cv2.imread('./test_images/test_40.jpg')
		self.grey_img = cv2.cvtColor(self.ori_img, cv2.COLOR_RGB2GRAY)
		self.gaussianblur = cv2.GaussianBlur(self.grey_img, (3,3),0)
		self.img_canny = cv2.Canny(self.gaussianblur, 100 , 200)
		self.tkImage = ImageTk.PhotoImage(image=Image.fromarray(self.img_canny))
		self.canvas = tk.Canvas(self,width=900,height=600,bg = 'white')
		self.canvas.create_image((450,250),image = self.tkImage)
		self.canvas.pack()
		self.scale1 = tk.Scale(self, label='最小值',from_=0,to=100,orient=tk.HORIZONTAL,
			length=800,showvalue=1,tickinterval=10,resolution=10,command=self.print_selection1)
		self.scale1.pack()
		self.scale2 = tk.Scale(self, label='最大值',from_=100,to=250,orient=tk.HORIZONTAL,
			length=800,showvalue=1,tickinterval=10,resolution=10,command=self.print_selection2)
		self.scale2.pack()


	def print_selection1(self, V):
		#self.title.config(text=V)
		self.min = int(V)
		self.img_canny = cv2.Canny(self.gaussianblur, self.min, self.max)
		self.tkImage = ImageTk.PhotoImage(image=Image.fromarray(self.img_canny))
		self.canvas.create_image((450,250),image = self.tkImage)
		self.canvas.pack()
		self.update()
		self.after(1000)

	def print_selection2(self, V):
		self.max = int(V)
		self.img_canny = cv2.Canny(self.gaussianblur, self.min, self.max)
		self.tkImage = ImageTk.PhotoImage(image=Image.fromarray(self.img_canny))
		self.canvas.create_image((450,250),image = self.tkImage)
		self.canvas.pack()
		self.update()
		self.after(1000)

	def processEvent(self, event):
		pass
 
if __name__ == '__main__':
	root = tk.Tk()
	root.title('边缘检测调试程序')
	app = App(root)
	root.mainloop()