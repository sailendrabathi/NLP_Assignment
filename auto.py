# import sys
import argparse
from assign1 import *
import tkinter as tk

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store_true', default=False, help='take input from GUI')
args = parser.parse_args()

entry = None
pred_lbl = None
prob_lbl = None

def handle_button():
	global entry
	global pred_lbl
	global prob_lbl
	text = entry.get()
	pred, probs = NN.test_single(text)
	pred_lbl["text"] = "Prediction: " + str(pred[0])
	prob_lbl["text"] = "Probabilities: " + str(probs[0])


if __name__ == "__main__":
	val=main("train1.csv","gold_test.csv")
	print(val)
	if(args.input):
		window = tk.Tk()
		window.columnconfigure(0, minsize=500)
		window.rowconfigure([0, 1, 2, 3, 4], minsize=50)
		greeting = tk.Label(text="Enter input to test.")
		greeting.grid(row=0, column=0)
		entry = tk.Entry(width=100)
		entry.grid(row=1, column=0)
		button = tk.Button(text="Test", width=10, height = 1, bg="yellow", fg="black", command=handle_button)
		button.grid(row=2, column=0)
		pred_lbl = tk.Label(text="Prediction:")
		pred_lbl.grid(row=3, column=0)
		prob_lbl = tk.Label(text="Probabilities:")
		prob_lbl.grid(row=4, column=0)
		window.mainloop()     