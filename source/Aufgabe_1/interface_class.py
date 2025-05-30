import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch

class CNNInterface:
    def __init__(self, model, class_names, device, train_function, test_transform):
        """
        GUI-based interface for CNN training and testing.

        Args:
            model: Trained or initialized PyTorch model
            class_names: List of class labels
            device: torch.device (cuda or cpu)
            train_function: Function that handles training logic
            test_transform: torchvision.transforms for test images
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.train_function = train_function
        self.test_transform = test_transform

        self.model.to(self.device)
        self.model.eval()

    def launch(self):
        """Launches the GUI."""
        self.window = tk.Tk()
        self.window.title("CNN Trainer & Tester")
        self.window.geometry("300x150")

        label = tk.Label(self.window, text="Choose action", font=("Arial", 14))
        label.pack(pady=10)

        train_button = tk.Button(self.window, text="Train Model", command=self._on_train, width=20)
        train_button.pack(pady=5)

        test_button = tk.Button(self.window, text="Test with Image", command=self._on_test, width=20)
        test_button.pack(pady=5)

        self.window.mainloop()

    def _on_train(self):
        """Trigger training."""
        messagebox.showinfo("Training", "Training will start now...")
        self.train_function()
        messagebox.showinfo("Training", "Training completed!")

    def _on_test(self):
        """Load image and classify it."""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        img = Image.open(file_path).convert("RGB")
        input_tensor = self.test_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted = torch.argmax(output, 1).item()
            predicted_label = self.class_names[predicted]

        self._show_result_window(img, predicted_label)

    def _show_result_window(self, image, predicted_label):
        """Display image and prediction."""
        result_window = tk.Toplevel(self.window)
        result_window.title("Prediction Result")

        img_resized = image.resize((224, 224))
        img_tk = ImageTk.PhotoImage(img_resized)

        panel = tk.Label(result_window, image=img_tk)
        panel.image = img_tk
        panel.pack()

        result_label = tk.Label(result_window, text=f"Predicted Class: {predicted_label}", font=("Arial", 14))
        result_label.pack()
