import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config import engine, fuel_2026, hub
from models import ShotState

class ShotCalculatorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FRC Aimbot")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")

        self.sidebar = ctk.CTkFrame(self, width=350)
        self.sidebar.pack(side = "left", fill = "y", padx = 10, pady = 10)

        self.tabs = ctk.CTkTabview(self.sidebar)
        self.tabs.pack(fill = "both", expand = True)

        self.tabs.add("Robot")
        self.tabs.add("Game Piece")
        self.tabs.add("Environment")
        self.tabs.add("Targets")
        self.tabs.add("Sandbox")

        sandbox = self.tabs.tab("Sandbox")

        ctk.CTkLabel(sandbox, text="Target Distance (m)").pack(pady=(10, 0))
        self.dist_slider = ctk.CTkSlider(sandbox, from_=1.0, to=20.0, command=self.update_plot)
        self.dist_slider.pack(pady=5)
        self.dist_slider.set(5.0)

        ctk.CTkLabel(sandbox, text="Flywheel RPM").pack(pady=(10, 0))
        self.rpm_slider = ctk.CTkSlider(sandbox, from_=0.0, to=6000.0, command=self.update_plot)
        self.rpm_slider.pack(pady=5)
        self.rpm_slider.set(3000.0)

        ctk.CTkLabel(sandbox, text="Hood Angle (deg)").pack(pady=(10, 0))
        self.hood_slider = ctk.CTkSlider(sandbox, from_=0.0, to=45.0, command=self.update_plot)
        self.hood_slider.pack(pady=5)
        self.hood_slider.set(10.0)

        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(side = "right", fill = "both", expand = True, padx = 10, pady = 10)
        self.fig = plt.Figure(figsize = (8,8), facecolor = "#2b2b2b")
        self.ax = self.fig.add_subplot(111, projection = '3d')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors = 'white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill = "both", expand = True)

    def update_plot(self, event=None):
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        
        self.ax.set_xlim([0, 20])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 4])
        
        dist = self.dist_slider.get()
        rpm = self.rpm_slider.get()
        hood = self.hood_slider.get()
        
        state = ShotState(
            a_rad = 0, a_tan = 0, alpha = 0,
            v_rad = 0, v_tan = 0, omega = 0,
            pitch = 0, roll = 0, distance = 0
        )
        
        result = engine.simulate_shot(
            piece = fuel_2026, 
            state = state, 
            rpm = rpm, 
            hood_deg = hood, 
            aim_offset_rad = 0.0, 
            target_z = hub.height, 
            return_path = True
        )
        
        if result[0] is not None:
            lx, ly, path_x, path_y, path_z = result
            self.ax.plot(path_x, path_y, path_z, color="#00ffcc", linewidth=2.5)
            self.ax.scatter([dist], [0], [hub.height], color="red", s=100)
            
        self.canvas.draw()




if __name__ == "__main__":
    app = ShotCalculatorGUI()
    app.mainloop()
