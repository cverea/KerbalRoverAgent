import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        """Set up interactive plot for rover position tracking."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.vessel_positions_lat = []
        self.vessel_positions_lon = []
        self.vessel_line = None
        self.ax.set_xlabel('Longitude (°)')
        self.ax.set_ylabel('Latitude (°)')
        self.ax.set_title('Rover Position Tracking')
        self.ax.grid(True)
        
    def setup_plot(self, target_latlon, vessel_latlon):     
        
        # Plot target
        self.ax.plot(target_latlon[1], target_latlon[0], 'r*', markersize=15, label='Target')
        
        # Initialize vessel position plot
        self.vessel_positions_lat = [vessel_latlon[0]]
        self.vessel_positions_lon = [vessel_latlon[1]]
        self.vessel_line, = self.ax.plot(self.vessel_positions_lon, self.vessel_positions_lat, 'b-', linewidth=2, label='Rover Path')
        self.vessel_point, = self.ax.plot(vessel_latlon[1], vessel_latlon[0], 'bo', markersize=8, label='Rover')
        
        self.ax.legend()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.01)
        
    
    def update_plot(self, vessel_latlon):
        # Update vessel position plot
        vessel_lat = vessel_latlon[0]
        vessel_lon = vessel_latlon[1]
        self.vessel_positions_lat.append(vessel_lat)
        self.vessel_positions_lon.append(vessel_lon)
        self.vessel_line.set_data(self.vessel_positions_lon, self.vessel_positions_lat)
        self.vessel_point.set_data([vessel_lon], [vessel_lat])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    def save_plot(self, filename):
        self.fig.savefig(filename)